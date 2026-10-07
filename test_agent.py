import copy
import csv
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAIError
from openai.types.chat import ChatCompletionMessage

from agent import AgentError, InventoryAgent, run_session


class LoggedAgentTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.log_path = Path(directory.name) / "conversation_log.csv"
        log_patch = patch("agent.CONVERSATION_LOG_FILE", self.log_path)
        log_patch.start()
        self.addCleanup(log_patch.stop)

    def read_log(self):
        with self.log_path.open(newline="", encoding="utf-8") as log_file:
            reader = csv.DictReader(log_file)
            self.assertEqual(reader.fieldnames, ["actor", "message", "tool_call", "timestamp"])
            return list(reader)


class ToolExecutionTests(LoggedAgentTestCase):
    def run_tool(
        self, name, arguments, status_code=200, data=None, network_error=False,
    ):
        requests = []
        snapshots = []
        responses = iter([
            ChatCompletionMessage(role="assistant", content="Consultando la API", tool_calls=[{
                "id": "call_inventory_1",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }]),
            ChatCompletionMessage(role="assistant", content="Resultado recibido"),
        ])

        def complete(**kwargs):
            snapshots.append(copy.deepcopy(kwargs["messages"]))
            return SimpleNamespace(choices=[
                SimpleNamespace(message=next(responses)),
            ])

        def handle_request(request):
            requests.append(request)
            if network_error:
                raise httpx.ConnectError("API no disponible", request=request)
            return httpx.Response(status_code, json=data)

        llm_client = Mock()
        llm_client.chat.completions.create.side_effect = complete
        with httpx.Client(
            base_url="http://inventory.test",
            transport=httpx.MockTransport(handle_request),
        ) as api_client:
            agent = InventoryAgent(llm_client, api_client)
            self.assertEqual(agent.run("Procesa esta peticion"), "Resultado recibido")

        self.assertEqual(len(snapshots), 2)
        self.assertEqual(snapshots[0][-1]["role"], "user")
        assistant_message, tool_message = snapshots[1][-2:]
        self.assertEqual(assistant_message["role"], "assistant")
        self.assertEqual(
            assistant_message["tool_calls"][0]["id"], "call_inventory_1",
        )
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(tool_message["tool_call_id"], "call_inventory_1")
        self.assertEqual(agent.messages[-1]["content"], "Resultado recibido")
        rows = self.read_log()[-5:]
        self.assertEqual(
            [row["actor"] for row in rows],
            ["user", "assistant", "assistant", "tool", "assistant"],
        )
        self.assertEqual(rows[0]["message"], "Procesa esta peticion")
        self.assertEqual(json.loads(rows[2]["tool_call"])["function"]["name"], name)
        self.assertEqual(json.loads(rows[2]["tool_call"])["id"], rows[3]["tool_call"])
        self.assertEqual(json.loads(rows[3]["message"]), json.loads(tool_message["content"]))
        self.assertEqual(rows[4]["message"], "Resultado recibido")
        return requests, json.loads(tool_message["content"])

    def test_tools_call_correct_endpoints_and_inject_results(self):
        product = {"id": 7, "name": "Arroz", "quantity": 3, "unit": "kg"}
        cases = [
            ("list_inventory", {}, "GET", "/inventory", None, {}, 200, [product]),
            (
                "create_inventory_product",
                {"name": "Arroz", "quantity": 3, "unit": "kg"},
                "POST", "/inventory",
                {"name": "Arroz", "quantity": 3, "unit": "kg"},
                {}, 201, product,
            ),
            (
                "adjust_inventory_stock", {"product_id": 7, "delta": -2},
                "PATCH", "/inventory/7", {"delta": -2}, {}, 200, product,
            ),
            (
                "get_inventory_alerts", {"threshold": 5}, "GET",
                "/inventory/alerts", None, {"threshold": "5"}, 200, [product],
            ),
            (
                "get_inventory_alerts", {}, "GET", "/inventory/alerts",
                None, {}, 200, [product],
            ),
        ]
        for name, arguments, method, path, body, params, status, data in cases:
            with self.subTest(name=name, arguments=arguments):
                requests, result = self.run_tool(name, json.dumps(arguments), status, data)
                self.assertEqual(len(requests), 1)
                request = requests[0]
                self.assertEqual(request.method, method)
                self.assertEqual(request.url.path, path)
                self.assertEqual(dict(request.url.params), params)
                if body is not None:
                    self.assertEqual(json.loads(request.content), body)
                else:
                    self.assertEqual(request.content, b"")
                self.assertEqual(result, {"ok": True, "status_code": status, "data": data})

    def test_api_errors_are_injected_into_context(self):
        for status, detail in ((404, "Producto no encontrado"), (409, "Stock insuficiente")):
            with self.subTest(status=status):
                requests, result = self.run_tool(
                    "adjust_inventory_stock", '{"product_id": 7, "delta": -2}',
                    status, {"detail": detail},
                )
                self.assertEqual(len(requests), 1)
                self.assertEqual(result, {
                    "ok": False, "status_code": status, "data": {"detail": detail},
                })

    def test_invalid_calls_return_errors_without_http_requests(self):
        cases = [
            ("unknown_tool", "{}"),
            ("list_inventory", "{invalid"),
            ("list_inventory", '{"extra": 1}'),
            ("create_inventory_product", "{}"),
            ("adjust_inventory_stock", '{"product_id": 7, "delta": true}'),
        ]
        for name, arguments in cases:
            with self.subTest(name=name, arguments=arguments):
                requests, result = self.run_tool(name, arguments)
                self.assertEqual(requests, [])
                self.assertFalse(result["ok"])
                self.assertIn("error", result)

    def test_network_errors_are_injected_into_context(self):
        requests, result = self.run_tool("list_inventory", "{}", network_error=True)
        self.assertEqual(len(requests), 1)
        self.assertFalse(result["ok"])
        self.assertIn("Fallo de red", result["error"])


class SessionTests(LoggedAgentTestCase):
    def test_final_response_ends_loop_without_executing_tools(self):
        for tool_calls in (None, []):
            with self.subTest(tool_calls=tool_calls):
                llm_client = Mock()
                api_client = Mock()
                message = ChatCompletionMessage(
                    role="assistant", content="Respuesta final", tool_calls=tool_calls,
                )
                llm_client.chat.completions.create.return_value = SimpleNamespace(
                    choices=[SimpleNamespace(message=message)],
                )
                agent = InventoryAgent(llm_client, api_client, max_iterations=1)

                self.assertEqual(agent.run("Hola"), "Respuesta final")
                llm_client.chat.completions.create.assert_called_once()
                self.assertEqual(api_client.mock_calls, [])
                self.assertEqual(agent.messages[-1]["content"], "Respuesta final")

    def test_cli_reads_input_and_prints_each_response(self):
        agent = Mock()
        agent.run.side_effect = ["Primera respuesta", "Segunda respuesta"]
        with patch(
            "builtins.input", side_effect=["", "Hola", "Otra pregunta", "/salir"],
        ) as read_input, patch("builtins.print") as print_output:
            run_session(agent)

        self.assertEqual(read_input.call_count, 4)
        self.assertTrue(all(call.args == ("Tu: ",) for call in read_input.call_args_list))
        self.assertEqual(
            [call.args[0] for call in agent.run.call_args_list],
            ["Hola", "Otra pregunta"],
        )
        print_output.assert_any_call("Agente: Primera respuesta")
        print_output.assert_any_call("Agente: Segunda respuesta")

    def test_cli_closes_cleanly_on_eof_or_interrupt(self):
        for interruption in (EOFError, KeyboardInterrupt):
            with self.subTest(interruption=interruption):
                agent = Mock()
                with patch("builtins.input", side_effect=interruption), patch(
                    "builtins.print",
                ) as print_output:
                    run_session(agent)
                agent.run.assert_not_called()
                print_output.assert_any_call("\nSesion finalizada.")


class LLMErrorTests(LoggedAgentTestCase):
    def test_http_errors_show_safe_actionable_diagnostics(self):
        cases = [
            (400, None, "Solicitud no valida"),
            (401, "invalid_api_key", "Clave de API invalida"),
            (403, None, "Sin permisos"),
            (404, "model_not_found", "Modelo no encontrado"),
            (429, "insufficient_quota", "Cuota de API insuficiente"),
            (429, "rate_limit_exceeded", "Limite de solicitudes"),
            (500, None, "servicio de OpenAI"),
        ]
        for status_code, code, expected in cases:
            with self.subTest(status_code=status_code, code=code):
                request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
                response = httpx.Response(status_code, request=request)
                error = APIStatusError(
                    "sensitive-provider-detail", response=response,
                    body={"code": code, "message": "sensitive-provider-detail"},
                )
                llm_client = Mock()
                llm_client.chat.completions.create.side_effect = error
                api_client = Mock()
                agent = InventoryAgent(llm_client, api_client)
                with self.assertRaises(AgentError) as caught:
                    agent.run("Hola")
                message = str(caught.exception)
                self.assertIn(f"HTTP {status_code}", message)
                self.assertIn(expected, message)
                self.assertNotIn("sensitive-provider-detail", message)
                self.assertIs(caught.exception.__cause__, error)
                self.assertEqual(api_client.mock_calls, [])

    def test_network_and_sdk_errors_have_distinct_diagnostics(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        cases = [
            (APITimeoutError(request=request), "no respondio a tiempo"),
            (APIConnectionError(request=request), "No se pudo conectar"),
            (OpenAIError("sensitive-provider-detail"), "Fallo del cliente"),
        ]
        for error, expected in cases:
            with self.subTest(error=type(error).__name__):
                llm_client = Mock()
                llm_client.chat.completions.create.side_effect = error
                agent = InventoryAgent(llm_client, Mock())
                with self.assertRaisesRegex(AgentError, expected) as caught:
                    agent.run("Hola")
                self.assertNotIn("sensitive-provider-detail", str(caught.exception))


class ConversationLogTests(LoggedAgentTestCase):
    def test_log_is_append_only_across_sessions(self):
        text = 'Hola, "inventario"\nSegunda linea'
        llm_client = Mock()
        llm_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=ChatCompletionMessage(
                role="assistant", content='Respuesta, "completa"\nOtro renglon',
            ))],
        )
        first = InventoryAgent(llm_client, Mock())
        first.run(text)
        existing_bytes = self.log_path.read_bytes()
        second = InventoryAgent(llm_client, Mock())
        second.run("Nueva sesion")

        self.assertTrue(self.log_path.read_bytes().startswith(existing_bytes))
        rows = self.read_log()
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["message"], text)
        self.assertEqual(rows[1]["message"], 'Respuesta, "completa"\nOtro renglon')
        self.assertEqual(rows[2]["message"], "Nueva sesion")
        self.assertEqual([row["actor"] for row in rows], ["user", "assistant"] * 2)
        for row in rows:
            self.assertEqual(datetime.fromisoformat(row["timestamp"]).utcoffset().total_seconds(), 0)
            self.assertEqual(row["tool_call"], "")
        self.assertEqual(len(second.messages), 3)

    def test_empty_log_gets_one_header(self):
        self.log_path.touch()
        agent = InventoryAgent(Mock(), Mock())
        agent.observe("Primer mensaje")
        agent.observe("Segundo mensaje")
        self.assertEqual(len(self.read_log()), 2)

    def test_log_failure_prevents_llm_and_tool_execution(self):
        llm_client = Mock()
        api_client = Mock()
        agent = InventoryAgent(llm_client, api_client)
        with patch("pathlib.Path.open", side_effect=PermissionError("No permitido")):
            with self.assertRaisesRegex(AgentError, "registro de conversacion"):
                agent.run("Consulta el inventario")
        llm_client.chat.completions.create.assert_not_called()
        self.assertEqual(api_client.mock_calls, [])


if __name__ == "__main__":
    unittest.main()