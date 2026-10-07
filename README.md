# FastAPI Inventory and LLM Agent

[Documentacion en espanol](README.es.md)

The API stores inventory in `products.csv`. The CLI agent calls the inventory
API through the functions defined in `tools.json`. Conversation events are
appended to `conversation_log.csv` and survive between sessions.

## Manual Agent Loop

The agent loop is implemented manually in Python in `InventoryAgent.run()`:
observe user input, ask the LLM for the next action, execute tool calls, append
their results to the conversation, and repeat until a final response or the
iteration limit. Session memory and CSV event logging are also managed by our
own code.

No agent framework is used: no LangChain, LlamaIndex, AutoGen, or equivalent.
The OpenAI SDK only calls the LLM, `httpx` only sends HTTP requests, and
`jsonschema` only validates tool arguments. FastAPI serves the inventory API;
none of these libraries orchestrates the agent.

## Start Both Processes

Run these commands from the repository root, using the same Python environment.
**The API must be running before you start the agent.**

### 1. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Terminal 1: Start the API

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Wait for `Application startup complete` and keep this terminal running. Use a
single API worker because inventory is stored in a CSV file. Interactive API
documentation is available at http://localhost:8000/docs.

### 3. Terminal 2: Check the API and Start the Agent

First verify that the API responds successfully:

```bash
curl --fail --show-error http://localhost:8000/inventory
```

If the check fails, fix the API startup before continuing. Configure
`OPENAI_API_KEY` in this second terminal. In Bash, read it without displaying it
or including the secret in command history:

```bash
read -rsp "OpenAI API key: " OPENAI_API_KEY
printf '\n'
export OPENAI_API_KEY
python agent.py
```

Enter a request at the `Tu:` prompt; the agent prints its response and keeps
conversation context in memory until the session ends. Type `/salir` to exit.
For a single request instead:

```bash
python agent.py "Lista el inventario y muestra las alertas"
```

Optional settings: `OPENAI_MODEL` (default `gpt-4o-mini`), `INVENTORY_API_URL`
(default `http://localhost:8000`), and `--max-iterations` (default `10`). If you
change the API port, update both the check URL and `INVENTORY_API_URL` in the
agent terminal to match.

Keep the API running throughout the agent session. Once the agent has exited,
stop the API with Ctrl+C in Terminal 1. Never commit your API key or publish
conversation logs without reviewing their contents.

## Requirements

Make sure you have Python installed in your computer. We strongly recommend [installing Python through Pyenv ](https://4geeks.com/how-to/what-is-pyenv-and-how-to-install-pyenv) to avoid version conflicts in the future.

### Contributors

This template was built as part of the [4Geeks Python Resources](https://4geeks.com/technology/python) for learning at [4Geeks.com](https://4geeks.com) by [Alejandro Sanchez](https://twitter.com/alesanchezr) and [many other contributors](https://github.com/4GeeksAcademy/python-hello/graphs/contributors).
