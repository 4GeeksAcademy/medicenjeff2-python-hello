# Inventario con FastAPI y agente LLM

La API guarda el inventario en `products.csv`. El agente usa las cuatro funciones
definidas en `tools.json` para consultar y modificar el inventario mediante HTTP.

## Arranque

Ejecuta los comandos desde la raíz del repositorio, usando el mismo entorno de
Python en ambas terminales. **La API debe estar en ejecución antes de iniciar
el agente.**

### 1. Instalar dependencias

```bash
python -m pip install -r requirements.txt
```

### 2. Terminal 1: iniciar la API

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Espera el mensaje `Application startup complete` y deja esta terminal abierta.
Usa una sola instancia de la API porque el inventario se guarda en CSV. La
documentación interactiva está en `http://localhost:8000/docs`.

### 3. Terminal 2: comprobar la API e iniciar el agente

Antes de iniciar el agente, comprueba que la API responde correctamente:

```bash
curl --fail --show-error http://localhost:8000/inventory
```

Si falla, resuelve el arranque de la API antes de continuar. Configura
`OPENAI_API_KEY` en esta segunda terminal sin guardarla en el repositorio.
En Bash puedes introducirla sin mostrarla ni incluirla en el historial de comandos:

```bash
read -rsp "Clave de OpenAI: " OPENAI_API_KEY
printf '\n'
export OPENAI_API_KEY
python agent.py
```

Escribe tu petición en el prompt `Tu:`; el agente imprime su respuesta y espera
la siguiente petición. Para ejecutar una sola petición en lugar de una sesión:

```bash
python agent.py "Lista el inventario y muestra los productos con stock bajo"
python agent.py "Crea un producto llamado Arroz con 12 kg, retira 4 kg y muestra las alertas"
```

El agente conserva en memoria las peticiones, respuestas y resultados de tools
entre turnos. Escribe `/salir`, pulsa Ctrl+C o cierra la entrada para terminar.
El contexto de la sesión se pierde al cerrarla, pero los eventos se registran en
`conversation_log.csv` y el inventario permanece en `products.csv`. El registro
no se carga automáticamente como contexto al iniciar una sesión nueva.

Mantén la API activa durante toda la sesión. Al terminar el agente, detén la API
con Ctrl+C en la terminal 1. Si cambias el puerto de la API, ajusta también la URL
de comprobación y `INVENTORY_API_URL` en la terminal del agente.

## Registro de conversación

`conversation_log.csv` se crea junto a `agent.py` con las columnas
`actor,message,tool_call,timestamp`. Se abre siempre en modo de adición: no se
sobrescribe y el encabezado solo se escribe cuando el archivo está vacío.

- Mensaje del usuario: `actor=user`, texto en `message`.
- Respuesta del agente, intermedia o final: `actor=assistant`, texto en `message`.
- Llamada a tool: `actor=assistant`, definición JSON con ID, nombre y argumentos en `tool_call`.
- Resultado de tool, incluidos errores: `actor=tool`, resultado JSON en `message` e ID de la llamada en `tool_call`.

Cada fila tiene un timestamp UTC en formato ISO 8601. El módulo `csv` conserva
correctamente comas, comillas y saltos de línea. Si no puede escribir el registro,
el agente devuelve un error en lugar de continuar silenciosamente.

El registro puede contener información sensible del usuario y del inventario;
protege su acceso y no lo publiques sin revisar su contenido.

Configuración opcional:

- `OPENAI_MODEL`: modelo con soporte de function calling; por defecto `gpt-4o-mini`.
- `INVENTORY_API_URL`: dirección de la API; por defecto `http://localhost:8000`.
- `--max-iterations`: máximo de consultas al LLM por petición; por defecto `10`.

```bash
python agent.py "Consulta las alertas con umbral 5" --max-iterations 6
```

## Bucle del agente

El bucle se implementa manualmente en Python en `InventoryAgent.run()`, sin
LangChain, LlamaIndex, AutoGen ni ningún otro framework de agentes. El historial,
la ejecución de tools, las iteraciones y el registro CSV los controla nuestro
propio código.

El SDK de OpenAI solo llama al LLM, `httpx` solo realiza peticiones HTTP y
`jsonschema` solo valida los argumentos. FastAPI sirve la API de inventario;
ninguna de estas bibliotecas orquesta el agente.

1. **Observar:** incorpora la petición del usuario al historial.
2. **Pensar:** envía el historial y las definiciones de tools al LLM.
3. **Actuar:** valida los argumentos con JSON Schema y ejecuta las llamadas HTTP permitidas.
4. **Actualizar:** añade los resultados y errores al historial con su `tool_call_id`.
5. **Repetir:** consulta nuevamente al LLM hasta obtener texto final o alcanzar el límite.

`InventoryAgent` conserva el historial durante su vida útil. Cada ejecución de
la CLI inicia una sesión nueva. Los errores de API se devuelven al modelo para
que pueda explicar el resultado. En modo interactivo, los errores del LLM y el
límite de iteraciones se muestran y permiten introducir otra petición sin
borrar el historial. En modo de petición única, terminan la ejecución con un
código de salida distinto de cero.

Las operaciones ya ejecutadas no se revierten al alcanzar el límite. Ante un
fallo de red, verifica el inventario antes de repetir una escritura, ya que su
resultado podría ser incierto. No expongas esta API sin autenticación a redes
no confiables.

## Requisitos

Asegúrate de tener Python instalado en tu computadora. Te recomendamos encarecidamente [instalar Python a través de Pyenv](https://4geeks.com/es/how-to/que-es-pyenv-y-como-instalar-pyenv) para evitar conflictos de versiones en el futuro.

### Contribuidores

Esta plantilla fue creada como parte de los [Recursos de Python de 4Geeks](https://4geeks.com/es/technology/python) para el aprendizaje en [4Geeks.com](https://4geeks.com) por [Alejandro Sanchez](https://twitter.com/alesanchezr) y [muchos otros contribuyentes](https://github.com/4GeeksAcademy/python-hello/graphs/contributors).
