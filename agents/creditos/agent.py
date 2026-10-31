r"""
Agente de Créditos de FintechLab (LangGraph).

Ubicación en el proyecto: agents/creditos/agent.py

El grafo tiene dos nodos y un ciclo:

    START -> modelo --(¿pidió una tool?)--> tools -> modelo -> ... -> END
                    \--(no, ya respondió)--> END

  - "modelo": el LLM lee la conversación y decide si responde o pide una tool.
  - "tools":  ejecuta las tools que pidió el LLM y le devuelve los resultados.

El ciclo se repite hasta que el LLM tiene todo lo que necesita y responde.
Este patrón se llama ReAct (Reason + Act).

Probar desde la raíz del proyecto (chat por consola):
    py -m agents.creditos.agent
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agents.creditos.tools import TOOLS_CREDITOS

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """Sos el asistente de Créditos de FintechLab, una financiera argentina.
Hablás en español rioplatense, de forma clara, breve y profesional.

Cómo trabajás:
- Para cualquier consulta sobre un cliente puntual, necesitás su CUIT. Si no lo dio, pedíselo.
- Para evaluar un préstamo necesitás CUIT, monto y plazo. Si falta alguno, pedilo antes de evaluar.
- La decisión de un préstamo la toma SIEMPRE la tool evaluar_credito. Vos nunca aprobás,
  rechazás ni prometés nada por tu cuenta: comunicás el resultado y explicás el motivo.
- Si el resultado es "revision_manual", explicá que un analista va a revisar el caso.
- Nunca inventes datos. Si una tool no encuentra información o falla, decilo.
- Solo atendés temas de créditos. Si te preguntan otra cosa, decí que no es tu área.
- Al comunicar una evaluación, contá la decisión, el motivo y, si fue aprobada, la tasa anual y la
  cuota. No muestres los chequeos internos (edad, tope de ingreso, etc.) salvo que te los pidan.
- Escribí los montos en formato argentino: $500.000 (punto para miles)."""


def construir_agente(checkpointer=None):
    """Arma y compila el grafo. El checkpointer guarda la memoria de cada conversación."""

    # bind_tools: le "muestra" al LLM el menú de tools (nombre, descripción, schema)
    llm = ChatOpenAI(model=CHAT_MODEL).bind_tools(TOOLS_CREDITOS)

    async def nodo_modelo(state: MessagesState) -> dict:
        mensajes = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
        respuesta = await llm.ainvoke(mensajes)
        return {"messages": [respuesta]}  # se AGREGA al historial, no lo reemplaza

    grafo = StateGraph(MessagesState)
    grafo.add_node("modelo", nodo_modelo)
    grafo.add_node("tools", ToolNode(TOOLS_CREDITOS))

    grafo.add_edge(START, "modelo")
    # tools_condition mira el último mensaje: si tiene tool_calls va a "tools", si no a END
    grafo.add_conditional_edges("modelo", tools_condition)
    grafo.add_edge("tools", "modelo")

    return grafo.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Chat por consola
# ---------------------------------------------------------------------------

async def _chat() -> None:
    # InMemorySaver: memoria en RAM (se pierde al cerrar). En el paso 4 va Redis.
    agente = construir_agente(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "prueba-consola"}}

    print("Agente de Créditos de FintechLab. Escribí 'salir' para terminar.\n")
    while True:
        texto = input("Vos: ").strip()
        if texto.lower() in {"salir", "exit", "quit"}:
            break
        if not texto:
            continue

        estado = await agente.ainvoke({"messages": [HumanMessage(texto)]}, config)

        # Mostramos qué tools usó en ESTE turno (los mensajes después del último humano)
        ultimos = []
        for m in reversed(estado["messages"]):
            if isinstance(m, HumanMessage):
                break
            ultimos.append(m)
        for m in reversed(ultimos):
            if isinstance(m, AIMessage) and m.tool_calls:
                for tc in m.tool_calls:
                    print(f"   [tool] {tc['name']}({tc['args']})")

        print(f"Agente: {estado['messages'][-1].content}\n")


if __name__ == "__main__":
    asyncio.run(_chat())
