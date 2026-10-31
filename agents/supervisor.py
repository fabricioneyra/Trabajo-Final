"""
Supervisor de FintechLab: el grafo principal del sistema (LangGraph).

Ubicación en el proyecto: agents/supervisor.py

El Supervisor recibe cada mensaje del usuario y decide quién lo atiende:

                       ┌──────────► atencion ─────┐
    START ─► supervisor ┼──────────► creditos ────┼─► END
                       └── directo (responde él) ─┘

  - atencion: preguntas generales sobre FintechLab y sus políticas (RAG)
  - creditos: todo lo de un cliente puntual o una solicitud de préstamo (agente con tools)
  - directo:  saludos, "¿en qué me podés ayudar?" y temas fuera de alcance

El agente de Créditos es a su vez un grafo (subgrafo) con su propio ciclo de tools.

La memoria de cada conversación la guarda el checkpointer que se le pasa a
construir_grafo() (Redis en producción). Cada conversación se identifica
con un thread_id.

Probar desde la raíz del proyecto (chat por consola):
    py -m agents.supervisor
"""

import asyncio
import os
import re
import unicodedata
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from agents.atencion.rag import responder as responder_atencion
from agents.creditos.agent import construir_agente as construir_agente_creditos

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

Destino = Literal["atencion", "creditos", "directo"]


# ---------------------------------------------------------------------------
# Estado compartido del grafo
# ---------------------------------------------------------------------------

class EstadoSistema(MessagesState):
    """Además del historial de mensajes, guardamos quién atendió el último turno."""

    agente: str | None
    ruta: str | None
    fuentes: list[dict]


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

class Ruteo(BaseModel):
    """Salida estructurada del Supervisor (validada con Pydantic)."""

    destino: Destino = Field(..., description="Quién atiende el mensaje")
    motivo: str = Field(..., description="Por qué elegiste ese destino, en una frase")
    respuesta_directa: str | None = Field(
        None, description="Solo si destino='directo': la respuesta para el usuario"
    )


PROMPT_SUPERVISOR = """Sos el Supervisor del asistente virtual de FintechLab, una financiera argentina
que ofrece préstamos personales. Tu trabajo es decidir quién atiende el ÚLTIMO mensaje del usuario:

- "atencion": preguntas generales sobre FintechLab y sus políticas: requisitos, plazos, tasas en
  general, documentación, tiempos de aprobación, cancelación anticipada, qué pasa si me rechazan,
  productos que ofrece. No requieren datos de una persona puntual.
- "creditos": todo lo que involucra a un cliente puntual o una solicitud concreta: pedir o
  simular un préstamo, evaluar si le dan un crédito, consultar su historial o su situación en el
  BCRA, o cuando el usuario da un CUIT, un monto o un plazo. También los mensajes que CONTINÚAN
  una conversación de créditos (ej. "¿y a 24 meses?").
- "directo": saludos, agradecimientos, despedidas, y temas que no tienen nada que ver con
  FintechLab. En ese caso escribí vos la respuesta en "respuesta_directa": breve, en español
  rioplatense; si el tema es ajeno, explicá con amabilidad que solo podés ayudar con préstamos
  personales de FintechLab."""

RESPUESTA_CAPACIDAD = (
    "Soy el asistente virtual de FintechLab. Puedo ayudarte con:\n"
    "- Consultas sobre nuestras políticas: requisitos, plazos, tasas, documentación.\n"
    "- Evaluar una solicitud de préstamo personal (necesito tu CUIT, el monto y el plazo).\n"
    "- Consultar tu historial con nosotros y tu situación en el BCRA.\n"
    "¿Por dónde empezamos?"
)


def _normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def es_pregunta_de_capacidad(texto: str) -> bool:
    """Detecta "¿en qué me podés/puedes ayudar?", "¿qué hacés?", "¿quién sos?" sin gastar LLM.

    Heurística de combinación: un verbo modal + un verbo de acción, en cualquier
    orden (cubre voseo y tuteo), más algunas frases fijas.
    """
    t = _normalizar(texto)
    modales = ("podes", "puedes", "sabes", "haces")
    acciones = ("ayudar", "hacer", "servir")
    if any(m in t for m in modales) and any(a in t for a in acciones) and len(t) < 60:
        return True
    return bool(re.search(r"\b(quien|que) (sos|eres)\b", t))


def _historial_como_texto(mensajes: list, ultimos: int = 6) -> str:
    """Solo los mensajes de texto (sin llamadas a tools), para que el Supervisor tenga contexto."""
    lineas = []
    for m in mensajes:
        if isinstance(m, HumanMessage):
            lineas.append(f"Usuario: {m.content}")
        elif isinstance(m, AIMessage) and m.content and not m.tool_calls:
            lineas.append(f"Asistente: {m.content}")
    return "\n".join(lineas[-ultimos:])


async def nodo_supervisor(state: EstadoSistema) -> dict:
    ultimo = state["messages"][-1].content

    # Camino rápido: preguntas de capacidad, sin LLM
    if es_pregunta_de_capacidad(ultimo):
        return {
            "agente": "directo",
            "ruta": "capacidad",
            "fuentes": [],
            "messages": [AIMessage(RESPUESTA_CAPACIDAD, name="supervisor")],
        }

    llm = ChatOpenAI(model=CHAT_MODEL).with_structured_output(Ruteo, method="function_calling")
    ruteo: Ruteo = await llm.ainvoke([
        SystemMessage(PROMPT_SUPERVISOR),
        HumanMessage(f"CONVERSACIÓN RECIENTE:\n{_historial_como_texto(state['messages'])}\n\n"
                     f"ÚLTIMO MENSAJE DEL USUARIO: {ultimo}"),
    ])

    if ruteo.destino == "directo":
        texto = ruteo.respuesta_directa or RESPUESTA_CAPACIDAD
        return {
            "agente": "directo",
            "ruta": "directo",
            "fuentes": [],
            "messages": [AIMessage(texto, name="supervisor")],
        }

    return {"agente": ruteo.destino, "ruta": None, "fuentes": []}


def elegir_destino(state: EstadoSistema) -> str:
    return END if state["agente"] == "directo" else state["agente"]


# ---------------------------------------------------------------------------
# Agente de Atención (envuelve el RAG como nodo)
# ---------------------------------------------------------------------------

async def nodo_atencion(state: EstadoSistema) -> dict:
    resultado = await responder_atencion(state["messages"][-1].content)
    return {
        "ruta": resultado.ruta,
        "fuentes": [f.model_dump() for f in resultado.fuentes],
        "messages": [AIMessage(resultado.respuesta, name="atencion")],
    }


# ---------------------------------------------------------------------------
# Grafo
# ---------------------------------------------------------------------------

def construir_grafo(checkpointer=None):
    grafo = StateGraph(EstadoSistema)
    grafo.add_node("supervisor", nodo_supervisor)
    grafo.add_node("atencion", nodo_atencion)
    # Un grafo compilado se puede usar como nodo: es el subgrafo de Créditos
    grafo.add_node("creditos", construir_agente_creditos())

    grafo.add_edge(START, "supervisor")
    grafo.add_conditional_edges(
        "supervisor", elegir_destino, {"atencion": "atencion", "creditos": "creditos", END: END}
    )
    grafo.add_edge("atencion", END)
    grafo.add_edge("creditos", END)

    return grafo.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Chat por consola (memoria en RAM; la API usa Redis)
# ---------------------------------------------------------------------------

async def _chat() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    grafo = construir_grafo(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "consola"}}
    print("FintechLab (Supervisor). Escribí 'salir' para terminar.\n")
    while True:
        texto = input("Vos: ").strip()
        if texto.lower() in {"salir", "exit", "quit"}:
            break
        if not texto:
            continue
        estado = await grafo.ainvoke({"messages": [HumanMessage(texto)]}, config)
        print(f"[{estado['agente']}] {estado['messages'][-1].content}\n")


if __name__ == "__main__":
    asyncio.run(_chat())
