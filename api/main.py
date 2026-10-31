"""
API de FintechLab (FastAPI).

Ubicación en el proyecto: api/main.py

Endpoints:
  POST /chat                        -> conversa con el sistema (Supervisor + agentes)
  GET  /chat/{thread_id}/historial  -> la conversación guardada en el checkpointer
  POST /creditos/evaluar            -> evaluación directa (sin LLM) de una solicitud
  GET  /health                      -> estado del servicio

Documentación interactiva: http://localhost:8000/docs

Levantar en desarrollo, desde la raíz del proyecto:
    py -m uvicorn api.main:app --reload
"""

import logging
import os
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage

from agents.creditos.bcra import BCRA_MODO
from agents.creditos.scoring import ResultadoEvaluacion, SolicitudCredito, evaluar_credito
from agents.supervisor import CHAT_MODEL, construir_grafo
from api.checkpointer import abrir_checkpointer
from api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    HistorialResponse,
    MensajeHistorial,
    ToolUsada,
)

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fintechlab")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Al arrancar: abre el checkpointer y compila el grafo. Al apagar: cierra todo."""
    async with AsyncExitStack() as stack:
        checkpointer, nombre = await abrir_checkpointer(stack)
        app.state.grafo = construir_grafo(checkpointer=checkpointer)
        app.state.checkpointer = nombre
        yield


app = FastAPI(
    title="FintechLab Intelligence System",
    description=(
        "Asistente de una financiera ficticia: un Supervisor (LangGraph) deriva cada "
        "mensaje al agente de Atención (RAG sobre políticas) o al agente de Créditos "
        "(tools sobre la base interna y el BCRA)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def error_inesperado(request: Request, exc: Exception):
    log.exception("Error no controlado en %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Error interno. Revisá los logs."})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["sistema"])
async def health() -> HealthResponse:
    return HealthResponse(
        estado="ok", checkpointer=app.state.checkpointer, bcra_modo=BCRA_MODO, modelo=CHAT_MODEL
    )


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(body: ChatRequest) -> ChatResponse:
    thread_id = body.thread_id or uuid.uuid4().hex[:12]
    config = {
        "configurable": {"thread_id": thread_id},
        # Esto aparece en LangSmith: permite filtrar las trazas por conversación
        "run_name": "fintechlab_chat",
        "metadata": {"thread_id": thread_id},
        "tags": ["api"],
    }

    inicio = time.perf_counter()
    estado = await app.state.grafo.ainvoke({"messages": [HumanMessage(body.mensaje)]}, config)
    latencia = int((time.perf_counter() - inicio) * 1000)

    # Mensajes de ESTE turno: los posteriores al último mensaje del usuario
    turno = []
    for m in reversed(estado["messages"]):
        if isinstance(m, HumanMessage):
            break
        turno.append(m)
    tools = [
        ToolUsada(nombre=tc["name"], argumentos=tc["args"])
        for m in reversed(turno)
        if isinstance(m, AIMessage)
        for tc in m.tool_calls
    ]

    log.info("thread=%s agente=%s tools=%s %sms", thread_id, estado.get("agente"),
             [t.nombre for t in tools], latencia)

    return ChatResponse(
        thread_id=thread_id,
        respuesta=estado["messages"][-1].content,
        agente=estado.get("agente") or "directo",
        ruta=estado.get("ruta"),
        tools_usadas=tools,
        fuentes=estado.get("fuentes") or [],
        latencia_ms=latencia,
    )


@app.get("/chat/{thread_id}/historial", response_model=HistorialResponse, tags=["chat"])
async def historial(thread_id: str) -> HistorialResponse:
    snapshot = await app.state.grafo.aget_state({"configurable": {"thread_id": thread_id}})
    mensajes = snapshot.values.get("messages", []) if snapshot.values else []
    if not mensajes:
        raise HTTPException(status_code=404, detail="No existe una conversación con ese thread_id")

    salida = []
    for m in mensajes:
        if isinstance(m, HumanMessage):
            salida.append(MensajeHistorial(rol="usuario", contenido=m.content))
        elif isinstance(m, AIMessage) and m.content and not m.tool_calls:
            salida.append(MensajeHistorial(rol="asistente", contenido=m.content))
    return HistorialResponse(thread_id=thread_id, mensajes=salida)


@app.post("/creditos/evaluar", response_model=ResultadoEvaluacion, tags=["créditos"])
async def evaluar(solicitud: SolicitudCredito) -> ResultadoEvaluacion:
    """Aplica la política de créditos directamente, sin pasar por el LLM."""
    return await evaluar_credito(solicitud)
