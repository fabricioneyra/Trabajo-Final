"""
Schemas Pydantic de la API: todo lo que entra y sale por HTTP se valida acá.

Ubicación en el proyecto: api/schemas.py
"""

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    mensaje: str = Field(..., min_length=1, max_length=2000, examples=["¿Puedo cancelar mi préstamo antes?"])
    thread_id: str | None = Field(
        None,
        min_length=3,
        max_length=64,
        pattern=r"^[A-Za-z0-9_\-]+$",
        description="Id de la conversación. Si no se envía, se crea una nueva.",
    )


class ToolUsada(BaseModel):
    nombre: str
    argumentos: dict


class FuenteRAG(BaseModel):
    fuente: str
    seccion: str
    similitud: float


class ChatResponse(BaseModel):
    thread_id: str
    respuesta: str
    agente: Literal["atencion", "creditos", "directo"] = Field(
        ..., description="Quién respondió este turno"
    )
    ruta: str | None = Field(None, description="Camino dentro del agente (rag, sin_contexto, capacidad...)")
    tools_usadas: list[ToolUsada] = []
    fuentes: list[FuenteRAG] = []
    latencia_ms: int


class MensajeHistorial(BaseModel):
    rol: Literal["usuario", "asistente"]
    contenido: str


class HistorialResponse(BaseModel):
    thread_id: str
    mensajes: list[MensajeHistorial]


class HealthResponse(BaseModel):
    estado: Literal["ok"]
    checkpointer: Literal["redis", "sqlite"]
    bcra_modo: str
    modelo: str
