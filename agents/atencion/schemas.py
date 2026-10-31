"""
Schemas Pydantic del agente de Atención (RAG).

Ubicación en el proyecto: agents/atencion/schemas.py
"""

from typing import Literal

from pydantic import BaseModel, Field


class Fragmento(BaseModel):
    """Un pedazo de una política, listo para buscar. Se guarda en el índice."""

    id: str
    fuente: str = Field(..., description="Archivo de origen, ej. politica_creditos.md")
    seccion: str = Field(..., description="Título de la sección (metadato)")
    texto: str
    embedding: list[float] = Field(default_factory=list, repr=False)


class FragmentoRecuperado(BaseModel):
    """Un fragmento devuelto por la búsqueda, con sus puntajes (para auditar)."""

    fuente: str
    seccion: str
    texto: str
    similitud: float = Field(..., description="Similitud coseno con la pregunta (0 a 1)")
    puntaje_hibrido: float = Field(..., description="Puntaje final combinado (RRF)")


RutaAtencion = Literal["rag", "sin_contexto"]


class FuenteCitada(BaseModel):
    fuente: str
    seccion: str
    similitud: float


class RespuestaAtencion(BaseModel):
    """Lo que devuelve el agente de Atención."""

    respuesta: str
    ruta: RutaAtencion = Field(
        ..., description="'rag' si respondió con contexto; 'sin_contexto' si cortó el guardrail"
    )
    fuentes: list[FuenteCitada] = []
