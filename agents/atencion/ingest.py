"""
Indexación de las políticas para el RAG del agente de Atención.

Ubicación en el proyecto: agents/atencion/ingest.py

Pasos:
  1. Lee los .md de data/policies/
  2. Los corta en fragmentos: uno por sección ("## ..."), con metadatos
     (archivo de origen y título de la sección)
  3. Calcula el embedding de cada fragmento (async, en un solo pedido)
  4. Guarda todo en data/indice_atencion.json

Es un paso de PREPARACIÓN: se corre una vez (y cada vez que cambien las políticas).

Decisión de diseño: el corpus son 2 documentos cortos (~15 fragmentos), así
que guardamos el índice en un JSON y buscamos en memoria. Una base vectorial
(Chroma, Pinecone, pgvector) agregaría un servicio más sin ganar nada a esta
escala. El código de búsqueda (rag.py) está aislado: cambiar a una base
vectorial solo toca ese archivo.

Correr desde la raíz del proyecto:
    py -m agents.atencion.ingest
"""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from agents.atencion.schemas import Fragmento

load_dotenv()

POLICIES_DIR = Path(os.getenv("POLICIES_DIR", "data/policies"))
INDICE_PATH = Path(os.getenv("INDICE_ATENCION_PATH", "data/indice_atencion.json"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def fragmentar(nombre_archivo: str, contenido: str) -> list[Fragmento]:
    """Corta un markdown en una sección por fragmento.

    Cortar por sección (y no por cantidad fija de caracteres) mantiene juntas
    las ideas que van juntas: cada regla queda con su título.
    """
    titulo_doc = ""
    fragmentos: list[Fragmento] = []
    seccion_actual: str | None = None
    lineas: list[str] = []

    def cerrar_seccion() -> None:
        if seccion_actual and any(l.strip() for l in lineas):
            cuerpo = "\n".join(lineas).strip()
            fragmentos.append(Fragmento(
                id=f"{nombre_archivo}#{len(fragmentos) + 1}",
                fuente=nombre_archivo,
                seccion=seccion_actual,
                # El título del documento y de la sección van dentro del texto:
                # mejoran el embedding y le dan contexto al LLM
                texto=f"{titulo_doc} — {seccion_actual}\n{cuerpo}",
            ))

    for linea in contenido.splitlines():
        if linea.startswith("# "):
            titulo_doc = linea[2:].strip()
        elif linea.startswith("## "):
            cerrar_seccion()
            seccion_actual = linea[3:].strip()
            lineas = []
        else:
            lineas.append(linea)
    cerrar_seccion()
    return fragmentos


async def indexar() -> list[Fragmento]:
    fragmentos: list[Fragmento] = []
    for archivo in sorted(POLICIES_DIR.glob("*.md")):
        fragmentos.extend(fragmentar(archivo.name, archivo.read_text(encoding="utf-8")))

    if not fragmentos:
        raise SystemExit(f"No encontré archivos .md en {POLICIES_DIR}")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectores = await embeddings.aembed_documents([f.texto for f in fragmentos])
    for fragmento, vector in zip(fragmentos, vectores):
        fragmento.embedding = vector

    INDICE_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDICE_PATH.write_text(
        json.dumps(
            {"modelo": EMBEDDING_MODEL, "fragmentos": [f.model_dump() for f in fragmentos]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return fragmentos


if __name__ == "__main__":
    resultado = asyncio.run(indexar())
    print(f"Índice creado en {INDICE_PATH} con {len(resultado)} fragmentos:")
    for f in resultado:
        print(f"  - {f.fuente:<26} {f.seccion}")
