"""
RAG del agente de Atención: responde preguntas sobre las políticas de FintechLab.

Ubicación en el proyecto: agents/atencion/rag.py

Recuperación HÍBRIDA, en dos búsquedas que se combinan:
  1. Semántica: similitud coseno entre embeddings (entiende sinónimos:
     "pagar antes" ~ "cancelación anticipada").
  2. Por palabras clave: BM25 (no se le escapan términos exactos como
     "cheques", "AFIP" o "$2.000.000").
  Los dos rankings se fusionan con Reciprocal Rank Fusion (RRF).

Guardrail anti-alucinación: si ni el mejor fragmento se parece lo suficiente
a la pregunta (RAG_MIN_SIMILITUD), NO se llama al LLM: se devuelve una
respuesta de cautela fija. Así el modelo nunca "rellena" sin contexto.

Probar desde la raíz del proyecto (después de correr ingest.py):
    py -m agents.atencion.rag
"""

import asyncio
import json
import math
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import traceable

from agents.atencion.schemas import (
    FragmentoRecuperado,
    Fragmento,
    FuenteCitada,
    RespuestaAtencion,
)

load_dotenv()

INDICE_PATH = Path(os.getenv("INDICE_ATENCION_PATH", "data/indice_atencion.json"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_MIN_SIMILITUD = float(os.getenv("RAG_MIN_SIMILITUD", "0.40"))

RESPUESTA_SIN_CONTEXTO = (
    "No encuentro información sobre eso en las políticas de FintechLab, así que "
    "prefiero no responderte algo que podría ser incorrecto. Si querés, puedo "
    "derivar tu consulta a un analista."
)

PROMPT_RESPUESTA = """Sos el asistente de Atención al Cliente de FintechLab, una financiera argentina.
Respondés en español rioplatense, claro y breve (máximo 5 oraciones).

Reglas:
- Respondé SOLO con la información del CONTEXTO. No agregues datos que no estén ahí.
- Si el contexto no alcanza para responder, decilo y ofrecé derivar a un analista.
- Si la política indica que un caso requiere revisión de un analista, decilo explícitamente.
- No inventes montos, plazos ni tasas."""


# ---------------------------------------------------------------------------
# Índice (se carga una sola vez y queda en memoria)
# ---------------------------------------------------------------------------

_fragmentos: list[Fragmento] | None = None
_bm25: "BM25 | None" = None


def _leer_indice() -> list[Fragmento]:
    if not INDICE_PATH.exists():
        raise FileNotFoundError(
            f"No existe {INDICE_PATH}. Corré primero: py -m agents.atencion.ingest"
        )
    datos = json.loads(INDICE_PATH.read_text(encoding="utf-8"))
    return [Fragmento(**f) for f in datos["fragmentos"]]


async def _cargar_indice() -> tuple[list[Fragmento], "BM25"]:
    global _fragmentos, _bm25
    if _fragmentos is None:
        # Leer un archivo es E/S de disco: lo mandamos a un hilo para no bloquear
        _fragmentos = await asyncio.to_thread(_leer_indice)
        _bm25 = BM25([f.texto for f in _fragmentos])
    return _fragmentos, _bm25


# ---------------------------------------------------------------------------
# Búsqueda por palabras clave (BM25)
# ---------------------------------------------------------------------------

STOPWORDS = set("""a al algo como con cual cuales de del el ella en es esta este hay la las
le lo los me mi mis no o para pero por que se si sin su sus te tu un una uno y ya puedo
puede pueden quiero tengo""".split())


def tokenizar(texto: str) -> list[str]:
    """Minúsculas, sin acentos, sin palabras vacías."""
    sin_acentos = unicodedata.normalize("NFKD", texto.lower())
    sin_acentos = "".join(c for c in sin_acentos if not unicodedata.combining(c))
    return [p for p in re.findall(r"[a-z0-9]+", sin_acentos) if p not in STOPWORDS and len(p) > 1]


class BM25:
    """BM25 clásico (k1=1.5, b=0.75). Puntúa documentos por términos en común."""

    def __init__(self, documentos: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [Counter(tokenizar(d)) for d in documentos]
        self.largos = [sum(d.values()) for d in self.docs]
        self.largo_medio = sum(self.largos) / len(self.largos)
        self.k1, self.b = k1, b
        n = len(self.docs)
        df = Counter(t for d in self.docs for t in d)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def puntajes(self, consulta: str) -> list[float]:
        terminos = tokenizar(consulta)
        resultado = []
        for doc, largo in zip(self.docs, self.largos):
            s = 0.0
            for t in terminos:
                if t in doc:
                    tf = doc[t]
                    s += self.idf[t] * tf * (self.k1 + 1) / (
                        tf + self.k1 * (1 - self.b + self.b * largo / self.largo_medio)
                    )
            resultado.append(s)
        return resultado


# ---------------------------------------------------------------------------
# Recuperación híbrida
# ---------------------------------------------------------------------------

def _coseno(a: list[float], b: list[float]) -> float:
    prod = sum(x * y for x, y in zip(a, b))
    norma = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return prod / norma if norma else 0.0


def _rangos(puntajes: list[float]) -> list[int]:
    """Posición (0 = mejor) de cada documento según su puntaje."""
    orden = sorted(range(len(puntajes)), key=lambda i: puntajes[i], reverse=True)
    rango = [0] * len(puntajes)
    for posicion, i in enumerate(orden):
        rango[i] = posicion
    return rango


@traceable(run_type="retriever", name="recuperar_politicas")
async def recuperar(pregunta: str, k: int = RAG_TOP_K) -> list[FragmentoRecuperado]:
    fragmentos, bm25 = await _cargar_indice()

    vector = await OpenAIEmbeddings(model=EMBEDDING_MODEL).aembed_query(pregunta)
    similitudes = [_coseno(vector, f.embedding) for f in fragmentos]
    palabras = bm25.puntajes(pregunta)

    # RRF: cada búsqueda "vota" por posición. 60 es la constante estándar.
    r_sem, r_kw = _rangos(similitudes), _rangos(palabras)
    hibrido = [1 / (60 + r_sem[i] + 1) + 1 / (60 + r_kw[i] + 1) for i in range(len(fragmentos))]

    mejores = sorted(range(len(fragmentos)), key=lambda i: hibrido[i], reverse=True)[:k]
    return [
        FragmentoRecuperado(
            fuente=fragmentos[i].fuente,
            seccion=fragmentos[i].seccion,
            texto=fragmentos[i].texto,
            similitud=round(similitudes[i], 4),
            puntaje_hibrido=round(hibrido[i], 5),
        )
        for i in mejores
    ]


# ---------------------------------------------------------------------------
# Respuesta
# ---------------------------------------------------------------------------

@traceable(name="rag_atencion")
async def responder(pregunta: str) -> RespuestaAtencion:
    recuperados = await recuperar(pregunta)
    mejor = max((r.similitud for r in recuperados), default=0.0)

    # Guardrail: sin contexto relevante, no se llama al LLM
    if mejor < RAG_MIN_SIMILITUD:
        return RespuestaAtencion(respuesta=RESPUESTA_SIN_CONTEXTO, ruta="sin_contexto")

    contexto = "\n\n---\n\n".join(r.texto for r in recuperados)
    llm = ChatOpenAI(model=CHAT_MODEL)
    mensaje = await llm.ainvoke([
        SystemMessage(PROMPT_RESPUESTA),
        HumanMessage(f"CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
    ])

    return RespuestaAtencion(
        respuesta=mensaje.content,
        ruta="rag",
        fuentes=[FuenteCitada(fuente=r.fuente, seccion=r.seccion, similitud=r.similitud)
                 for r in recuperados],
    )


# ---------------------------------------------------------------------------
# Prueba manual
# ---------------------------------------------------------------------------

async def _main() -> None:
    preguntas = [
        "¿Puedo pagar mi préstamo antes de tiempo?",
        "¿Cuánto tarda la aprobación si pido 3 millones?",
        "¿Tienen tarjetas de crédito?",
        "¿Cuál es la capital de Francia?",   # fuera de tema: debería cortar el guardrail
    ]
    for p in preguntas:
        r = await responder(p)
        print(f"\nPREGUNTA: {p}\nRUTA: {r.ruta}")
        for f in r.fuentes:
            print(f"  fuente: {f.fuente} > {f.seccion} (similitud {f.similitud})")
        print(f"RESPUESTA: {r.respuesta}")


if __name__ == "__main__":
    asyncio.run(_main())
