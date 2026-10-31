"""
Pruebas de punta a punta del sistema, a través de la API (quedan trazadas en LangSmith).

Ubicación en el proyecto: scripts/probar_sistema.py

Recorre los caminos principales del grafo:
  1. Supervisor directo (pregunta de capacidad, sin LLM)
  2. Atención con RAG (pregunta cubierta por las políticas)
  3. Atención con guardrail (pregunta NO cubierta -> sin inventar)
  4. Créditos: solicitud aprobada (tools: evaluar_credito)
  5. Créditos: seguimiento en la misma conversación (memoria del Checkpointer)
  6. Créditos: derivación a analista humano (revision_manual)
  7. Créditos: rechazo automático por política
  8. Supervisor directo: tema fuera de alcance

Los CUITs se eligen solos: se evalúan los clientes de la base con el motor de
scoring (sin LLM) y se toma uno de cada resultado.

Con la API levantada, desde la raíz del proyecto:
    py -m scripts.probar_sistema
Con Docker:
    docker compose exec api python -m scripts.probar_sistema
"""

import asyncio
import json
import os
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv

from agents.creditos.db import listar_cuits_de_ejemplo
from agents.creditos.scoring import SolicitudCredito, evaluar_credito

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
SALIDA = Path("evidencia/resultados_pruebas.json")


async def elegir_cuits() -> dict[str, str]:
    """Un CUIT por cada decisión posible, para un préstamo de $500.000 a 12 meses."""
    elegidos: dict[str, str] = {}
    for cuit in await listar_cuits_de_ejemplo(100):
        r = await evaluar_credito(SolicitudCredito(cuit=cuit, monto=500_000, plazo_meses=12))
        elegidos.setdefault(r.decision, cuit)
    return elegidos


async def main() -> None:
    cuits = await elegir_cuits()
    print("CUITs elegidos:", cuits, "\n")

    lote = uuid.uuid4().hex[:6]  # para no mezclar con corridas anteriores
    hilo = lambda nombre: f"prueba-{lote}-{nombre}"  # noqa: E731

    casos = [
        ("1. Capacidad (directo)", hilo("capacidad"), "¿En qué me podés ayudar?"),
        ("2. FAQ con RAG", hilo("faq"), "¿Puedo cancelar mi préstamo antes de tiempo? ¿Me cobran algo?"),
        ("3. Guardrail sin contexto", hilo("guardrail"), "¿Cuál es el horario de las sucursales en Córdoba?"),
        ("4. Crédito aprobado", hilo("credito"),
         f"Hola, mi CUIT es {cuits.get('aprobado')}. Quiero un préstamo de 500000 a 12 meses."),
        ("5. Seguimiento (memoria)", hilo("credito"), "¿Y si lo pido a 24 meses?"),
        ("6. Revisión manual", hilo("revision"),
         f"Quiero pedir 500000 a 12 meses, mi CUIT es {cuits.get('revision_manual')}"),
        ("7. Rechazo automático", hilo("rechazo"),
         f"CUIT {cuits.get('rechazado')}, necesito 500000 en 12 cuotas"),
        ("8. Fuera de alcance", hilo("fuera"), "¿Me recomendás una serie para ver este finde?"),
    ]

    resultados = []
    async with httpx.AsyncClient(base_url=API_URL, timeout=120) as client:
        for titulo, thread_id, mensaje in casos:
            if "None" in mensaje:
                print(f"{titulo}: SALTEADO (no hay un cliente con ese resultado en la base)\n")
                continue
            resp = await client.post("/chat", json={"mensaje": mensaje, "thread_id": thread_id})
            resp.raise_for_status()
            r = resp.json()
            resultados.append({"caso": titulo, "mensaje": mensaje, **r})

            tools = ", ".join(t["nombre"] for t in r["tools_usadas"]) or "-"
            print(f"{titulo}\n  agente={r['agente']}  ruta={r['ruta']}  tools={tools}  {r['latencia_ms']} ms")
            print(f"  > {mensaje}\n  < {r['respuesta']}\n")

    SALIDA.parent.mkdir(exist_ok=True)
    SALIDA.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Resultados guardados en {SALIDA}")
    print("Las trazas están en LangSmith, proyecto:", os.getenv("LANGSMITH_PROJECT", "(sin configurar)"))


if __name__ == "__main__":
    asyncio.run(main())
