"""
Tools del agente de Créditos.

Ubicación en el proyecto: agents/creditos/tools.py

Una "tool" es una función común con tres cosas extra que lee el LLM:
  - un nombre        -> cómo la llama
  - una descripción  -> CUÁNDO conviene usarla (el LLM decide en base a esto)
  - un schema        -> QUÉ argumentos le tiene que pasar (Pydantic)

El LLM nunca ejecuta código: elige una tool y sus argumentos, y LangGraph
la ejecuta por él y le devuelve el resultado.

Probar desde la raíz del proyecto (después de correr seed.py):
    py -m agents.creditos.tools
"""

import asyncio

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError

from agents.creditos.bcra import ConsultaBCRAInput, consultar_bcra
from agents.creditos.db import buscar_cliente, listar_cuits_de_ejemplo
from agents.creditos.schemas import Cuit
from agents.creditos.scoring import SolicitudCredito, evaluar_credito


def _error_de_argumentos(e: ValidationError) -> str:
    """Lo que ve el LLM si pasa argumentos inválidos: el motivo y qué hacer."""
    return (
        f"Argumento inválido: {e.errors()[0]['msg']}. "
        "Pedile al usuario un CUIT/CUIL de 11 dígitos."
    )


# ---------------------------------------------------------------------------
# Tool 1: buscar_cliente (datos internos de FintechLab)
# ---------------------------------------------------------------------------

class BuscarClienteInput(BaseModel):
    cuit: Cuit = Field(..., description="CUIT/CUIL de 11 dígitos, con o sin guiones")


async def _buscar_cliente_tool(cuit: str) -> dict:
    resultado = await buscar_cliente(cuit)
    if resultado is None:
        return {
            "encontrado": False,
            "mensaje": "El CUIT no está registrado en FintechLab.",
        }
    return {
        "encontrado": True,
        # mode="json": convierte fechas a texto para que el LLM las pueda leer
        "cliente": resultado.cliente.model_dump(mode="json"),
        "prestamos": [p.model_dump(mode="json") for p in resultado.prestamos],
        "resumen": {
            "tiene_mora": resultado.tiene_mora,
            "prestamos_pagados": resultado.prestamos_pagados,
            "es_cliente_nuevo": resultado.es_cliente_nuevo,
        },
    }


buscar_cliente_tool = StructuredTool.from_function(
    coroutine=_buscar_cliente_tool,
    name="buscar_cliente",
    description=(
        "Busca un cliente en la base interna de FintechLab por CUIT/CUIL y devuelve "
        "sus datos (ingreso mensual, antigüedad laboral, fecha de nacimiento) y su "
        "historial de préstamos con nosotros, con un resumen (si tuvo mora, cuántos "
        "pagó, si es cliente nuevo). Usala primero ante cualquier consulta sobre un "
        "cliente puntual. Si 'encontrado' es false, el CUIT no está registrado: "
        "no inventes datos del cliente."
    ),
    args_schema=BuscarClienteInput,
    handle_validation_error=_error_de_argumentos,
)


# ---------------------------------------------------------------------------
# Tool 2: consultar_bcra (sistema financiero completo)
# ---------------------------------------------------------------------------

async def _consultar_bcra_tool(cuit: str) -> dict:
    resultado = await consultar_bcra(cuit)
    return resultado.model_dump()


consultar_bcra_tool = StructuredTool.from_function(
    coroutine=_consultar_bcra_tool,
    name="consultar_bcra",
    description=(
        "Consulta la Central de Deudores del BCRA para un CUIT/CUIL y devuelve "
        "la peor situación crediticia (1 = normal, 2 o más = alerta, 5 o más = grave), "
        "la deuda total en miles de pesos y el detalle por entidad. "
        "Usala cuando necesites conocer el comportamiento del cliente en TODO el "
        "sistema financiero, no solo en FintechLab. "
        "Si el campo 'error' viene con texto, el BCRA no pudo consultarse: "
        "no inventes datos, informalo y sugerí derivar a un analista."
    ),
    args_schema=ConsultaBCRAInput,
    handle_validation_error=_error_de_argumentos,
)


# ---------------------------------------------------------------------------
# Tool 3: evaluar_credito (la decisión, según la política)
# ---------------------------------------------------------------------------

async def _evaluar_credito_tool(cuit: str, monto: float, plazo_meses: int) -> dict:
    solicitud = SolicitudCredito(cuit=cuit, monto=monto, plazo_meses=plazo_meses)
    resultado = await evaluar_credito(solicitud)
    return resultado.model_dump()


evaluar_credito_tool = StructuredTool.from_function(
    coroutine=_evaluar_credito_tool,
    name="evaluar_credito",
    description=(
        "Evalúa una solicitud de préstamo contra la política de créditos de FintechLab "
        "y devuelve la decisión: 'aprobado' (con tasa y cuota estimadas), 'rechazado' "
        "o 'revision_manual' (lo revisa un analista), con el motivo. Requiere CUIT, "
        "monto (entre 100.000 y 5.000.000) y plazo en meses (6, 12, 18, 24 o 36). "
        "Si el usuario no dio alguno de los tres datos, pedíselo antes de llamarla. "
        "La decisión de esta tool es final: comunicala tal cual, nunca la cambies."
    ),
    args_schema=SolicitudCredito,
    handle_validation_error=lambda e: (
        f"Solicitud inválida: {e.errors()[0]['msg']}. "
        "Pedile al usuario que corrija ese dato."
    ),
)


# La lista que le pasamos al agente
TOOLS_CREDITOS = [buscar_cliente_tool, consultar_bcra_tool, evaluar_credito_tool]


# ---------------------------------------------------------------------------
# Prueba manual
# ---------------------------------------------------------------------------

async def _main() -> None:
    print("=== El menú que ve el LLM ===")
    for tool in TOOLS_CREDITOS:
        print(f"- {tool.name}: {tool.args}")

    cuit = (await listar_cuits_de_ejemplo(1))[0]

    print(f"\n=== buscar_cliente con un CUIT de la base ({cuit}) ===")
    print(await buscar_cliente_tool.ainvoke({"cuit": cuit}))

    print("\n=== buscar_cliente con un CUIT inválido ===")
    print(await buscar_cliente_tool.ainvoke({"cuit": "123"}))

    print("\n=== buscar_cliente con un CUIT que no existe ===")
    print(await buscar_cliente_tool.ainvoke({"cuit": "20-00000000-1"}))


if __name__ == "__main__":
    asyncio.run(_main())
