"""
Motor de evaluación de créditos de FintechLab.

Ubicación en el proyecto: agents/creditos/scoring.py

Aplica data/policies/politica_creditos.md a una solicitud concreta y decide:
  - "aprobado"        -> cumple todo; se estima tasa y cuota
  - "rechazado"       -> cae en una causal de rechazo automático
  - "revision_manual" -> la política obliga a que lo vea un analista humano

Es 100% DETERMINÍSTICO (sin LLM): mismos datos, misma decisión. A propósito:
la decisión de dar un crédito tiene que ser auditable. El LLM solo conversa
con el cliente y le explica el resultado; nunca decide.

Limitaciones conocidas (documentadas, no escondidas):
  - "CUIT activo en AFIP": solo validamos formato, no hay integración con AFIP.
  - Cheques rechazados: no se consultan todavía (extensión futura).
  - Tasa de referencia del BCRA: constante configurable (TASA_REFERENCIA_ANUAL).

Probar desde la raíz del proyecto:
    py -m agents.creditos.scoring
"""

import asyncio
import os
from datetime import date
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from agents.creditos.bcra import consultar_bcra
from agents.creditos.db import buscar_cliente, listar_cuits_de_ejemplo
from agents.creditos.schemas import Cuit

load_dotenv()

# --- Parámetros de la política (en un solo lugar, fáciles de auditar) ---
MONTO_MINIMO = 100_000
MONTO_MAXIMO = 5_000_000
MONTO_REVISION_MANUAL = 2_000_000
MONTO_MAXIMO_CON_MORA = 1_000_000      # "máximo autorizado para el perfil de riesgo"
PLAZOS_VALIDOS = (6, 12, 18, 24, 36)
EDAD_MINIMA, EDAD_MAXIMA = 18, 75
ANTIGUEDAD_MINIMA_MESES = 6
CUOTA_MAXIMA_SOBRE_INGRESO = 0.30
TASA_REFERENCIA_ANUAL = float(os.getenv("TASA_REFERENCIA_ANUAL", "0.35"))


# ---------------------------------------------------------------------------
# Schemas de entrada y salida
# ---------------------------------------------------------------------------

class SolicitudCredito(BaseModel):
    """Lo que pide el cliente: quién, cuánto y en cuántas cuotas."""

    cuit: Cuit = Field(..., description="CUIT/CUIL de 11 dígitos")
    monto: float = Field(..., ge=MONTO_MINIMO, le=MONTO_MAXIMO,
                         description="Monto en pesos, entre 100.000 y 5.000.000")
    plazo_meses: int = Field(..., description="Uno de: 6, 12, 18, 24 o 36")

    @field_validator("plazo_meses")
    @classmethod
    def plazo_permitido(cls, v: int) -> int:
        if v not in PLAZOS_VALIDOS:
            raise ValueError(f"El plazo debe ser uno de {PLAZOS_VALIDOS}")
        return v


Decision = Literal["aprobado", "rechazado", "revision_manual"]


class ResultadoEvaluacion(BaseModel):
    """La decisión, con el porqué y todos los chequeos hechos (trazabilidad)."""

    cuit: str
    decision: Decision
    motivo: str
    chequeos: list[str] = []
    tasa_anual: float | None = None
    cuota_mensual: float | None = None


# ---------------------------------------------------------------------------
# Cálculos auxiliares
# ---------------------------------------------------------------------------

def calcular_edad(nacimiento: date, hoy: date) -> int:
    cumplio = (hoy.month, hoy.day) >= (nacimiento.month, nacimiento.day)
    return hoy.year - nacimiento.year - (0 if cumplio else 1)


def calcular_cuota(monto: float, tasa_anual: float, plazo_meses: int) -> float:
    """Cuota fija mensual (sistema francés, el habitual en préstamos personales)."""
    i = tasa_anual / 12
    return round(monto * i / (1 - (1 + i) ** -plazo_meses), 2)


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

async def evaluar_credito(solicitud: SolicitudCredito) -> ResultadoEvaluacion:
    chequeos: list[str] = []
    rechazos: list[str] = []
    revisiones: list[str] = []

    # 1) Datos internos
    datos = await buscar_cliente(solicitud.cuit)
    if datos is None:
        return ResultadoEvaluacion(
            cuit=solicitud.cuit,
            decision="revision_manual",
            motivo="El CUIT no está registrado como cliente de FintechLab; "
                   "primero tiene que darse de alta.",
            chequeos=["Cliente no encontrado en la base interna"],
        )
    cliente = datos.cliente

    # 2) BCRA
    bcra = await consultar_bcra(solicitud.cuit)

    # 3) Reglas de la política
    edad = calcular_edad(cliente.fecha_nacimiento, date.today())
    chequeos.append(f"Edad: {edad} años")
    if not EDAD_MINIMA <= edad < EDAD_MAXIMA:
        rechazos.append(f"Edad fuera del rango permitido ({EDAD_MINIMA} a {EDAD_MAXIMA} años)")

    chequeos.append(f"Antigüedad laboral: {cliente.antiguedad_laboral_meses} meses")
    if cliente.antiguedad_laboral_meses < ANTIGUEDAD_MINIMA_MESES:
        rechazos.append(f"Antigüedad laboral menor a {ANTIGUEDAD_MINIMA_MESES} meses")

    if bcra.error:
        chequeos.append(f"BCRA: no se pudo consultar ({bcra.error})")
        revisiones.append("No se pudo verificar la situación en el BCRA")
    elif bcra.peor_situacion is None:
        chequeos.append("BCRA: sin deudas informadas")
    else:
        chequeos.append(f"BCRA: peor situación {bcra.peor_situacion}")
        if bcra.peor_situacion >= 4:
            rechazos.append(f"Situación {bcra.peor_situacion} en el BCRA")
        elif bcra.peor_situacion == 3:
            revisiones.append("Situación 3 en el BCRA (riesgo intermedio)")

    chequeos.append(f"Historial interno: mora={datos.tiene_mora}, "
                    f"pagados={datos.prestamos_pagados}")
    if datos.tiene_mora and solicitud.monto > MONTO_MAXIMO_CON_MORA:
        rechazos.append(f"Con antecedentes de mora el máximo es ${MONTO_MAXIMO_CON_MORA:,.0f}")

    if solicitud.monto > MONTO_REVISION_MANUAL:
        revisiones.append(f"Monto mayor a ${MONTO_REVISION_MANUAL:,.0f}")

    # 4) Tasa = referencia + spread de riesgo, y cuota
    spread = 0.0
    if bcra.peor_situacion == 2:
        spread += 0.10
    if datos.tiene_mora:
        spread += 0.10
    if datos.es_cliente_nuevo:
        spread += 0.05
    tasa = round(TASA_REFERENCIA_ANUAL + spread, 4)
    cuota = calcular_cuota(solicitud.monto, tasa, solicitud.plazo_meses)

    tope_cuota = cliente.ingreso_mensual * CUOTA_MAXIMA_SOBRE_INGRESO
    chequeos.append(f"Cuota estimada ${cuota:,.0f} vs. tope 30% del ingreso ${tope_cuota:,.0f}")
    if cuota > tope_cuota:
        rechazos.append("La cuota supera el 30% del ingreso mensual")

    # 5) Decisión: el rechazo pesa más que la revisión
    if rechazos:
        return ResultadoEvaluacion(cuit=solicitud.cuit, decision="rechazado",
                                   motivo="; ".join(rechazos), chequeos=chequeos)
    if revisiones:
        return ResultadoEvaluacion(cuit=solicitud.cuit, decision="revision_manual",
                                   motivo="; ".join(revisiones), chequeos=chequeos)
    return ResultadoEvaluacion(cuit=solicitud.cuit, decision="aprobado",
                               motivo="Cumple todos los requisitos de la política",
                               chequeos=chequeos, tasa_anual=tasa, cuota_mensual=cuota)


# ---------------------------------------------------------------------------
# Prueba manual: evalúa a los primeros clientes con la misma solicitud
# ---------------------------------------------------------------------------

async def _main() -> None:
    for cuit in await listar_cuits_de_ejemplo(8):
        solicitud = SolicitudCredito(cuit=cuit, monto=500_000, plazo_meses=12)
        r = await evaluar_credito(solicitud)
        print(f"{cuit}  ->  {r.decision.upper():<16} {r.motivo}")


if __name__ == "__main__":
    asyncio.run(_main())
