"""
Cliente async de la Central de Deudores del BCRA.

Ubicación en el proyecto: agents/creditos/bcra.py

Pieza aislada del agente de Créditos: recibe un CUIT, consulta la API pública
del BCRA y devuelve un resultado validado con Pydantic. Todavía NO es una tool
de LangChain; eso lo hacemos en el paso siguiente, envolviendo esta función.

Probar desde la raíz del proyecto:
    py -m agents.creditos.bcra 20123456789
"""

import asyncio
import os
import sys

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

load_dotenv()

# Nada hard-codeado: URL y timeout salen del .env (con defaults razonables)
BCRA_BASE_URL = os.getenv("BCRA_BASE_URL", "https://api.bcra.gob.ar")
BCRA_TIMEOUT = float(os.getenv("BCRA_TIMEOUT", "10"))
BCRA_REINTENTOS = int(os.getenv("BCRA_REINTENTOS", "3"))

# "mock" (default): respuestas simuladas, estables y sin salir a internet.
# "real": consulta la API del BCRA.
# Mock por defecto porque nuestros CUITs son sintéticos: consultarlos en el BCRA
# real podría traer datos de una persona real que coincida por azar.
BCRA_MODO = os.getenv("BCRA_MODO", "mock").lower()

# Errores del lado del servidor que vale la pena reintentar
ESTADOS_REINTENTABLES = {500, 502, 503, 504}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ConsultaBCRAInput(BaseModel):
    """Lo que entra: un CUIT/CUIL. Se normaliza y valida antes de salir a la red."""

    cuit: str = Field(..., description="CUIT/CUIL de 11 dígitos, con o sin guiones")

    @field_validator("cuit")
    @classmethod
    def normalizar_cuit(cls, v: str) -> str:
        limpio = v.replace("-", "").replace(" ", "").strip()
        if not (limpio.isdigit() and len(limpio) == 11):
            raise ValueError("El CUIT debe tener exactamente 11 dígitos")
        return limpio


class DeudaEntidad(BaseModel):
    """Una línea de deuda con una entidad financiera, tal como la informa el BCRA."""

    # extra="ignore": el BCRA manda más campos de los que usamos; no rompemos por eso
    model_config = ConfigDict(extra="ignore")

    entidad: str
    situacion: int = Field(..., description="1 = normal ... 5/6 = irrecuperable")
    monto: float = Field(0, description="Monto de la deuda, en miles de pesos")
    diasAtrasoPago: int | None = None


class ResultadoBCRA(BaseModel):
    """Lo que sale: un resumen listo para que el agente razone sobre él."""

    cuit: str
    encontrado: bool = Field(..., description="False si el BCRA no tiene deudas informadas")
    denominacion: str | None = None
    periodo: str | None = Field(None, description="Período informado, formato AAAAMM")
    peor_situacion: int | None = None
    deuda_total_miles: float = 0
    entidades: list[DeudaEntidad] = []
    error: str | None = Field(None, description="Si no se pudo consultar, el motivo")


# ---------------------------------------------------------------------------
# Modo simulado
# ---------------------------------------------------------------------------

def _respuesta_simulada(cuit: str) -> ResultadoBCRA:
    """Respuesta inventada pero DETERMINÍSTICA: el mismo CUIT siempre da lo mismo.

    Reparte los CUITs en casos parecidos a la realidad, para que el agente
    pueda ejercitar todas las ramas de la política:
      ~20% sin deudas informadas, ~50% situación 1, ~12% situación 2,
      ~10% situación 3 (revisión manual), ~8% situación 4-5 (rechazo).
    """
    caso = int(cuit) % 50
    if caso < 10:
        return ResultadoBCRA(cuit=cuit, encontrado=False)

    if caso < 35:
        situacion = 1
    elif caso < 41:
        situacion = 2
    elif caso < 46:
        situacion = 3
    else:
        situacion = 4 + (caso % 2)  # 4 o 5

    entidad = DeudaEntidad(
        entidad="BANCO SIMULADO S.A.",
        situacion=situacion,
        monto=float(50 + caso * 13),
        diasAtrasoPago=0 if situacion == 1 else 30 * situacion,
    )
    return ResultadoBCRA(
        cuit=cuit,
        encontrado=True,
        denominacion="CLIENTE SIMULADO",
        periodo="202609",
        peor_situacion=situacion,
        deuda_total_miles=entidad.monto,
        entidades=[entidad],
    )


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

async def consultar_bcra(cuit: str) -> ResultadoBCRA:
    """Consulta las deudas actuales de un CUIT en la Central de Deudores."""

    # 1) Validar la entrada. Si el CUIT es inválido, esto lanza ValidationError
    #    y no gastamos una llamada de red.
    datos = ConsultaBCRAInput(cuit=cuit)

    if BCRA_MODO == "mock":
        return _respuesta_simulada(datos.cuit)

    url = f"{BCRA_BASE_URL}/CentralDeDeudores/v1.0/Deudas/{datos.cuit}"

    # 2) Llamada async con httpx (nunca requests dentro de async def).
    #    Los 5xx suelen ser caídas momentáneas del BCRA: reintentamos con espera
    #    creciente (1s, 2s, 4s...) antes de darnos por vencidos.
    headers = {"User-Agent": "FintechLab/1.0 (proyecto academico)", "Accept": "application/json"}
    resp = None
    try:
        async with httpx.AsyncClient(timeout=BCRA_TIMEOUT, headers=headers) as client:
            for intento in range(BCRA_REINTENTOS):
                resp = await client.get(url)
                if resp.status_code not in ESTADOS_REINTENTABLES:
                    break
                if intento < BCRA_REINTENTOS - 1:
                    await asyncio.sleep(2 ** intento)  # async: no bloquea el servidor
    except httpx.HTTPError as e:
        # No lanzamos: devolvemos el error como dato para que el agente
        # pueda decidir (por ejemplo, escalar a un analista).
        return ResultadoBCRA(
            cuit=datos.cuit,
            encontrado=False,
            error=f"No se pudo contactar al BCRA ({type(e).__name__})",
        )

    # 3) Interpretar la respuesta
    if resp.status_code == 404:
        # El BCRA no tiene deudas informadas para ese CUIT
        return ResultadoBCRA(cuit=datos.cuit, encontrado=False)

    if resp.status_code != 200:
        # El BCRA suele explicar el motivo en "errorMessages"; si viene, lo usamos
        try:
            mensajes = resp.json().get("errorMessages") or []
        except ValueError:
            mensajes = []
        detalle = f": {'; '.join(mensajes)}" if mensajes else ""
        return ResultadoBCRA(
            cuit=datos.cuit,
            encontrado=False,
            error=f"El BCRA respondió con estado {resp.status_code}{detalle}",
        )

    results = resp.json().get("results") or {}
    periodos = results.get("periodos") or []

    if not periodos:
        return ResultadoBCRA(
            cuit=datos.cuit,
            encontrado=True,
            denominacion=results.get("denominacion"),
        )

    # Nos quedamos con el período más reciente
    ultimo = max(periodos, key=lambda p: str(p.get("periodo", "")))
    entidades = [DeudaEntidad(**e) for e in ultimo.get("entidades", [])]

    return ResultadoBCRA(
        cuit=datos.cuit,
        encontrado=True,
        denominacion=results.get("denominacion"),
        periodo=str(ultimo.get("periodo")),
        peor_situacion=max((e.situacion for e in entidades), default=None),
        deuda_total_miles=sum(e.monto for e in entidades),
        entidades=entidades,
    )


# ---------------------------------------------------------------------------
# Prueba manual
# ---------------------------------------------------------------------------

async def _main() -> None:
    cuit = sys.argv[1] if len(sys.argv) > 1 else "20123456789"
    try:
        resultado = await consultar_bcra(cuit)
    except ValidationError as e:
        print("CUIT inválido:", e.errors()[0]["msg"])
        return
    print(resultado.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
