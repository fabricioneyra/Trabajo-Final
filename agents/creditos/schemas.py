"""
Schemas Pydantic del agente de Créditos.

Ubicación en el proyecto: agents/creditos/schemas.py

Son los "moldes" de los datos internos de FintechLab: qué es un cliente,
qué es un préstamo y cómo se ven juntos. Todo lo que salga de la base de
datos pasa por acá antes de que lo use cualquier otra parte del sistema.

Probar desde la raíz del proyecto:
    py -m agents.creditos.schemas
"""

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, ValidationError


# ---------------------------------------------------------------------------
# Tipo reutilizable: CUIT
# ---------------------------------------------------------------------------

def _normalizar_cuit(valor: str) -> str:
    """Saca guiones y espacios, y exige 11 dígitos."""
    limpio = str(valor).replace("-", "").replace(" ", "").strip()
    if not (limpio.isdigit() and len(limpio) == 11):
        raise ValueError("El CUIT debe tener exactamente 11 dígitos")
    return limpio


# Un "str que además es un CUIT válido". Se usa como tipo en cualquier modelo.
Cuit = Annotated[str, BeforeValidator(_normalizar_cuit)]


# ---------------------------------------------------------------------------
# Modelos de datos
# ---------------------------------------------------------------------------

# Literal = solo se aceptan exactamente estos textos. Cualquier otro valor falla.
EstadoPrestamo = Literal["activo", "pagado", "en_mora", "rechazado"]


class Cliente(BaseModel):
    """Una persona registrada en FintechLab."""

    id: int
    cuit: Cuit
    nombre: str
    fecha_nacimiento: date
    ingreso_mensual: float = Field(..., gt=0, description="Ingreso neto mensual en pesos")
    antiguedad_laboral_meses: int = Field(..., ge=0)
    fecha_alta: date


class Prestamo(BaseModel):
    """Un préstamo que el cliente tuvo o tiene con FintechLab."""

    id: int
    cliente_id: int
    monto: float = Field(..., gt=0)
    plazo_meses: int = Field(..., gt=0)
    estado: EstadoPrestamo
    fecha_solicitud: date


class ClienteConHistorial(BaseModel):
    """Un cliente junto con todos sus préstamos.

    Es lo que va a devolver la tool `buscar_cliente`: todo lo que FintechLab
    sabe internamente del cliente, antes de cruzarlo con el BCRA.
    """

    cliente: Cliente
    prestamos: list[Prestamo] = []

    # @property = un dato calculado a partir de otros. No se guarda en la base.
    @property
    def tiene_mora(self) -> bool:
        """True si alguna vez tuvo un préstamo en mora."""
        return any(p.estado == "en_mora" for p in self.prestamos)

    @property
    def prestamos_pagados(self) -> int:
        """Cuántos préstamos terminó de pagar."""
        return sum(1 for p in self.prestamos if p.estado == "pagado")

    @property
    def es_cliente_nuevo(self) -> bool:
        """True si nunca tuvo un préstamo con nosotros."""
        return len(self.prestamos) == 0


# ---------------------------------------------------------------------------
# Prueba manual
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Prueba 1: cliente válido con historial ===")
    datos = ClienteConHistorial(
        cliente=Cliente(
            id=1,
            cuit="20-12345678-9",            # con guiones: el validador los saca
            nombre="Ana Pérez",
            fecha_nacimiento="1990-05-14",   # texto: Pydantic lo convierte a date
            ingreso_mensual=850000,
            antiguedad_laboral_meses=36,
            fecha_alta="2024-01-10",
        ),
        prestamos=[
            Prestamo(id=1, cliente_id=1, monto=300000, plazo_meses=12,
                     estado="pagado", fecha_solicitud="2024-02-01"),
            Prestamo(id=2, cliente_id=1, monto=500000, plazo_meses=18,
                     estado="en_mora", fecha_solicitud="2025-03-15"),
        ],
    )
    print("CUIT normalizado:  ", datos.cliente.cuit)
    print("Tipo de fecha:     ", type(datos.cliente.fecha_nacimiento).__name__)
    print("Tiene mora:        ", datos.tiene_mora)
    print("Préstamos pagados: ", datos.prestamos_pagados)
    print("Es cliente nuevo:  ", datos.es_cliente_nuevo)

    print("\n=== Prueba 2: datos inválidos ===")
    try:
        Cliente(
            id=2,
            cuit="123",                  # CUIT corto
            nombre="Juan",
            fecha_nacimiento="1985-01-01",
            ingreso_mensual=-100,        # ingreso negativo
            antiguedad_laboral_meses=12,
            fecha_alta="2024-01-01",
        )
    except ValidationError as e:
        for error in e.errors():
            print(f"- Campo {error['loc'][0]}: {error['msg']}")

    print("\n=== Prueba 3: estado de préstamo inexistente ===")
    try:
        Prestamo(id=3, cliente_id=1, monto=1000, plazo_meses=6,
                 estado="cancelado", fecha_solicitud="2025-01-01")
    except ValidationError as e:
        print("-", e.errors()[0]["msg"])
