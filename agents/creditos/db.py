"""
Acceso async a la base de clientes de FintechLab.

Ubicación en el proyecto: agents/creditos/db.py

Es la ÚNICA puerta de entrada a la base en tiempo real: ninguna otra parte
del sistema abre data/fintechlab.db directamente. Usa aiosqlite (la versión
async de sqlite3) para no bloquear el servidor mientras espera al disco.

Probar desde la raíz del proyecto (después de correr seed.py):
    py -m agents.creditos.db
"""

import asyncio
import os
from pathlib import Path

import aiosqlite
from dotenv import load_dotenv

from agents.creditos.schemas import Cliente, ClienteConHistorial, Prestamo

load_dotenv()

DB_PATH = Path(os.getenv("DB_PATH", "data/fintechlab.db"))


async def buscar_cliente(cuit: str) -> ClienteConHistorial | None:
    """Busca un cliente por CUIT junto con todos sus préstamos.

    Devuelve None si el CUIT no está registrado. No es un error: puede ser
    alguien que nunca operó con FintechLab.
    """
    async with aiosqlite.connect(DB_PATH) as conn:
        # row_factory = Row: cada fila se puede leer como diccionario
        conn.row_factory = aiosqlite.Row

        cursor = await conn.execute("SELECT * FROM clientes WHERE cuit = ?", (cuit,))
        fila = await cursor.fetchone()
        if fila is None:
            return None

        # La fila cruda pasa por el schema: si algo vino mal de la base, falla acá
        cliente = Cliente(**dict(fila))

        cursor = await conn.execute(
            "SELECT * FROM prestamos WHERE cliente_id = ? ORDER BY fecha_solicitud DESC",
            (cliente.id,),
        )
        prestamos = [Prestamo(**dict(f)) for f in await cursor.fetchall()]

    return ClienteConHistorial(cliente=cliente, prestamos=prestamos)


async def listar_cuits_de_ejemplo(limite: int = 5) -> list[str]:
    """Algunos CUITs existentes, para probar sin abrir la base a mano."""
    async with aiosqlite.connect(DB_PATH) as conn:
        cursor = await conn.execute("SELECT cuit FROM clientes LIMIT ?", (limite,))
        return [fila[0] for fila in await cursor.fetchall()]


# ---------------------------------------------------------------------------
# Prueba manual
# ---------------------------------------------------------------------------

async def _main() -> None:
    cuits = await listar_cuits_de_ejemplo(1)
    if not cuits:
        print("La base está vacía. Corré primero: py -m agents.creditos.seed")
        return

    print("=== Prueba 1: cliente existente ===")
    resultado = await buscar_cliente(cuits[0])
    print("Nombre:            ", resultado.cliente.nombre)
    print("Préstamos:         ", len(resultado.prestamos))
    print("Tiene mora:        ", resultado.tiene_mora)
    print("Es cliente nuevo:  ", resultado.es_cliente_nuevo)

    print("\n=== Prueba 2: CUIT que no existe ===")
    print(await buscar_cliente("20000000001"))


if __name__ == "__main__":
    asyncio.run(_main())
