"""
Genera la base sintética de clientes y préstamos de FintechLab.

Ubicación en el proyecto: agents/creditos/seed.py

Es un paso de PREPARACIÓN (se corre una vez, antes de levantar el sistema),
no algo que el agente use en tiempo real. Por eso usa sqlite3 normal
(síncrono): acá no hay usuarios esperando una respuesta.

Correr desde la raíz del proyecto:
    py -m agents.creditos.seed

Cada vez que se corre, BORRA y vuelve a crear la base. Usa una semilla fija,
así que siempre genera exactamente los mismos clientes: eso hace que las
pruebas (y las trazas en LangSmith) sean reproducibles.
"""

import os
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from faker import Faker

load_dotenv()

DB_PATH = Path(os.getenv("DB_PATH", "data/fintechlab.db"))
CANTIDAD_CLIENTES = 40
SEMILLA = 42

fake = Faker("es_AR")
Faker.seed(SEMILLA)
random.seed(SEMILLA)


# ---------------------------------------------------------------------------
# CUIT con dígito verificador válido (algoritmo de AFIP)
# ---------------------------------------------------------------------------

PESOS_CUIT = [5, 4, 3, 2, 7, 6, 5, 4, 3, 2]


def generar_cuit() -> str:
    """Arma un CUIT de persona física con dígito verificador correcto.

    Es matemáticamente válido pero inventado: no corresponde a una persona
    de la base real (aunque podría coincidir por azar, por eso NUNCA se
    consultan estos CUITs en masa contra el BCRA).
    """
    while True:
        prefijo = random.choice(["20", "27"])            # 20 = masculino, 27 = femenino
        dni = str(random.randint(20_000_000, 45_000_000))
        base = prefijo + dni
        suma = sum(int(d) * p for d, p in zip(base, PESOS_CUIT))
        verificador = 11 - (suma % 11)
        if verificador == 11:
            verificador = 0
        if verificador == 10:                            # caso especial: probamos otro
            continue
        return base + str(verificador)


# ---------------------------------------------------------------------------
# Perfiles de cliente
# ---------------------------------------------------------------------------
# Repartimos los clientes en tres perfiles para que el agente tenga casos
# variados para razonar: buen pagador, riesgoso y sin historial.

def generar_prestamos(perfil: str, cliente_id: int, fecha_alta: date) -> list[tuple]:
    """Devuelve los préstamos de un cliente según su perfil."""
    if perfil == "nuevo":
        return []

    prestamos = []
    for _ in range(random.randint(1, 3)):
        monto = random.choice([200_000, 350_000, 500_000, 800_000, 1_200_000])
        plazo = random.choice([6, 12, 18, 24, 36])
        dias_desde_alta = (date.today() - fecha_alta).days
        fecha = fecha_alta + timedelta(days=random.randint(0, max(dias_desde_alta, 1)))

        if perfil == "bueno":
            estado = random.choice(["pagado", "pagado", "activo"])
        else:  # riesgoso
            estado = random.choice(["en_mora", "rechazado", "pagado"])

        prestamos.append((cliente_id, monto, plazo, estado, fecha.isoformat()))

    # Garantizamos que todo "riesgoso" tenga al menos una mora
    if perfil == "riesgoso" and not any(p[3] == "en_mora" for p in prestamos):
        cid, monto, plazo, _, fecha = prestamos[0]
        prestamos[0] = (cid, monto, plazo, "en_mora", fecha)

    return prestamos


# ---------------------------------------------------------------------------
# Creación de la base
# ---------------------------------------------------------------------------

def crear_tablas(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        DROP TABLE IF EXISTS prestamos;
        DROP TABLE IF EXISTS clientes;

        CREATE TABLE clientes (
            id                       INTEGER PRIMARY KEY AUTOINCREMENT,
            cuit                     TEXT NOT NULL UNIQUE,
            nombre                   TEXT NOT NULL,
            fecha_nacimiento         TEXT NOT NULL,
            ingreso_mensual          REAL NOT NULL,
            antiguedad_laboral_meses INTEGER NOT NULL,
            fecha_alta               TEXT NOT NULL
        );

        CREATE TABLE prestamos (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            cliente_id      INTEGER NOT NULL REFERENCES clientes(id),
            monto           REAL NOT NULL,
            plazo_meses     INTEGER NOT NULL,
            estado          TEXT NOT NULL,
            fecha_solicitud TEXT NOT NULL
        );
    """)


def sembrar() -> dict[str, list[str]]:
    """Crea la base y devuelve algunos CUITs de ejemplo por perfil."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    perfiles = ["bueno", "riesgoso", "nuevo"]
    ejemplos: dict[str, list[str]] = {p: [] for p in perfiles}

    with sqlite3.connect(DB_PATH) as conn:
        crear_tablas(conn)

        for i in range(CANTIDAD_CLIENTES):
            perfil = perfiles[i % 3]                     # reparte en partes iguales
            cuit = generar_cuit()
            fecha_alta = fake.date_between(start_date="-4y", end_date="-1m")

            cursor = conn.execute(
                """INSERT INTO clientes
                   (cuit, nombre, fecha_nacimiento, ingreso_mensual,
                    antiguedad_laboral_meses, fecha_alta)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    cuit,
                    fake.name(),
                    fake.date_of_birth(minimum_age=19, maximum_age=70).isoformat(),
                    round(random.uniform(400_000, 3_000_000), -3),
                    random.randint(0, 180),
                    fecha_alta.isoformat(),
                ),
            )
            cliente_id = cursor.lastrowid

            conn.executemany(
                """INSERT INTO prestamos
                   (cliente_id, monto, plazo_meses, estado, fecha_solicitud)
                   VALUES (?, ?, ?, ?, ?)""",
                generar_prestamos(perfil, cliente_id, fecha_alta),
            )

            if len(ejemplos[perfil]) < 2:
                ejemplos[perfil].append(cuit)

    return ejemplos


if __name__ == "__main__":
    ejemplos = sembrar()

    with sqlite3.connect(DB_PATH) as conn:
        n_clientes = conn.execute("SELECT COUNT(*) FROM clientes").fetchone()[0]
        n_prestamos = conn.execute("SELECT COUNT(*) FROM prestamos").fetchone()[0]

    print(f"Base creada en {DB_PATH}")
    print(f"  {n_clientes} clientes, {n_prestamos} préstamos\n")
    print("CUITs de ejemplo para probar (anotalos):")
    for perfil, cuits in ejemplos.items():
        print(f"  {perfil:<9} -> {', '.join(cuits)}")
