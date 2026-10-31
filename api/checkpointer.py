"""
Elección del Checkpointer de LangGraph (la memoria de las conversaciones).

Ubicación en el proyecto: api/checkpointer.py

  - Si hay REDIS_URL y Redis responde -> AsyncRedisSaver (producción / Docker)
  - Si no -> AsyncSqliteSaver en un archivo local (desarrollo sin Docker)

Los dos persisten en disco: las conversaciones NO se pierden al reiniciar el
servidor. Nunca usamos InMemorySaver en la API.
"""

import logging
import os
from contextlib import AsyncExitStack

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("fintechlab")

REDIS_URL = os.getenv("REDIS_URL", "").strip()
CHECKPOINTS_SQLITE_PATH = os.getenv("CHECKPOINTS_SQLITE_PATH", "data/checkpoints.db")


async def abrir_checkpointer(stack: AsyncExitStack):
    """Devuelve (checkpointer, nombre). El stack se encarga de cerrarlo al apagar."""
    if REDIS_URL:
        try:
            from langgraph.checkpoint.redis.aio import AsyncRedisSaver

            saver = await stack.enter_async_context(AsyncRedisSaver.from_conn_string(REDIS_URL))
            log.info("Checkpointer: Redis (%s)", REDIS_URL)
            return saver, "redis"
        except Exception as e:  # noqa: BLE001 - queremos seguir funcionando sin Redis
            log.warning("No se pudo conectar a Redis (%s). Uso SQLite local.", e)

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    os.makedirs(os.path.dirname(CHECKPOINTS_SQLITE_PATH) or ".", exist_ok=True)
    saver = await stack.enter_async_context(
        AsyncSqliteSaver.from_conn_string(CHECKPOINTS_SQLITE_PATH)
    )
    log.info("Checkpointer: SQLite (%s)", CHECKPOINTS_SQLITE_PATH)
    return saver, "sqlite"
