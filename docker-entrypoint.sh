#!/bin/sh
# Prepara los datos (solo si faltan) y levanta la API.
set -e

if [ ! -f "${DB_PATH:-data/fintechlab.db}" ]; then
  echo ">> Generando base sintética de clientes..."
  python -m agents.creditos.seed
fi

if [ ! -f "${INDICE_ATENCION_PATH:-data/indice_atencion.json}" ]; then
  echo ">> Indexando políticas para el RAG..."
  python -m agents.atencion.ingest
fi

echo ">> Levantando API en http://localhost:8000 (docs en /docs)"
exec uvicorn api.main:app --host 0.0.0.0 --port 8000
