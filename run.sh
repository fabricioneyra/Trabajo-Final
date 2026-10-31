#!/bin/sh
# Levanta todo el sistema (API + Redis) con una sola instrucción.
set -e
if [ ! -f .env ]; then
  echo "Falta el archivo .env. Copiá .env.example como .env y completá tus claves."
  exit 1
fi
docker compose up --build
