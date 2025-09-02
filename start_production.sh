#!/bin/bash
# Script para iniciar el bot en producción (VPS)

# Cargar variables de entorno
export $(cat .env | grep -v '^#' | xargs)

# Iniciar con Waitress en modo producción
echo "Iniciando chatbot en modo producción..."
waitress-serve --listen=127.0.0.1:8000 webhook:app
