#!/bin/bash

# ===== SCRIPT DE INICIO PARA VPS - CHATBOT XTALENTO =====
# Este script prepara y ejecuta el chatbot en un VPS de producción

echo "🚀 Iniciando Chatbot Xtalento en VPS..."

# Verificar si existe el archivo .env
if [ ! -f ".env" ]; then
    echo "⚠️  Archivo .env no encontrado."
    echo "📋 Copiando config_vps.env como .env..."
    cp config_vps.env .env
    echo "✅ Archivo .env creado."
    echo "🔧 IMPORTANTE: Edita el archivo .env con tus configuraciones reales antes de continuar."
    echo "   - OPENAI_API_KEY"
    echo "   - EVO_APIKEY" 
    echo "   - PUBLIC_WEBHOOK_URL"
    echo "   - EVO_INSTANCE"
    exit 1
fi

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 no está instalado"
    exit 1
fi

# Verificar pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 no está instalado"
    exit 1
fi

# Instalar dependencias
echo "📦 Instalando dependencias..."
pip3 install -r requirements.txt

# Verificar vectorstore
if [ ! -d "vectorstore" ]; then
    echo "📚 Vectorstore no encontrado. Se creará automáticamente al iniciar."
fi

# Configurar variables de entorno
export $(cat .env | xargs)

echo "✅ Configuración completada"
echo "🌐 Webhook URL configurada: $PUBLIC_WEBHOOK_URL"
echo "🔗 Evolution API URL: $EVO_API_URL"
echo "📱 Instancia: $EVO_INSTANCE"

# Iniciar servidor
echo "🚀 Iniciando servidor en modo producción..."
echo "📍 Puerto: 8000"
echo "🌍 Host: 0.0.0.0 (accesible desde internet)"

# Usar waitress para producción
waitress-serve --listen=0.0.0.0:8000 webhook:app
