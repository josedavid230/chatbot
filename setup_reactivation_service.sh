#!/bin/bash
# Script para configurar el servicio de reactivación automática

echo "🔧 CONFIGURANDO SERVICIO DE REACTIVACIÓN"
echo "========================================"

# Variables
SERVICE_NAME="xtalento-reactivation"
WORKING_DIR="/opt/xtalento/bot/chatbot"
PYTHON_PATH="/opt/xtalento/bot/chatbot/venv/bin/python"
SCRIPT_PATH="$WORKING_DIR/reactivation_task.py"

# Verificar que el directorio existe
if [ ! -d "$WORKING_DIR" ]; then
    echo "❌ ERROR: Directorio $WORKING_DIR no existe"
    exit 1
fi

# Verificar que el script existe
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ ERROR: Script $SCRIPT_PATH no existe"
    exit 1
fi

# Crear archivo de servicio systemd
echo "📝 Creando archivo de servicio systemd..."

sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null << EOF
[Unit]
Description=XTalento Bot Reactivation Service
After=network.target
Wants=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$WORKING_DIR
Environment=PATH=$WORKING_DIR/venv/bin
ExecStart=$PYTHON_PATH $SCRIPT_PATH
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=xtalento-reactivation

[Install]
WantedBy=multi-user.target
EOF

# Recargar systemd
echo "🔄 Recargando systemd..."
sudo systemctl daemon-reload

# Habilitar el servicio
echo "✅ Habilitando servicio para arranque automático..."
sudo systemctl enable ${SERVICE_NAME}

echo ""
echo "✅ SERVICIO CONFIGURADO EXITOSAMENTE"
echo "========================================"
echo ""
echo "📋 COMANDOS ÚTILES:"
echo "   Iniciar:    sudo systemctl start ${SERVICE_NAME}"
echo "   Detener:    sudo systemctl stop ${SERVICE_NAME}"
echo "   Estado:     sudo systemctl status ${SERVICE_NAME}"
echo "   Logs:       sudo journalctl -u ${SERVICE_NAME} -f"
echo "   Reiniciar:  sudo systemctl restart ${SERVICE_NAME}"
echo ""
echo "🚀 Para iniciar el servicio ahora:"
echo "   sudo systemctl start ${SERVICE_NAME}"
