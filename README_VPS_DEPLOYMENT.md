# 🚀 Deployment Chatbot Xtalento en VPS

## 📋 Configuración Previa

### 1. Preparar el VPS
```bash
# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar Python y pip
sudo apt install python3 python3-pip python3-venv -y

# Instalar git (si no está)
sudo apt install git -y

# Instalar Caddy (proxy reverso moderno)
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install caddy
```

### 2. Clonar el Proyecto
```bash
# Ir al directorio deseado
cd /opt

# Clonar el repositorio
sudo git clone [URL_DE_TU_REPO] chatbot_xtalento
cd chatbot_xtalento

# Dar permisos
sudo chown -R $USER:$USER /opt/chatbot_xtalento
```

## 🔧 Configuración del Chatbot

### 1. Configurar Variables de Entorno
```bash
# Copiar archivo de configuración
cp config_vps.env .env

# Editar configuración
nano .env
```

**Variables importantes a configurar en .env:**
```env
OPENAI_API_KEY=sk-tu-api-key-real
EVO_API_URL=https://evolution.midominio.com
EVO_APIKEY=tu-api-key-evolution-real
EVO_INSTANCE=xtalento-bot
PUBLIC_WEBHOOK_URL=https://chatbot.midominio.com/webhook
```

### 2. Instalar y Ejecutar
```bash
# Hacer ejecutable el script
chmod +x start_vps.sh

# Ejecutar (primera vez para verificar configuración)
./start_vps.sh
```

## 🔗 Configuración Evolution API (VPS Separado)

### 1. Evolution API en VPS Separado

**Arquitectura de deployment:**
```
VPS 1 (Evolution API)     VPS 2 (Chatbot)
├── Evolution API         ├── Tu Chatbot
├── Puerto 8080           ├── Puerto 8000  
└── https://evolution     └── https://chatbot
    .midominio.com            .midominio.com
```

### 2. Instalar Evolution API en su VPS

**En el VPS de Evolution API:**
```bash
# Crear directorio para Evolution API
sudo mkdir /opt/evolution-api
cd /opt/evolution-api

# Descargar Docker
sudo curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Crear docker-compose para Evolution API
sudo nano docker-compose.yml
```

**Contenido del docker-compose.yml para Evolution API:**
```yaml
version: '3.8'

services:
  evolution-api:
    image: atendai/evolution-api:latest
    ports:
      - "8080:8080"
    environment:
      - AUTHENTICATION_API_KEY=tu-api-key-aqui
      - AUTHENTICATION_EXPOSE_IN_FETCH_INSTANCES=true
      - QRCODE_LIMIT=30
      - QRCODE_COLOR=#198754
      - WEBHOOK_GLOBAL_URL=''
      - WEBHOOK_GLOBAL_ENABLED=false
      - WEBHOOK_GLOBAL_WEBHOOK_BY_EVENTS=false
      - CONFIG_SESSION_PHONE_CLIENT=Xtalento Bot
      - CONFIG_SESSION_PHONE_NAME=Chrome
    volumes:
      - evolution_instances:/evolution/instances
    restart: unless-stopped

volumes:
  evolution_instances:
```

### 3. Configurar Caddy para Evolution API
```bash
# En el VPS de Evolution API
sudo nano /etc/caddy/Caddyfile
```

**Configuración Caddyfile para Evolution API:**
```caddyfile
# Caddyfile para Evolution API
evolution.midominio.com {
    reverse_proxy localhost:8080 {
        header_up X-Real-IP {remote_host}
        header_up X-Forwarded-For {remote_host}
        header_up X-Forwarded-Proto {scheme}
        header_up Host {host}
    }
    
    # SSL automático por Caddy
    log {
        output file /var/log/caddy/evolution.log
    }
}
```

### 4. Iniciar Evolution API y Caddy
```bash
# Iniciar Evolution API
sudo docker-compose up -d

# Verificar que está corriendo
sudo docker-compose ps

# Reiniciar Caddy para cargar configuración
sudo systemctl reload caddy

# Verificar estado de Caddy
sudo systemctl status caddy

# SSL se configura automáticamente por Caddy
# No necesitas certbot ni configuración manual
```

## 🌐 Configuración Caddy para Chatbot (VPS del Chatbot)

### 1. Configurar Caddy en VPS del Chatbot
```bash
# En el VPS del chatbot
sudo nano /etc/caddy/Caddyfile
```

**Contenido del Caddyfile:**
```caddyfile
# Caddyfile para Chatbot Xtalento
chatbot.midominio.com {
    # Webhook principal del chatbot
    handle /webhook {
        reverse_proxy localhost:8000 {
            header_up X-Real-IP {remote_host}
            header_up X-Forwarded-For {remote_host}
            header_up X-Forwarded-Proto {scheme}
            
            transport http {
                read_timeout 300s
                write_timeout 300s
                dial_timeout 30s
            }
        }
    }
    
    # Endpoints de administración
    handle /healthz {
        reverse_proxy localhost:8000
    }
    
    handle /register_webhook {
        reverse_proxy localhost:8000
    }
    
    handle /check_webhook {
        reverse_proxy localhost:8000
    }
    
    handle /sessions* {
        reverse_proxy localhost:8000
    }
    
    # Página de estado
    handle / {
        respond "🤖 Xtalento Chatbot - Sistema Operativo" 200
    }
    
    # SSL automático y logs
    log {
        output file /var/log/caddy/chatbot.log
    }
}
```

### 2. Activar configuración del Chatbot
```bash
# En el VPS del chatbot
# Verificar configuración de Caddy
sudo caddy validate --config /etc/caddy/Caddyfile

# Recargar configuración
sudo systemctl reload caddy

# Verificar estado
sudo systemctl status caddy

# Ver logs en tiempo real (opcional)
sudo journalctl -f -u caddy
```

## 🔒 SSL Automático con Caddy (HTTPS)

### ✅ **SSL Completamente Automático**

Con Caddy **NO necesitas** instalar ni configurar SSL manualmente:

```bash
# ❌ NO NECESITAS ESTO con Caddy:
# sudo apt install certbot
# sudo certbot --nginx
# sudo crontab -e (renovación)

# ✅ Caddy hace TODO automáticamente:
# - Obtiene certificados Let's Encrypt
# - Configura HTTPS automáticamente 
# - Renueva certificados automáticamente
# - Redirecciona HTTP a HTTPS
```

### Verificar SSL automático:
```bash
# Verificar que Caddy obtuvo los certificados
sudo caddy list-certificates

# Ver logs de certificados
sudo journalctl -u caddy | grep -i cert

# Probar HTTPS
curl -I https://chatbot.midominio.com/healthz
curl -I https://evolution.midominio.com
```

### Si hay problemas con SSL:
```bash
# Forzar renovación de certificados
sudo caddy reload --config /etc/caddy/Caddyfile

# Ver logs detallados
sudo journalctl -f -u caddy
```

## 🚀 Servicios Systemd (Auto-inicio)

### 1. Crear servicio para el chatbot
```bash
sudo nano /etc/systemd/system/chatbot-xtalento.service
```

**Contenido del servicio:**
```ini
[Unit]
Description=Chatbot Xtalento
After=network.target

[Service]
Type=exec
User=ubuntu
WorkingDirectory=/opt/chatbot_xtalento
Environment=PATH=/usr/bin:/usr/local/bin
ExecStart=/usr/bin/waitress-serve --listen=0.0.0.0:8000 webhook:app
Restart=always

[Install]
WantedBy=multi-user.target
```

### 2. Activar servicio
```bash
# Recargar systemd
sudo systemctl daemon-reload

# Habilitar servicio
sudo systemctl enable chatbot-xtalento

# Iniciar servicio
sudo systemctl start chatbot-xtalento

# Verificar estado
sudo systemctl status chatbot-xtalento
```

## 🔧 Conectar Evolution API con el Chatbot (VPS Separados)

### 1. Crear instancia en Evolution API
```bash
# Hacer POST a Evolution API para crear instancia
curl -X POST "https://evolution.midominio.com/instance/create" \
  -H "Content-Type: application/json" \
  -H "apikey: tu-api-key-evolution" \
  -d '{
    "instanceName": "xtalento-bot",
    "token": "optional-token",
    "qrcode": true,
    "webhook": {
      "url": "https://chatbot.midominio.com/webhook",
      "events": ["MESSAGES_UPSERT"]
    }
  }'
```

### 2. Registrar webhook automáticamente
```bash
# Usar el endpoint del chatbot para registrar el webhook
curl -X POST "https://chatbot.midominio.com/register_webhook"
```

### 3. Verificar conexión
```bash
# Verificar estado del webhook
curl "https://chatbot.midominio.com/check_webhook"

# Verificar salud del sistema
curl "https://chatbot.midominio.com/healthz"

# Verificar Evolution API
curl "https://evolution.midominio.com/instance/fetchInstances" \
  -H "apikey: tu-api-key-evolution"
```

## 📱 Conectar WhatsApp

1. **Generar QR**: Accede a `https://evolution.midominio.com/instance/connect/xtalento-bot`
2. **Escanear QR**: Usa WhatsApp Web para escanear el código
3. **Verificar conexión**: El bot debería estar listo para recibir mensajes

## 🐛 Troubleshooting

### Ver logs del chatbot
```bash
sudo journalctl -u chatbot-xtalento -f
```

### Ver logs de Evolution API
```bash
sudo docker-compose logs -f evolution-api
```

### Ver logs de Caddy
```bash
# Logs del sistema
sudo journalctl -u caddy -f

# Logs de acceso personalizados
sudo tail -f /var/log/caddy/chatbot.log
sudo tail -f /var/log/caddy/evolution.log
```

### Verificar puertos
```bash
sudo netstat -tlnp | grep -E ':8000|:8080|:80|:443'
```

### Comandos útiles de Caddy
```bash
# Verificar configuración
sudo caddy validate --config /etc/caddy/Caddyfile

# Recargar configuración sin downtime
sudo systemctl reload caddy

# Ver certificados SSL activos
sudo caddy list-certificates

# Formato del Caddyfile
sudo caddy fmt --overwrite /etc/caddy/Caddyfile

# Ver información del adaptador
sudo caddy adapt --config /etc/caddy/Caddyfile
```

### Reiniciar servicios
```bash
# Reiniciar chatbot
sudo systemctl restart chatbot-xtalento

# Reiniciar Evolution API
sudo docker-compose restart evolution-api

# Reiniciar Caddy
sudo systemctl restart caddy

# Recargar Caddy (sin downtime)
sudo systemctl reload caddy
```

## ✅ Verificación Final

1. ✅ Evolution API corriendo en puerto 8080
2. ✅ Chatbot corriendo en puerto 8000 
3. ✅ Nginx configurado correctamente
4. ✅ SSL certificado activo
5. ✅ Webhook registrado en Evolution API
6. ✅ WhatsApp conectado
7. ✅ Bot respondiendo mensajes

## 🔗 URLs Importantes

### VPS del Chatbot:
- **Webhook**: https://chatbot.midominio.com/webhook
- **Health Check**: https://chatbot.midominio.com/healthz
- **Registro Webhook**: https://chatbot.midominio.com/register_webhook
- **Verificar Webhook**: https://chatbot.midominio.com/check_webhook
- **Sesiones**: https://chatbot.midominio.com/sessions

### VPS de Evolution API:
- **Evolution API**: https://evolution.midominio.com
- **Crear Instancia**: https://evolution.midominio.com/instance/create
- **Conectar WhatsApp**: https://evolution.midominio.com/instance/connect/xtalento-bot
- **Ver Instancias**: https://evolution.midominio.com/instance/fetchInstances

---

**Importante**: 
- Reemplaza `midominio.com` con tu dominio real
- Configura `tu-api-key-evolution` con tu API key real de Evolution API
- Asegúrate de que ambos VPS puedan comunicarse entre sí
- Evolution API debe poder enviar webhooks al chatbot (firewall y DNS configurados)
