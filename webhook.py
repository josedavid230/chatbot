"""
Webhook especializado para Evolution API

Expone:
- POST /webhook           -> recibe eventos de Evolution (Baileys)
- POST /register_webhook  -> registra el webhook en Evolution API
- GET  /check_webhook     -> consulta configuración del webhook en Evolution API
- GET  /healthz           -> healthcheck

Variables de entorno:
- EVO_API_URL (ej. http://localhost:8080)
- EVO_APIKEY  (AUTHENTICATION_API_KEY)
- EVO_INSTANCE (nombre de la instancia en Evolution)
- PUBLIC_WEBHOOK_URL (URL pública hacia este /webhook)
"""

from flask import Flask, request, jsonify
from dotenv import load_dotenv
from main import Chatbot, load_vector_store, load_documents, create_vector_store
from threading import RLock
from concurrent.futures import ThreadPoolExecutor
import time
import os
import requests
import redis
import json
from datetime import datetime, timedelta

# Keyword para detección de intervención humana (mantener solo para fromMe)
HUMAN_INTERVENTION_KEYWORD = "Hola soy un agente de ventas de xtalento, gracias por escribir"

# ===== SISTEMA REDIS PARA ESTADOS DE CONVERSACIÓN =====

# Estados de conversación
STATE_BOT = "BOT"
STATE_HUMANO = "HUMANO"

# Configuración Redis
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)
REDIS_USERNAME = os.getenv('REDIS_USERNAME', 'default')
# Redis Cloud requiere SSL por defecto
REDIS_SSL = os.getenv('REDIS_SSL', 'true').lower() == 'true'

# Configuración de tiempos
INACTIVITY_TIMEOUT_HOURS = 1  # Tiempo para reactivar bot automáticamente
BOT_SENT_MESSAGE_IDS = set()  # Para detectar ecos del bot
BOT_SENT_MESSAGES_TIMING = {}  # Timestamp de mensajes del bot para detección por tiempo
BOT_SENT_MESSAGES_CONTENT = {}  # Contenido de mensajes del bot para detección por contenido

# Conexión Redis
def get_redis_connection():
    """Obtiene conexión a Redis Cloud usando la configuración que funciona (SIN SSL)."""
    
    print(f"[REDIS] Conectando a {REDIS_HOST}:{REDIS_PORT}")
    
    # Configuración que funciona (TEST 7: SIN SSL)
    try:
        print("[REDIS] Usando conexión SIN SSL...")
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            username=REDIS_USERNAME if REDIS_PASSWORD else None,
            decode_responses=True,
            ssl=False,
            socket_connect_timeout=15,
            socket_timeout=15,
            retry_on_timeout=True
        )
        r.ping()
        print("[REDIS] ✅ Conexión exitosa SIN SSL")
        return r
    except Exception as e:
        print(f"[REDIS ERROR] ❌ Conexión falló: {e}")
        print(f"[REDIS DEBUG] Host: {REDIS_HOST}, Port: {REDIS_PORT}")
        print(f"[REDIS DEBUG] Password configurado: {'Sí' if REDIS_PASSWORD else 'No'}")
        return None

# Inicializar conexión Redis
redis_client = get_redis_connection()

# ===== FUNCIONES CORE PARA MANEJO DE ESTADOS =====

def get_conversation_state(chat_id: str) -> str:
    """
    Obtiene el estado actual de una conversación.
    Retorna STATE_BOT por defecto si no existe.
    """
    if not redis_client:
        print("[REDIS WARNING] Redis no disponible, usando estado BOT por defecto")
        return STATE_BOT
    
    try:
        state_key = f"chat_state:{chat_id}"
        state_data = redis_client.hgetall(state_key)
        
        if state_data and 'state' in state_data:
            print(f"[REDIS GET] Estado para {chat_id}: {state_data['state']}")
            return state_data['state']
        else:
            print(f"[REDIS GET] No hay estado para {chat_id}, usando BOT por defecto")
            return STATE_BOT
    except Exception as e:
        print(f"[REDIS ERROR] Error obteniendo estado para {chat_id}: {e}")
        return STATE_BOT

def set_conversation_state(chat_id: str, state: str, reason: str = ""):
    """
    Establece el estado de una conversación en Redis.
    """
    if not redis_client:
        print("[REDIS WARNING] Redis no disponible, no se puede guardar estado")
        return False
    
    try:
        state_key = f"chat_state:{chat_id}"
        current_time = int(time.time())
        
        state_data = {
            'state': state,
            'last_activity': str(current_time),
            'reason': reason,
            'updated_at': datetime.now().isoformat()
        }
        
        # Guardar con TTL de 24 horas (auto-cleanup)
        redis_client.hset(state_key, mapping=state_data)
        redis_client.expire(state_key, 24 * 3600)  # 24 horas
        
        print(f"[REDIS SET] Estado {state} establecido para {chat_id} (razón: {reason})")
        return True
    except Exception as e:
        print(f"[REDIS ERROR] Error estableciendo estado para {chat_id}: {e}")
        return False

def update_last_activity(chat_id: str):
    """
    Actualiza el timestamp de última actividad para un chat.
    """
    if not redis_client:
        return False
    
    try:
        state_key = f"chat_state:{chat_id}"
        current_time = int(time.time())
        
        redis_client.hset(state_key, 'last_activity', str(current_time))
        print(f"[REDIS UPDATE] Actividad actualizada para {chat_id}")
        return True
    except Exception as e:
        print(f"[REDIS ERROR] Error actualizando actividad para {chat_id}: {e}")
        return False

def add_bot_message_info(message_id: str = None, content: str = "", sender_number: str = ""):
    """
    Agrega información de mensaje enviado por el bot para detección múltiple.
    """
    current_time = time.time()
    
    # Estrategia 1: Por ID (si está disponible)
    if message_id:
        BOT_SENT_MESSAGE_IDS.add(message_id)
        # Mantener solo los últimos 100 IDs
        if len(BOT_SENT_MESSAGE_IDS) > 100:
            BOT_SENT_MESSAGE_IDS.pop()
    
    # Estrategia 2: Por timing (ventana de tiempo)
    if sender_number:
        BOT_SENT_MESSAGES_TIMING[sender_number] = current_time
        # Limpiar entradas antiguas (más de 30 segundos)
        cutoff_time = current_time - 30
        BOT_SENT_MESSAGES_TIMING = {k: v for k, v in BOT_SENT_MESSAGES_TIMING.items() if v > cutoff_time}
    
    # Estrategia 3: Por contenido (hash del mensaje)
    if content and sender_number:
        content_hash = hash(content.strip().lower()[:100])  # Primeros 100 chars
        BOT_SENT_MESSAGES_CONTENT[f"{sender_number}:{content_hash}"] = current_time
        # Limpiar entradas antiguas (más de 30 segundos)
        cutoff_time = current_time - 30
        BOT_SENT_MESSAGES_CONTENT = {k: v for k, v in BOT_SENT_MESSAGES_CONTENT.items() if v > cutoff_time}

def add_bot_message_id(message_id: str):
    """
    Función legacy - wrapper para compatibilidad.
    """
    add_bot_message_info(message_id=message_id)

def is_bot_message_echo(message_id: str, chat_id: str = "", content: str = "") -> bool:
    """
    Verifica si un mensaje fromMe es un eco del bot usando múltiples estrategias robustas.
    """
    current_time = time.time()
    
    # Estrategia 1: Verificación por ID (método principal)
    if message_id and message_id in BOT_SENT_MESSAGE_IDS:
        BOT_SENT_MESSAGE_IDS.remove(message_id)
        print(f"[ECHO DETECTION] ✅ ID {message_id} confirmado como eco del bot (Estrategia ID)")
        return True
    
    # Estrategia 2: Verificación por timing (el bot envió un mensaje recientemente a este chat)
    if chat_id and chat_id in BOT_SENT_MESSAGES_TIMING:
        time_diff = current_time - BOT_SENT_MESSAGES_TIMING[chat_id]
        if time_diff <= 10:  # Ventana de 10 segundos
            print(f"[ECHO DETECTION] ✅ Timing coincide para {chat_id} (hace {time_diff:.1f}s) - Estrategia Timing")
            # Remover para evitar reutilización
            del BOT_SENT_MESSAGES_TIMING[chat_id]
            return True
    
    # Estrategia 3: Verificación por contenido (hash del mensaje)
    if content and chat_id:
        content_hash = hash(content.strip().lower()[:100])
        content_key = f"{chat_id}:{content_hash}"
        if content_key in BOT_SENT_MESSAGES_CONTENT:
            time_diff = current_time - BOT_SENT_MESSAGES_CONTENT[content_key]
            if time_diff <= 10:  # Ventana de 10 segundos
                print(f"[ECHO DETECTION] ✅ Contenido coincide para {chat_id} (hace {time_diff:.1f}s) - Estrategia Contenido")
                # Remover para evitar reutilización
                del BOT_SENT_MESSAGES_CONTENT[content_key]
                return True
    
    # Estrategia 4: Verificación por estado Redis (si el chat está en estado BOT, es probable que sea eco)
    try:
        conversation_state = get_conversation_state(chat_id)
        if conversation_state == STATE_BOT and chat_id:
            # Si está en estado BOT y recibimos un fromMe, muy probablemente es eco del bot
            recent_timing = chat_id in BOT_SENT_MESSAGES_TIMING and (current_time - BOT_SENT_MESSAGES_TIMING[chat_id]) <= 30
            if recent_timing:
                print(f"[ECHO DETECTION] ✅ Estado BOT + timing reciente para {chat_id} - Estrategia Estado")
                return True
    except:
        pass
    
    print(f"[ECHO DETECTION] ❌ NO es eco del bot confirmado")
    print(f"[ECHO DETECTION] ID: {message_id}, Chat: {chat_id}, Contenido: {content[:50]}...")
    print(f"[ECHO DETECTION] IDs conocidos: {list(BOT_SENT_MESSAGE_IDS)}")
    print(f"[ECHO DETECTION] Timings: {BOT_SENT_MESSAGES_TIMING}")
    return False

def should_change_to_human_state(chat_id: str, message_data: dict) -> tuple[bool, str]:
    """
    Determina si se debe cambiar a estado HUMANO y por qué razón.
    
    Returns:
        (should_change: bool, reason: str)
    """
    reasons = []
    
    # Caso 1: Mensaje fromMe que no es eco del bot (agente interviene)
    if message_data.get('fromMe', False):
        msg_id = message_data.get('id', '')
        print(f"[HUMAN DETECTION] Verificando mensaje fromMe. ID: {msg_id}")
        print(f"[HUMAN DETECTION] IDs del bot en memoria: {list(BOT_SENT_MESSAGE_IDS)}")
        
        # Extraer contenido y chat_id para detección robusta
        agent_text = message_data.get('text', '')
        is_echo = is_bot_message_echo(msg_id, chat_id, agent_text)
        print(f"[HUMAN DETECTION] ¿Es eco del bot? {is_echo}")
        
        if not is_echo:
            print(f"[HUMAN DETECTION] ❌ NO es eco del bot → DETECTANDO AGENTE_INTERVIENE")
            reasons.append("AGENTE_INTERVIENE")
        else:
            print(f"[HUMAN DETECTION] ✅ ES eco del bot → Ignorando mensaje")
    
    # CASO 2 ELIMINADO: Los clientes (fromMe=false) NO activan intervención humana aquí
    # La detección de "agente" en mensajes de cliente se maneja SOLO en main.py
    # El webhook SOLO detecta intervención de agentes reales (fromMe=true)
    
    # Caso 3: Keyword específica de agente
    text = message_data.get('text', '').lower().strip()
    if (message_data.get('fromMe', False) and 
        HUMAN_INTERVENTION_KEYWORD.lower() in text):
        reasons.append("AGENTE_KEYWORD")
    
    if reasons:
        return True, " + ".join(reasons)
    
    return False, ""

# 1) Configuración
load_dotenv(override=True)
EVO_API_URL = os.getenv("EVO_API_URL", "https://evolution-api-domain.com")
EVO_APIKEY = os.getenv("EVO_APIKEY", "j.d1036448838")
EVO_INSTANCE = os.getenv("EVO_INSTANCE", "test")
PUBLIC_WEBHOOK_URL = os.getenv("PUBLIC_WEBHOOK_URL", "https://tu-dominio.com/webhook")
WEBHOOK_BY_EVENTS = os.getenv("WEBHOOK_BY_EVENTS", "false").strip().lower() in ("1", "true", "yes")
MAX_WORKERS = int(os.getenv("WEBHOOK_MAX_WORKERS", "16"))

app = Flask(__name__)

# 2) Vectorstore compartido + sesiones por número
def _ensure_vectorstore():
    vector = load_vector_store()
    if vector is None:
        try:
            print("[INIT] Vectorstore no encontrado. Creándolo...")
            docs = load_documents()
            vector = create_vector_store(docs)
            print("[INIT] Vectorstore creado.")
        except Exception as err:
            print("[INIT] Error creando vectorstore:", err)
            vector = None
    return vector

VECTORSTORE = _ensure_vectorstore()

_user_bots: dict[str, Chatbot] = {}
_bots_lock = RLock()

# Sistema de bloqueo temporal para usuarios que solicitan agente humano
_blocked_users: dict[str, datetime] = {}
_blocked_lock = RLock()
BLOCK_DURATION_HOURS = 4

# Pool de hilos para procesar mensajes en background
EXECUTOR = ThreadPoolExecutor(max_workers=MAX_WORKERS)

def is_user_blocked(sender_number: str) -> bool:
    """Verifica si el usuario está bloqueado temporalmente."""
    key = sender_number or "anonymous"
    with _blocked_lock:
        if key in _blocked_users:
            block_time = _blocked_users[key]
            # Verificar si han pasado las 4 horas
            if datetime.now() - block_time >= timedelta(hours=BLOCK_DURATION_HOURS):
                # El bloqueo expiró, remover al usuario
                del _blocked_users[key]
                print(f"[UNBLOCK] Usuario {sender_number} desbloqueado automáticamente")
                return False
            else:
                # Usuario sigue bloqueado
                remaining = block_time + timedelta(hours=BLOCK_DURATION_HOURS) - datetime.now()
                print(f"[BLOCKED] Usuario {sender_number} bloqueado por {remaining}")
                return True
        return False

def block_user(sender_number: str):
    """Bloquea temporalmente al usuario por 4 horas."""
    key = sender_number or "anonymous"
    with _blocked_lock:
        _blocked_users[key] = datetime.now()
        print(f"[BLOCK] Usuario {sender_number} bloqueado por {BLOCK_DURATION_HOURS} horas")

def get_user_bot(sender_number: str) -> Chatbot:
    """Devuelve un bot por número; crea uno nuevo si no existe (memoria aislada por usuario)."""
    key = sender_number or "anonymous"
    with _bots_lock:
        bot = _user_bots.get(key)
        if bot is None:
            bot = Chatbot(VECTORSTORE, chat_id=sender_number)
            _user_bots[key] = bot
    return bot

# 3) Utilidades Evolution
def _auth_headers() -> dict:
    return {"Content-Type": "application/json", "apikey": EVO_APIKEY}

def _jid_to_number(jid: str) -> str:
    if not jid:
        return ""
    return jid.split("@")[0]

def _extract_text_from_baileys(message_obj: dict) -> str | None:
    if not message_obj:
        return None
    # Soporta dos formas: { message: {...} } y campos de texto directos
    message = message_obj.get("message", {}) or {}
    if "conversation" in message:
        return message.get("conversation")
    ext = message.get("extendedTextMessage")
    if isinstance(ext, dict) and "text" in ext:
        return ext.get("text")
    btn = message.get("buttonsResponseMessage")
    if isinstance(btn, dict) and "selectedButtonId" in btn:
        return btn.get("selectedButtonId")
    lst = message.get("listResponseMessage")
    if isinstance(lst, dict):
        sel = lst.get("singleSelectReply") or {}
        return sel.get("selectedRowId")
    # Fallbacks cuando no hay 'message' (payload plano por eventos)
    if isinstance(message_obj, dict):
        for field in ("text", "body", "caption"):
            if isinstance(message_obj.get(field), str) and message_obj.get(field):
                return message_obj.get(field)
    return None

def send_whatsapp_text(number: str, text: str) -> tuple[int, str]:
    """Envía texto (solo chats 1:1) probando variantes de endpoint y payload según versión de Evolution."""
    endpoints = [
        f"{EVO_API_URL}/message/sendText/{EVO_INSTANCE}",
        f"{EVO_API_URL}/v2/message/sendText/{EVO_INSTANCE}",
    ]
    payload_variants: list[dict] = [
        {"number": number, "text": text},  # muchas versiones requieren 'text' plano
        {"number": number, "message": text},  # alternativa legacy
        {"number": number, "options": {"presence": "composing"}, "textMessage": {"text": text}},  # variante moderna
    ]
    retry_statuses = set([429] + list(range(500, 600)))
    last_status, last_text = 0, ""
    for attempt in range(3):
        for url in endpoints:
            for payload in payload_variants:
                try:
                    print(f"[SEND TRY] POST {url} payload_keys={list(payload.keys())}")
                    r = requests.post(url, headers=_auth_headers(), json=payload, timeout=30)
                    print(f"[SEND RESP] {r.status_code} -> {r.text[:300]}")
                    last_status, last_text = r.status_code, r.text
                    if r.status_code in (200, 201):
                        return last_status, last_text
                    # Para 4xx intentamos otras variantes antes de rendirnos
                    if r.status_code in retry_statuses:
                        continue
                except Exception as e:
                    last_status, last_text = 0, str(e)
                    print("[SEND ERROR]", e)
        # backoff exponencial entre intentos
        sleep_s = 0.5 * (2 ** attempt)
        time.sleep(sleep_s)
    return last_status, last_text


def handle_message_async(sender_number: str, text_in: str) -> None:
    """Procesa el mensaje y envía la respuesta en background."""
    try:
        # Verificar estado actual antes de procesar (puede haber cambiado)
        current_state = get_conversation_state(sender_number)
        if current_state == STATE_HUMANO:
            print(f"[SKIP] Usuario {sender_number} está en estado HUMANO - no responder")
            return
        
        # Verificar si el usuario está bloqueado temporalmente (sistema legacy)
        if is_user_blocked(sender_number):
            print(f"[SKIP] Usuario {sender_number} está bloqueado temporalmente")
            return
        
        user_bot = get_user_bot(sender_number)
        reply_text = user_bot.process_message(text_in) or "🤖"
        
        # Verificar nuevamente el estado después del procesamiento
        # (el bot puede haber detectado "agente" y cambiado el estado)
        final_state = get_conversation_state(sender_number)
        if final_state == STATE_HUMANO:
            print(f"[STATE CHANGED] Usuario {sender_number} cambió a HUMANO durante procesamiento")
            # Si el mensaje es de conexión con agente, enviarlo y luego silenciar
            if "agente" in reply_text.lower():
                print(f"[SENDING AGENT MESSAGE] Enviando mensaje de conexión a {sender_number}")
            else:
                # Si cambió a HUMANO por otra razón, no enviar respuesta
                print(f"[SKIP SEND] No enviar respuesta, estado es HUMANO")
                return
        
        # Detectar si el bot activó el modo agente humano (sistema legacy)
        if "Perfecto. Te conecto con un agente humano inmediatamente" in reply_text:
            print(f"[AGENT MODE] Bloqueando usuario {sender_number} por {BLOCK_DURATION_HOURS} horas")
            block_user(sender_number)
            
    except Exception as e:
        reply_text = "Lo siento, tuve un problema procesando tu mensaje. Si quieres comunicarte con un humano, menciona la palabra 'agente' en el chat."
        print("[BOT] ERROR (bg):", e)
    
    # Enviar mensaje
    status, body = send_whatsapp_text(sender_number, reply_text)
    print(f"[SEND (bg)] -> {sender_number} [{status}] {body}")
    
    # Registrar información del mensaje del bot para detección robusta
    if status in (200, 201):
        try:
            # Estrategia principal: Extraer ID del mensaje de la respuesta
            import json
            response_data = json.loads(body)
            print(f"[BOT MSG DEBUG] Response body structure: {body[:200]}...")
            
            msg_id = None
            if isinstance(response_data, dict) and 'key' in response_data:
                msg_id = response_data['key'].get('id')
                if msg_id:
                    print(f"[BOT MSG ID] ✅ ID extraído: {msg_id}")
                else:
                    print(f"[BOT MSG ID] ⚠️ No se encontró 'id' en key: {response_data.get('key')}")
            else:
                print(f"[BOT MSG ID] ⚠️ Estructura inesperada. Keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'No es dict'}")
            
            # Registrar con estrategias múltiples (ID + timing + contenido)
            add_bot_message_info(
                message_id=msg_id,
                content=reply_text,
                sender_number=sender_number
            )
            print(f"[BOT MSG] ✅ Información registrada para detección robusta")
                
        except Exception as e:
            print(f"[BOT MSG ID] ❌ Error extrayendo ID: {e}")
            print(f"[BOT MSG ID] Body problemático: {body}")
            # Aún registrar timing y contenido aunque falle la extracción de ID
            add_bot_message_info(
                content=reply_text,
                sender_number=sender_number
            )
    else:
        print(f"[BOT MSG ID] ❌ Envío falló con status {status}, no se puede extraer ID")
        # No registrar nada si el envío falló


def handle_message_async_with_remote(sender_number: str, text_in: str, remote_jid: str | None) -> None:
    try:
        user_bot = get_user_bot(sender_number)
        reply_text = user_bot.process_message(text_in) or "🤖"
    except Exception as e:
        reply_text = "Lo siento, tuve un problema procesando tu mensaje. Si quieres comunicarte con un humano, menciona la palabra 'agente' en el chat."
        print("[BOT] ERROR (bg):", e)
    status, body = send_whatsapp_text(sender_number, reply_text, remote_jid=remote_jid)
    print(f"[SEND (bg)] -> {sender_number} [{status}] {body}")

def register_webhook() -> tuple[int, str]:
    """Registra el webhook en Evolution API, probando rutas v2 y legacy."""
    webhook_cfg = {
        "enabled": True,
        "url": PUBLIC_WEBHOOK_URL,
        "webhookByEvents": WEBHOOK_BY_EVENTS,
        "webhookBase64": False,
        "events": [
            "QRCODE_UPDATED",
            "CONNECTION_UPDATE",
            "MESSAGES_UPSERT",
            "MESSAGES_UPDATE",
            "MESSAGES_DELETE",
            "SEND_MESSAGE",
        ],
    }
    # Algunos despliegues requieren que el payload esté dentro de la propiedad 'webhook'
    payload_variants = [
        webhook_cfg,
        {"webhook": webhook_cfg},
    ]
    candidates = [
        f"{EVO_API_URL}/webhook/set/{EVO_INSTANCE}",
        f"{EVO_API_URL}/v2/webhook/set/{EVO_INSTANCE}",
        f"{EVO_API_URL}/instance/{EVO_INSTANCE}/webhook",
        f"{EVO_API_URL}/instances/{EVO_INSTANCE}/webhook",
    ]
    last_status, last_text = 0, ""
    for url in candidates:
        for payload in payload_variants:
            try:
                print(f"[REGISTER] POST {url} payload_keys={list(payload.keys())}")
                r = requests.post(url, headers=_auth_headers(), json=payload, timeout=20)
                print(f"[REGISTER] RESP {r.status_code} -> {r.text[:300]}")
                last_status, last_text = r.status_code, r.text
                if r.status_code in (200, 201):
                    return last_status, last_text
            except Exception as e:
                last_status, last_text = 0, str(e)
                print("[REGISTER] ERROR", e)
    return last_status, last_text

def find_webhook() -> tuple[int, str]:
    candidates = [
        f"{EVO_API_URL}/webhook/find/{EVO_INSTANCE}",
        f"{EVO_API_URL}/v2/webhook/find/{EVO_INSTANCE}",
    ]
    last_status, last_text = 0, ""
    for url in candidates:
        try:
            print(f"[FIND] GET {url}")
            r = requests.get(url, headers={"apikey": EVO_APIKEY}, timeout=20)
            last_status, last_text = r.status_code, r.text
            if r.status_code == 200:
                return last_status, last_text
        except Exception as e:
            last_status, last_text = 0, str(e)
            print("[FIND] ERROR", e)
    return last_status, last_text

# 4) Endpoints
@app.get("/healthz")
def healthz():
    return jsonify({"ok": True, "instance": EVO_INSTANCE}), 200

@app.post("/register_webhook")
def register_webhook_endpoint():
    status, body = register_webhook()
    return jsonify({"ok": status in (200, 201), "status": status, "body": body}), 200

@app.get("/check_webhook")
def check_webhook_endpoint():
    status, body = find_webhook()
    return jsonify({"status": status, "body": body}), 200

# Endpoints del sistema anterior eliminados - serán reemplazados por sistema Redis

@app.post("/webhook")
def webhook():
    """Recibe eventos Evolution y responde 200 rápidamente."""
    try:
        payload = request.get_json(force=True, silent=True) or {}
        event_raw = payload.get("event") or payload.get("type") or ""
        event = str(event_raw).upper().replace(".", "_")
        print("[WEBHOOK INCOMING] event=", event, "keys=", list(payload.keys()))

        # Extrae mensajes desde distintas variantes de payload
        data_obj = payload.get("data")
        messages = []
        if isinstance(data_obj, dict):
            if isinstance(data_obj.get("messages"), list):
                messages = data_obj.get("messages")
            elif ("message" in data_obj) or ("key" in data_obj) or ("text" in data_obj) or ("body" in data_obj):
                messages = [data_obj]
        elif isinstance(data_obj, list):
            messages = data_obj
        elif isinstance(payload.get("messages"), list):
            messages = payload.get("messages")

        if event in ("MESSAGES_UPSERT", "MESSAGES_UPDATE") or (messages and "message" in (messages[0] or {})):
            if not messages:
                print("[WEBHOOK] No messages array found")
                return jsonify({"ok": True, "skip": "no-messages"}), 200

            msg = messages[0] or {}
            key = msg.get("key", {}) or {}
            if key.get("fromMe", False):
                # Obtener información del mensaje para detectar intervención humana
                remote_jid = key.get("remoteJid") or msg.get("from") or payload.get("sender") or ""
                client_number = _jid_to_number(str(remote_jid).split(":")[0])
                msg_id = key.get("id", "")
                agent_text = _extract_text_from_baileys(msg)
                
                # Crear datos del mensaje para should_change_to_human_state
                message_data = {
                    'fromMe': True,
                    'text': agent_text or "",
                    'id': msg_id
                }
                
                # Verificar si debe cambiar a estado HUMANO
                should_change, reason = should_change_to_human_state(client_number, message_data)
                
                if should_change:
                    print(f"[HUMAN TAKEOVER] {reason} en chat {client_number}")
                    
                    # Cambiar estado a HUMANO en Redis
                    success = set_conversation_state(client_number, STATE_HUMANO, reason)
                    if success:
                        print(f"[REDIS] Estado cambiado a HUMANO para {client_number}")
                    else:
                        print(f"[REDIS ERROR] No se pudo cambiar estado para {client_number}")
                    
                    return jsonify({"ok": True, "human_intervention": True, "reason": reason}), 200
                
                # Si no es intervención humana, ignorar (eco del bot u otro mensaje)
                return jsonify({"ok": True, "skip": "fromMe_no_intervention"}), 200

            remote_jid = key.get("remoteJid") or msg.get("from") or payload.get("sender") or ""
            sender_number = _jid_to_number(str(remote_jid).split(":")[0])
            text_in = _extract_text_from_baileys(msg)
            print(f"[WEBHOOK PARSED] number={sender_number} text={text_in!r}")
            if not text_in:
                return jsonify({"ok": True, "skip": "no-text"}), 200

            # Verificar estado actual de la conversación
            current_state = get_conversation_state(sender_number)
            print(f"[REDIS STATE] {sender_number} -> {current_state}")
            
            if current_state == STATE_HUMANO:
                # Si está en estado HUMANO, el bot debe permanecer en silencio
                print(f"[BOT SILENT] Chat {sender_number} está en estado HUMANO - bot en silencio")
                
                # Actualizar última actividad para evitar reactivación prematura
                update_last_activity(sender_number)
                
                return jsonify({"ok": True, "skip": "human_state"}), 200
            
            # Los mensajes de clientes (fromMe=false) NO activan intervención humana en webhook
            # La detección de "agente" se maneja ÚNICAMENTE en main.py (proceso del bot)
            # El webhook solo detecta intervención real de agentes humanos (fromMe=true)

            # Estado BOT normal - procesar mensaje
            EXECUTOR.submit(handle_message_async, sender_number, text_in)
            print(f"[ENQUEUED] bot reply task for {sender_number}")

        elif event in ("QRCODE_UPDATED", "CONNECTION_UPDATE"):
            print("[EVOLUTION]", event, payload.get("data"))

        return jsonify({"ok": True}), 200
    except Exception as e:
        print("[WEBHOOK] ERROR:", e)
        return jsonify({"ok": False, "error": str(e)}), 200

# Sesiones: utilidades opcionales
@app.get("/sessions")
def list_sessions():
    with _bots_lock:
        return jsonify({"sessions": list(_user_bots.keys())}), 200

@app.delete("/sessions")
def clear_sessions():
    with _bots_lock:
        _user_bots.clear()
    return jsonify({"ok": True, "cleared": True}), 200

# Gestión de usuarios bloqueados (sistema legacy)
@app.get("/blocked_users")
def list_blocked_users():
    """Lista usuarios bloqueados y tiempo restante."""
    with _blocked_lock:
        blocked_info = {}
        current_time = datetime.now()
        for user, block_time in _blocked_users.items():
            remaining = block_time + timedelta(hours=BLOCK_DURATION_HOURS) - current_time
            if remaining.total_seconds() > 0:
                blocked_info[user] = {
                    "blocked_at": block_time.isoformat(),
                    "remaining_seconds": int(remaining.total_seconds()),
                    "remaining_readable": str(remaining).split('.')[0]
                }
        return jsonify({"blocked_users": blocked_info}), 200

# Gestión de usuarios pausados (sistema Redis)
@app.get("/paused_users")
def list_paused_users():
    """Lista usuarios en estado HUMANO (pausados)."""
    if not redis_client:
        return jsonify({"error": "Redis no disponible"}), 500
    
    try:
        chat_keys = redis_client.keys("chat_state:*")
        current_time = int(time.time())
        paused_users = {}
        
        for key in chat_keys:
            try:
                state_data = redis_client.hgetall(key)
                
                if state_data.get('state') == STATE_HUMANO:
                    chat_id = key.replace("chat_state:", "")
                    last_activity = int(state_data.get('last_activity', 0))
                    paused_seconds = current_time - last_activity
                    
                    paused_users[chat_id] = {
                        "reason": state_data.get('reason', 'UNKNOWN'),
                        "paused_seconds": paused_seconds,
                        "paused_readable": str(timedelta(seconds=paused_seconds)).split('.')[0],
                        "updated_at": state_data.get('updated_at', 'UNKNOWN'),
                        "changed_by": state_data.get('changed_by', 'UNKNOWN')
                    }
                    
            except Exception as e:
                print(f"[PAUSED USERS ERROR] Error procesando {key}: {e}")
                continue
        
        return jsonify({
            "paused_users": paused_users,
            "total_paused": len(paused_users)
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/unpause_user/<user_number>")
def unpause_specific_user(user_number: str):
    """Despausar un usuario específico."""
    if not redis_client:
        return jsonify({"error": "Redis no disponible"}), 500
    
    try:
        state_key = f"chat_state:{user_number}"
        state_data = redis_client.hgetall(state_key)
        
        if not state_data:
            return jsonify({"error": "Usuario no encontrado"}), 404
        
        if state_data.get('state') != STATE_HUMANO:
            return jsonify({
                "error": "Usuario no está pausado",
                "current_state": state_data.get('state')
            }), 400
        
        # Cambiar a estado BOT
        current_time = int(time.time())
        reactivation_data = {
            'state': STATE_BOT,
            'last_activity': str(current_time),
            'reason': 'REACTIVACION_MANUAL_ENDPOINT',
            'updated_at': datetime.now().isoformat(),
            'changed_by': 'webhook_endpoint',
            'previous_state': STATE_HUMANO,
            'previous_reason': state_data.get('reason', 'UNKNOWN')
        }
        
        redis_client.hset(state_key, mapping=reactivation_data)
        redis_client.expire(state_key, 24 * 3600)
        
        return jsonify({
            "ok": True,
            "message": f"Usuario {user_number} despausado exitosamente",
            "previous_reason": state_data.get('reason')
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/unpause_all_users")
def unpause_all_users():
    """Despausar todos los usuarios."""
    if not redis_client:
        return jsonify({"error": "Redis no disponible"}), 500
    
    try:
        chat_keys = redis_client.keys("chat_state:*")
        unpausados = 0
        errores = []
        
        for key in chat_keys:
            try:
                state_data = redis_client.hgetall(key)
                
                if state_data.get('state') == STATE_HUMANO:
                    chat_id = key.replace("chat_state:", "")
                    
                    # Cambiar a estado BOT
                    current_time = int(time.time())
                    reactivation_data = {
                        'state': STATE_BOT,
                        'last_activity': str(current_time),
                        'reason': 'REACTIVACION_MASIVA_ENDPOINT',
                        'updated_at': datetime.now().isoformat(),
                        'changed_by': 'webhook_endpoint',
                        'previous_state': STATE_HUMANO,
                        'previous_reason': state_data.get('reason', 'UNKNOWN')
                    }
                    
                    redis_client.hset(key, mapping=reactivation_data)
                    redis_client.expire(key, 24 * 3600)
                    unpausados += 1
                    
            except Exception as e:
                errores.append(f"{key}: {str(e)}")
        
        return jsonify({
            "ok": True,
            "unpausados": unpausados,
            "errores": errores
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.delete("/blocked_users")
def clear_blocked_users():
    """Limpia todos los bloqueos (para emergencias)."""
    with _blocked_lock:
        count = len(_blocked_users)
        _blocked_users.clear()
    return jsonify({"ok": True, "unblocked_count": count}), 200

@app.delete("/blocked_users/<user_number>")
def unblock_specific_user(user_number: str):
    """Desbloquea un usuario específico."""
    with _blocked_lock:
        if user_number in _blocked_users:
            del _blocked_users[user_number]
            return jsonify({"ok": True, "unblocked": user_number}), 200
        else:
            return jsonify({"ok": False, "error": "Usuario no estaba bloqueado"}), 404

if __name__ == "__main__":
    # Opcional: registra automáticamente el webhook al iniciar
    # register_webhook()
    # Para desarrollo local usar: app.run(host="0.0.0.0", port=8000, debug=False, threaded=True, use_reloader=False)
    # Para producción en VPS usar: waitress-serve --listen=0.0.0.0:8000 webhook:app
    print("=== MODO PRODUCCIÓN VPS ===")
    print("Ejecutar: waitress-serve --listen=0.0.0.0:8000 webhook:app")
    print("O usar: python webhook.py para auto-configuración")
    print("Asegúrate de configurar las variables de entorno en .env")
