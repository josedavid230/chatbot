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
from datetime import datetime, timedelta

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
            bot = Chatbot(VECTORSTORE)
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
        # Verificar si el usuario está bloqueado
        if is_user_blocked(sender_number):
            print(f"[SKIP] Usuario {sender_number} está bloqueado temporalmente")
            return
        
        user_bot = get_user_bot(sender_number)
        reply_text = user_bot.process_message(text_in) or "🤖"
        
        # Detectar si el bot activó el modo agente humano
        if "Perfecto. Pauso este chat" in reply_text:
            print(f"[AGENT MODE] Bloqueando usuario {sender_number} por {BLOCK_DURATION_HOURS} horas")
            block_user(sender_number)
            
    except Exception as e:
        reply_text = "Lo siento, tuve un problema procesando tu mensaje. Si quieres comunicarte con un humano, menciona la palabra 'agente' en el chat."
        print("[BOT] ERROR (bg):", e)
    
    status, body = send_whatsapp_text(sender_number, reply_text)
    print(f"[SEND (bg)] -> {sender_number} [{status}] {body}")


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
                return jsonify({"ok": True, "skip": "fromMe"}), 200

            remote_jid = key.get("remoteJid") or msg.get("from") or payload.get("sender") or ""
            sender_number = _jid_to_number(str(remote_jid).split(":")[0])
            text_in = _extract_text_from_baileys(msg)
            print(f"[WEBHOOK PARSED] number={sender_number} text={text_in!r}")
            if not text_in:
                return jsonify({"ok": True, "skip": "no-text"}), 200

            # Encolar procesamiento en background para responder sin bloquear el webhook
            EXECUTOR.submit(handle_message_async, sender_number, text_in)
            print(f"[ENQUEUED] reply task for {sender_number}")

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

# Gestión de usuarios bloqueados
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
