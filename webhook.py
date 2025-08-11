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

# 1) Configuración
load_dotenv()
EVO_API_URL = os.getenv("EVO_API_URL", "http://localhost:8080")
EVO_APIKEY = os.getenv("EVO_APIKEY", "j.d1036448838")
EVO_INSTANCE = os.getenv("EVO_INSTANCE", "test")
PUBLIC_WEBHOOK_URL = os.getenv("PUBLIC_WEBHOOK_URL", "http://localhost:8000/webhook")
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

# Pool de hilos para procesar mensajes en background
EXECUTOR = ThreadPoolExecutor(max_workers=MAX_WORKERS)

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

def send_whatsapp_text(number: str, text: str, remote_jid: str | None = None) -> tuple[int, str]:
    """Envía texto intentando variantes de endpoint y payload según versión de Evolution.
    Soporta chats individuales (number) y grupos (groupJid/groupId).
    """
    endpoints = [
        f"{EVO_API_URL}/message/sendText/{EVO_INSTANCE}",
        f"{EVO_API_URL}/v2/message/sendText/{EVO_INSTANCE}",
    ]
    is_group = False
    group_jid_candidates: list[str] = []
    if remote_jid and isinstance(remote_jid, str) and remote_jid.endswith("@g.us"):
        is_group = True
        group_jid_candidates.append(remote_jid)
    if ("-" in (number or "")) and not group_jid_candidates:
        # reconstruye candidato de group jid si viene sin sufijo
        group_jid_candidates.append(f"{number}@g.us")

    payload_variants: list[dict] = []
    if is_group or group_jid_candidates:
        # Variantes para grupos
        for gj in group_jid_candidates:
            payload_variants.extend([
                {"groupJid": gj, "text": text},
                {"groupId": gj, "text": text},
                {"groupJid": gj, "options": {"presence": "composing"}, "textMessage": {"text": text}},
            ])
    # Variantes para chat 1:1
    payload_variants.extend([
        {"number": number, "text": text},  # muchas versiones requieren 'text' plano
        {"number": number, "message": text},  # alternativa legacy
        {"number": number, "options": {"presence": "composing"}, "textMessage": {"text": text}},  # variante moderna
    ])
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
        user_bot = get_user_bot(sender_number)
        reply_text = user_bot.process_message(text_in) or "🤖"
    except Exception as e:
        reply_text = "Lo siento, tuve un problema procesando tu mensaje."
        print("[BOT] ERROR (bg):", e)
    status, body = send_whatsapp_text(sender_number, reply_text)
    print(f"[SEND (bg)] -> {sender_number} [{status}] {body}")


def handle_message_async_with_remote(sender_number: str, text_in: str, remote_jid: str | None) -> None:
    try:
        user_bot = get_user_bot(sender_number)
        reply_text = user_bot.process_message(text_in) or "🤖"
    except Exception as e:
        reply_text = "Lo siento, tuve un problema procesando tu mensaje."
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
            EXECUTOR.submit(handle_message_async_with_remote, sender_number, text_in, remote_jid)
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

if __name__ == "__main__":
    # Opcional: registra automáticamente el webhook al iniciar
    # register_webhook()
    # Para mayor concurrencia en desarrollo, activar threaded=True
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True, use_reloader=False)
