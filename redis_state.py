"""
Funciones de manejo de estado Redis para el chatbot
Módulo separado para evitar importaciones circulares
"""

import redis
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv(override=True)

# Estados de conversación
STATE_BOT = "BOT"
STATE_HUMANO = "HUMANO"

# Configuración Redis
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)
REDIS_USERNAME = os.getenv('REDIS_USERNAME', 'default')
REDIS_SSL = os.getenv('REDIS_SSL', 'false').lower() == 'true'

# Cliente Redis global
_redis_client = None

def get_redis_client():
    """Obtiene el cliente Redis global."""
    global _redis_client
    if _redis_client is None:
        _redis_client = _create_redis_connection()
    return _redis_client

def _create_redis_connection():
    """Crea conexión a Redis Cloud usando la configuración que funciona (SIN SSL)."""
    
    print(f"[REDIS STATE] Conectando a {REDIS_HOST}:{REDIS_PORT}")
    
    try:
        print("[REDIS STATE] Usando conexión SIN SSL...")
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
        print("[REDIS STATE] ✅ Conexión exitosa SIN SSL")
        return r
    except Exception as e:
        print(f"[REDIS STATE ERROR] ❌ Conexión falló: {e}")
        return None

def set_conversation_state_from_bot(chat_id: str, state: str, reason: str = ""):
    """
    Establece el estado de una conversación desde el bot.
    Esta función es específica para cuando el bot detecta "agente".
    """
    client = get_redis_client()
    if not client:
        print("[REDIS STATE WARNING] Redis no disponible desde bot")
        return False
    
    try:
        state_key = f"chat_state:{chat_id}"
        current_time = int(time.time())
        
        state_data = {
            'state': state,
            'last_activity': str(current_time),
            'reason': reason,
            'updated_at': datetime.now().isoformat(),
            'changed_by': 'bot'
        }
        
        # Guardar con TTL de 24 horas
        client.hset(state_key, mapping=state_data)
        client.expire(state_key, 24 * 3600)
        
        print(f"[REDIS STATE SET] Estado {state} establecido por bot para {chat_id} (razón: {reason})")
        return True
    except Exception as e:
        print(f"[REDIS STATE ERROR] Error estableciendo estado para {chat_id}: {e}")
        return False

def get_conversation_state_from_bot(chat_id: str) -> str:
    """
    Obtiene el estado actual de una conversación desde el bot.
    """
    client = get_redis_client()
    if not client:
        return STATE_BOT
    
    try:
        state_key = f"chat_state:{chat_id}"
        state_data = client.hgetall(state_key)
        
        if state_data and 'state' in state_data:
            return state_data['state']
        else:
            return STATE_BOT
    except Exception as e:
        print(f"[REDIS STATE ERROR] Error obteniendo estado para {chat_id}: {e}")
        return STATE_BOT
