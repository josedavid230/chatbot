#!/usr/bin/env python3
"""
Script para despausar usuarios específicos o todos
"""

import sys
import os
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)

from redis_state import get_redis_client, STATE_BOT, STATE_HUMANO

def unpause_user(chat_id):
    """Despausar un usuario específico."""
    client = get_redis_client()
    if not client:
        print("❌ Error: No hay conexión Redis")
        return False
    
    try:
        state_key = f"chat_state:{chat_id}"
        
        # Verificar que existe y está pausado
        state_data = client.hgetall(state_key)
        if not state_data:
            print(f"❌ Usuario {chat_id} no encontrado")
            return False
        
        if state_data.get('state') != STATE_HUMANO:
            print(f"⚠️  Usuario {chat_id} no está pausado (estado: {state_data.get('state')})")
            return False
        
        # Cambiar a estado BOT
        current_time = int(time.time())
        reactivation_data = {
            'state': STATE_BOT,
            'last_activity': str(current_time),
            'reason': 'REACTIVACION_MANUAL',
            'updated_at': datetime.now().isoformat(),
            'changed_by': 'manual_unpause',
            'previous_state': STATE_HUMANO,
            'previous_reason': state_data.get('reason', 'UNKNOWN')
        }
        
        client.hset(state_key, mapping=reactivation_data)
        client.expire(state_key, 24 * 3600)
        
        print(f"✅ Usuario {chat_id} despausado exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error despausando {chat_id}: {e}")
        return False

def unpause_all_users():
    """Despausar todos los usuarios."""
    client = get_redis_client()
    if not client:
        print("❌ Error: No hay conexión Redis")
        return
    
    try:
        chat_keys = client.keys("chat_state:*")
        unpausados = 0
        
        for key in chat_keys:
            try:
                state_data = client.hgetall(key)
                
                if state_data.get('state') == STATE_HUMANO:
                    chat_id = key.replace("chat_state:", "")
                    if unpause_user(chat_id):
                        unpausados += 1
                        
            except Exception as e:
                print(f"⚠️  Error procesando {key}: {e}")
        
        print(f"\n🎉 Total usuarios despausados: {unpausados}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    if len(sys.argv) < 2:
        print("📋 USO:")
        print("  python unpause_user.py <numero>     - Despausar usuario específico")
        print("  python unpause_user.py all          - Despausar todos los usuarios")
        print()
        print("📱 EJEMPLOS:")
        print("  python unpause_user.py 573001234567")
        print("  python unpause_user.py all")
        return
    
    action = sys.argv[1].strip()
    
    if action.lower() == 'all':
        print("🔄 DESPAUSANDO TODOS LOS USUARIOS...")
        confirm = input("¿Estás seguro? (s/N): ").strip().lower()
        if confirm in ['s', 'si', 'sí', 'y', 'yes']:
            unpause_all_users()
        else:
            print("❌ Operación cancelada")
    else:
        print(f"🔄 DESPAUSANDO USUARIO: {action}")
        unpause_user(action)

if __name__ == "__main__":
    main()
