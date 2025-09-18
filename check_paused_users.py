#!/usr/bin/env python3
"""
Script para ver usuarios pausados (en estado HUMANO)
"""

import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv(override=True)

from redis_state import get_redis_client, STATE_BOT, STATE_HUMANO

def check_paused_users():
    """Muestra todos los usuarios en estado HUMANO."""
    print("👥 USUARIOS PAUSADOS (ESTADO HUMANO)")
    print("="*60)
    
    client = get_redis_client()
    if not client:
        print("❌ Error: No hay conexión Redis")
        return
    
    try:
        # Obtener todas las claves de estado
        chat_keys = client.keys("chat_state:*")
        current_time = int(time.time())
        
        paused_users = []
        
        for key in chat_keys:
            try:
                state_data = client.hgetall(key)
                
                if state_data.get('state') == STATE_HUMANO:
                    chat_id = key.replace("chat_state:", "")
                    last_activity = int(state_data.get('last_activity', 0))
                    reason = state_data.get('reason', 'UNKNOWN')
                    updated_at = state_data.get('updated_at', 'UNKNOWN')
                    
                    # Calcular tiempo pausado
                    paused_seconds = current_time - last_activity
                    paused_time = str(timedelta(seconds=paused_seconds)).split('.')[0]
                    
                    paused_users.append({
                        'chat_id': chat_id,
                        'reason': reason,
                        'paused_time': paused_time,
                        'updated_at': updated_at,
                        'last_activity': last_activity
                    })
                    
            except Exception as e:
                print(f"⚠️  Error procesando {key}: {e}")
        
        if paused_users:
            print(f"📊 Total usuarios pausados: {len(paused_users)}")
            print()
            
            for i, user in enumerate(paused_users, 1):
                print(f"{i}. 📱 {user['chat_id']}")
                print(f"   📝 Razón: {user['reason']}")
                print(f"   ⏰ Pausado por: {user['paused_time']}")
                print(f"   📅 Actualizado: {user['updated_at']}")
                print()
        else:
            print("✅ No hay usuarios pausados actualmente")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_paused_users()
