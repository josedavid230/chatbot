#!/usr/bin/env python3
"""
Tarea de reactivación automática para conversaciones en estado HUMANO
Se ejecuta cada 10 minutos para verificar conversaciones inactivas
"""

import time
import schedule
from datetime import datetime, timedelta
from redis_state import get_redis_client, STATE_BOT, STATE_HUMANO
from dotenv import load_dotenv
import os

# Cargar configuración
load_dotenv(override=True)

INACTIVITY_TIMEOUT_HOURS = int(os.getenv('INACTIVITY_TIMEOUT_HOURS', 1))

def reactivate_inactive_conversations():
    """
    Revisa todas las conversaciones en estado HUMANO y reactiva 
    aquellas que han estado inactivas por más del tiempo configurado.
    """
    print(f"\n[REACTIVATION] Verificando conversaciones inactivas (timeout: {INACTIVITY_TIMEOUT_HOURS}h)")
    
    client = get_redis_client()
    if not client:
        print("[REACTIVATION ERROR] No hay conexión Redis")
        return
    
    try:
        # Obtener todas las claves de estado de chat
        chat_keys = client.keys("chat_state:*")
        current_time = int(time.time())
        reactivated_count = 0
        
        for key in chat_keys:
            try:
                # Obtener datos del estado
                state_data = client.hgetall(key)
                
                if not state_data or state_data.get('state') != STATE_HUMANO:
                    continue
                
                # Verificar inactividad
                last_activity = int(state_data.get('last_activity', 0))
                inactive_seconds = current_time - last_activity
                inactive_hours = inactive_seconds / 3600
                
                if inactive_hours >= INACTIVITY_TIMEOUT_HOURS:
                    # Reactivar conversación
                    chat_id = key.replace("chat_state:", "")
                    
                    # Actualizar estado a BOT
                    reactivation_data = {
                        'state': STATE_BOT,
                        'last_activity': str(current_time),
                        'reason': f'REACTIVACION_AUTOMATICA_INACTIVIDAD_{inactive_hours:.1f}h',
                        'updated_at': datetime.now().isoformat(),
                        'changed_by': 'reactivation_task',
                        'previous_state': STATE_HUMANO,
                        'previous_reason': state_data.get('reason', 'UNKNOWN')
                    }
                    
                    client.hset(key, mapping=reactivation_data)
                    client.expire(key, 24 * 3600)  # Renovar TTL
                    
                    print(f"[REACTIVATED] {chat_id} - inactivo por {inactive_hours:.1f}h")
                    reactivated_count += 1
                else:
                    remaining_hours = INACTIVITY_TIMEOUT_HOURS - inactive_hours
                    chat_id = key.replace("chat_state:", "")
                    print(f"[PENDING] {chat_id} - {remaining_hours:.1f}h para reactivación")
                    
            except Exception as e:
                print(f"[REACTIVATION ERROR] Error procesando {key}: {e}")
                continue
        
        if reactivated_count > 0:
            print(f"[REACTIVATION SUMMARY] ✅ {reactivated_count} conversaciones reactivadas")
        else:
            print(f"[REACTIVATION SUMMARY] ⏳ No hay conversaciones para reactivar")
            
    except Exception as e:
        print(f"[REACTIVATION ERROR] Error general: {e}")

def get_conversation_stats():
    """Obtiene estadísticas de las conversaciones para monitoreo."""
    print(f"\n[STATS] Estadísticas de conversaciones")
    
    client = get_redis_client()
    if not client:
        print("[STATS ERROR] No hay conexión Redis")
        return
    
    try:
        chat_keys = client.keys("chat_state:*")
        
        stats = {
            'total': len(chat_keys),
            'bot': 0,
            'humano': 0,
            'inactive_soon': 0  # Próximas a reactivar
        }
        
        current_time = int(time.time())
        
        for key in chat_keys:
            try:
                state_data = client.hgetall(key)
                state = state_data.get('state', STATE_BOT)
                
                if state == STATE_BOT:
                    stats['bot'] += 1
                elif state == STATE_HUMANO:
                    stats['humano'] += 1
                    
                    # Verificar si está próximo a reactivar
                    last_activity = int(state_data.get('last_activity', 0))
                    inactive_hours = (current_time - last_activity) / 3600
                    
                    if inactive_hours >= (INACTIVITY_TIMEOUT_HOURS * 0.8):  # 80% del tiempo
                        stats['inactive_soon'] += 1
                        
            except Exception:
                continue
        
        print(f"[STATS] Total: {stats['total']}, BOT: {stats['bot']}, HUMANO: {stats['humano']}")
        print(f"[STATS] Próximas a reactivar: {stats['inactive_soon']}")
        
    except Exception as e:
        print(f"[STATS ERROR] Error: {e}")

def run_reactivation_service():
    """Ejecuta el servicio de reactivación."""
    print("🔄 SERVICIO DE REACTIVACIÓN INICIADO")
    print(f"⏰ Verificación cada 10 minutos")
    print(f"⌛ Timeout de inactividad: {INACTIVITY_TIMEOUT_HOURS} hora(s)")
    print("="*50)
    
    # Configurar tareas programadas
    schedule.every(10).minutes.do(reactivate_inactive_conversations)
    schedule.every(30).minutes.do(get_conversation_stats)
    
    # Ejecutar verificación inicial
    reactivate_inactive_conversations()
    get_conversation_stats()
    
    # Loop principal
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Verificar cada minuto
        except KeyboardInterrupt:
            print("\n[REACTIVATION] Servicio detenido por usuario")
            break
        except Exception as e:
            print(f"[REACTIVATION ERROR] Error en loop principal: {e}")
            time.sleep(60)

if __name__ == "__main__":
    run_reactivation_service()
