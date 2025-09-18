#!/usr/bin/env python3
"""
Script de prueba para las funciones Redis del sistema de estados.
"""

import sys
import os

# Configurar path y variables de entorno ANTES de importar
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from dotenv import load_dotenv
# Cargar .env desde el directorio actual
load_dotenv(os.path.join(current_dir, '.env'), override=True)

print(f"📁 Directorio actual: {current_dir}")
print(f"🔐 REDIS_HOST: {os.getenv('REDIS_HOST', 'NO_CONFIGURADO')}")
print(f"🔐 REDIS_PORT: {os.getenv('REDIS_PORT', 'NO_CONFIGURADO')}")
print(f"🔐 REDIS_PASSWORD configurado: {'Sí' if os.getenv('REDIS_PASSWORD') else 'No'}")
print(f"🔐 REDIS_USERNAME: {os.getenv('REDIS_USERNAME', 'NO_CONFIGURADO')}")
print(f"🔐 REDIS_SSL: {os.getenv('REDIS_SSL', 'NO_CONFIGURADO')}")

print("\n" + "="*50)
print("🔄 IMPORTANDO FUNCIONES DEL WEBHOOK...")
print("="*50)

# Importar funciones del webhook (esto activará la nueva conexión Redis)
from webhook import (
    redis_client, 
    get_conversation_state, 
    set_conversation_state,
    update_last_activity,
    add_bot_message_id,
    is_bot_message_echo,
    should_change_to_human_state,
    STATE_BOT,
    STATE_HUMANO
)

print(f"✅ Importación completada. Redis client: {'Conectado' if redis_client else 'No conectado'}")
print("="*50)

def test_redis_connection():
    """Probar conexión a Redis"""
    print("=== TEST 1: CONEXIÓN REDIS ===")
    if redis_client:
        try:
            response = redis_client.ping()
            print(f"✅ Redis conectado: {response}")
            return True
        except Exception as e:
            print(f"❌ Error de conexión: {e}")
            return False
    else:
        print("❌ redis_client es None")
        return False

def test_conversation_states():
    """Probar funciones de estado de conversación"""
    print("\n=== TEST 2: ESTADOS DE CONVERSACIÓN ===")
    
    test_chat_id = "test_573001318931@s.whatsapp.net"
    
    # Test 1: Estado inicial (debe ser BOT)
    initial_state = get_conversation_state(test_chat_id)
    print(f"Estado inicial: {initial_state}")
    assert initial_state == STATE_BOT, f"Esperado {STATE_BOT}, obtenido {initial_state}"
    
    # Test 2: Cambiar a HUMANO
    success = set_conversation_state(test_chat_id, STATE_HUMANO, "TEST_AGENTE_INTERVIENE")
    print(f"Cambio a HUMANO exitoso: {success}")
    assert success, "No se pudo cambiar estado a HUMANO"
    
    # Test 3: Verificar estado HUMANO
    current_state = get_conversation_state(test_chat_id)
    print(f"Estado actual: {current_state}")
    assert current_state == STATE_HUMANO, f"Esperado {STATE_HUMANO}, obtenido {current_state}"
    
    # Test 4: Actualizar actividad
    activity_success = update_last_activity(test_chat_id)
    print(f"Actualización de actividad exitosa: {activity_success}")
    assert activity_success, "No se pudo actualizar actividad"
    
    # Test 5: Volver a BOT
    success = set_conversation_state(test_chat_id, STATE_BOT, "TEST_REACTIVACION")
    print(f"Cambio a BOT exitoso: {success}")
    assert success, "No se pudo cambiar estado a BOT"
    
    # Test 6: Limpiar
    if redis_client:
        redis_client.delete(f"chat_state:{test_chat_id}")
    
    print("✅ Todos los tests de estados pasaron")

def test_bot_message_detection():
    """Probar detección de mensajes del bot"""
    print("\n=== TEST 3: DETECCIÓN MENSAJES BOT ===")
    
    test_msg_id = "test_message_123456"
    
    # Test 1: Agregar ID de mensaje del bot
    add_bot_message_id(test_msg_id)
    print(f"ID agregado: {test_msg_id}")
    
    # Test 2: Verificar que es eco del bot
    is_echo = is_bot_message_echo(test_msg_id)
    print(f"Es eco del bot: {is_echo}")
    assert is_echo, "Debería detectar como eco del bot"
    
    # Test 3: Verificar que ya no está (se removió)
    is_echo_again = is_bot_message_echo(test_msg_id)
    print(f"Es eco del bot (segunda vez): {is_echo_again}")
    assert not is_echo_again, "No debería detectar como eco la segunda vez"
    
    print("✅ Tests de detección de mensajes pasaron")

def test_human_intervention_detection():
    """Probar detección de intervención humana"""
    print("\n=== TEST 4: DETECCIÓN INTERVENCIÓN HUMANA ===")
    
    test_chat_id = "test_573002222222@s.whatsapp.net"
    
    # Test 1: Cliente escribe "agente"
    message_data_1 = {
        'fromMe': False,
        'text': 'Quiero hablar con un agente',
        'id': 'msg_001'
    }
    should_change_1, reason_1 = should_change_to_human_state(test_chat_id, message_data_1)
    print(f"Cliente escribe 'agente' - Cambiar: {should_change_1}, Razón: {reason_1}")
    assert should_change_1, "Debería detectar solicitud de agente"
    assert "CLIENTE_SOLICITA_AGENTE" in reason_1, "Razón incorrecta"
    
    # Test 2: Agente interviene (fromMe sin eco)
    message_data_2 = {
        'fromMe': True,
        'text': 'Hola, soy María de Xtalento',
        'id': 'msg_002'
    }
    should_change_2, reason_2 = should_change_to_human_state(test_chat_id, message_data_2)
    print(f"Agente interviene - Cambiar: {should_change_2}, Razón: {reason_2}")
    assert should_change_2, "Debería detectar intervención de agente"
    assert "AGENTE_INTERVIENE" in reason_2, "Razón incorrecta"
    
    # Test 3: Eco del bot (no debe cambiar)
    add_bot_message_id('msg_003')
    message_data_3 = {
        'fromMe': True,
        'text': 'Respuesta del bot',
        'id': 'msg_003'
    }
    should_change_3, reason_3 = should_change_to_human_state(test_chat_id, message_data_3)
    print(f"Eco del bot - Cambiar: {should_change_3}, Razón: {reason_3}")
    assert not should_change_3, "No debería cambiar por eco del bot"
    
    print("✅ Tests de detección de intervención pasaron")

def main():
    """Ejecutar todos los tests"""
    print("🧪 INICIANDO TESTS DE FUNCIONES REDIS\n")
    
    try:
        # Test conexión
        if not test_redis_connection():
            print("❌ Tests fallaron en conexión Redis")
            return False
        
        # Test estados
        test_conversation_states()
        
        # Test detección bot
        test_bot_message_detection()
        
        # Test intervención humana
        test_human_intervention_detection()
        
        print("\n🎉 ¡TODOS LOS TESTS PASARON EXITOSAMENTE!")
        print("✅ Sistema Redis listo para integración")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR EN TESTS: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

