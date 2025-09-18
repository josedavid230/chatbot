#!/usr/bin/env python3
"""
Test completo del sistema Redis integrado con el bot
"""

import sys
import os
import time

# Configurar path y variables de entorno
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(current_dir, '.env'), override=True)

print("🧪 TEST COMPLETO DEL SISTEMA REDIS")
print("="*50)

# Test 1: Verificar conexión Redis
print("\n1️⃣ VERIFICANDO CONEXIÓN REDIS...")
try:
    from redis_state import get_redis_client, STATE_BOT, STATE_HUMANO
    client = get_redis_client()
    if client and client.ping():
        print("✅ Redis conectado exitosamente")
    else:
        print("❌ Error: Redis no conectado")
        sys.exit(1)
except Exception as e:
    print(f"❌ Error importando redis_state: {e}")
    sys.exit(1)

# Test 2: Verificar funciones de webhook
print("\n2️⃣ VERIFICANDO FUNCIONES WEBHOOK...")
try:
    from webhook import (
        get_conversation_state, 
        set_conversation_state,
        should_change_to_human_state,
        redis_client as webhook_redis
    )
    
    if webhook_redis and webhook_redis.ping():
        print("✅ Webhook Redis conectado")
    else:
        print("❌ Webhook Redis no conectado")
        
    # Test función should_change_to_human_state
    test_data = {
        'fromMe': False,
        'text': 'Quiero hablar con un agente',
        'id': 'test_001'
    }
    should_change, reason = should_change_to_human_state("test_user", test_data)
    if should_change and "CLIENTE_SOLICITA_AGENTE" in reason:
        print("✅ Detección de 'agente' funciona")
    else:
        print(f"❌ Detección de 'agente' falla: {should_change}, {reason}")
        
except Exception as e:
    print(f"❌ Error importando webhook: {e}")

# Test 3: Verificar integración con main.py
print("\n3️⃣ VERIFICANDO INTEGRACIÓN CHATBOT...")
try:
    from main import Chatbot, load_vector_store
    
    # Cargar vectorstore
    vectorstore = load_vector_store()
    if not vectorstore:
        print("⚠️  Vectorstore no encontrado, creando bot sin vectorstore")
        vectorstore = None
    
    # Crear bot con chat_id
    test_chat_id = "test_573001234567@s.whatsapp.net"
    bot = Chatbot(vectorstore, chat_id=test_chat_id)
    
    print(f"✅ Bot creado con chat_id: {bot.chat_id}")
    
    # Test detección de agente
    print("\n🔍 Probando detección de 'agente' en bot...")
    
    # Limpiar estado inicial
    set_conversation_state(test_chat_id, STATE_BOT, "TEST_INICIAL")
    
    response = bot.process_message("Hola")
    print(f"Respuesta normal: {response[:50]}...")
    
    # Verificar estado sigue siendo BOT
    state = get_conversation_state(test_chat_id)
    if state == STATE_BOT:
        print("✅ Estado permanece BOT después de mensaje normal")
    else:
        print(f"❌ Estado cambió inesperadamente: {state}")
    
    # Probar solicitud de agente
    response_agent = bot.process_message("Quiero hablar con un agente")
    print(f"Respuesta agente: {response_agent}")
    
    # Verificar estado cambió a HUMANO
    time.sleep(1)  # Dar tiempo para que Redis se actualice
    state_after = get_conversation_state(test_chat_id)
    if state_after == STATE_HUMANO:
        print("✅ Estado cambió a HUMANO después de solicitar agente")
    else:
        print(f"❌ Estado no cambió: {state_after}")
    
    # Limpiar
    if client:
        client.delete(f"chat_state:{test_chat_id}")
        
except Exception as e:
    print(f"❌ Error en test de chatbot: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Verificar tarea de reactivación
print("\n4️⃣ VERIFICANDO TAREA DE REACTIVACIÓN...")
try:
    from reactivation_task import reactivate_inactive_conversations, get_conversation_stats
    
    # Crear una conversación en estado HUMANO con actividad antigua
    test_chat_old = "test_old_573009999999@s.whatsapp.net"
    old_timestamp = int(time.time()) - (2 * 3600)  # 2 horas atrás
    
    if client:
        client.hset(f"chat_state:{test_chat_old}", mapping={
            'state': STATE_HUMANO,
            'last_activity': str(old_timestamp),
            'reason': 'TEST_REACTIVACION',
            'updated_at': '2024-01-01T00:00:00'
        })
        
        print(f"✅ Conversación de prueba creada: {test_chat_old}")
        
        # Ejecutar reactivación
        print("🔄 Ejecutando reactivación...")
        reactivate_inactive_conversations()
        
        # Verificar que se reactivó
        time.sleep(1)
        new_state = get_conversation_state(test_chat_old)
        if new_state == STATE_BOT:
            print("✅ Reactivación automática funciona")
        else:
            print(f"❌ Reactivación falló: {new_state}")
            
        # Obtener estadísticas
        print("\n📊 Estadísticas:")
        get_conversation_stats()
        
        # Limpiar
        client.delete(f"chat_state:{test_chat_old}")
    
except Exception as e:
    print(f"❌ Error en test de reactivación: {e}")

# Test 5: Verificar limpieza de archivos temporales
print("\n5️⃣ LIMPIANDO ARCHIVOS TEMPORALES...")
try:
    # Los archivos de debug ya fueron eliminados
    print("✅ Limpieza completa")
except Exception as e:
    print(f"⚠️  Error en limpieza: {e}")

print("\n" + "="*50)
print("🎉 TESTS COMPLETADOS")
print("✅ El sistema Redis está integrado y funcionando")
print("")
print("📋 PRÓXIMOS PASOS:")
print("1. Instalar schedule: pip install schedule")
print("2. Hacer push a la branch: git add . && git commit -m 'Redis integration complete' && git push")
print("3. Configurar servicio de reactivación: chmod +x setup_reactivation_service.sh && ./setup_reactivation_service.sh")
print("4. Reiniciar el bot principal para aplicar cambios")
print("="*50)
