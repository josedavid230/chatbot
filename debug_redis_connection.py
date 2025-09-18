#!/usr/bin/env python3
"""
Script de diagnóstico para Redis Cloud
"""

import redis
import ssl
import os
from dotenv import load_dotenv

# Cargar variables
load_dotenv(override=True)

REDIS_HOST = os.getenv('REDIS_HOST')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')
REDIS_USERNAME = os.getenv('REDIS_USERNAME', 'default')

print("🔧 DIAGNÓSTICO REDIS CLOUD")
print("="*50)
print(f"Host: {REDIS_HOST}")
print(f"Port: {REDIS_PORT}")
print(f"Username: {REDIS_USERNAME}")
print(f"Password: {'***' if REDIS_PASSWORD else 'NO_CONFIG'}")
print("="*50)

# Test 1: SSL estricto (Redis Cloud estándar)
print("\n🧪 TEST 1: SSL ESTRICTO")
try:
    r1 = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        username=REDIS_USERNAME,
        decode_responses=True,
        ssl=True,
        ssl_cert_reqs=ssl.CERT_REQUIRED,
        ssl_check_hostname=True
    )
    result1 = r1.ping()
    print(f"✅ SSL ESTRICTO: {result1}")
except Exception as e:
    print(f"❌ SSL ESTRICTO: {e}")

# Test 2: SSL sin verificación de certificado
print("\n🧪 TEST 2: SSL SIN VERIFICACIÓN DE CERTIFICADO")
try:
    r2 = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        username=REDIS_USERNAME,
        decode_responses=True,
        ssl=True,
        ssl_cert_reqs=ssl.CERT_NONE,
        ssl_check_hostname=False
    )
    result2 = r2.ping()
    print(f"✅ SSL SIN VERIFICACIÓN: {result2}")
except Exception as e:
    print(f"❌ SSL SIN VERIFICACIÓN: {e}")

# Test 3: SSL con cert_reqs=None (Redis Cloud común)
print("\n🧪 TEST 3: SSL CON CERT_REQS=NONE")
try:
    r3 = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        username=REDIS_USERNAME,
        decode_responses=True,
        ssl=True,
        ssl_cert_reqs=None,
        ssl_check_hostname=False
    )
    result3 = r3.ping()
    print(f"✅ SSL CERT_REQS=NONE: {result3}")
except Exception as e:
    print(f"❌ SSL CERT_REQS=NONE: {e}")

# Test 4: URL rediss://
print("\n🧪 TEST 4: URL REDISS://")
try:
    redis_url = f"rediss://{REDIS_USERNAME}:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"
    r4 = redis.from_url(redis_url, decode_responses=True)
    result4 = r4.ping()
    print(f"✅ URL REDISS: {result4}")
except Exception as e:
    print(f"❌ URL REDISS: {e}")

# Test 5: SSL con configuración específica (como webhook.py actual)
print("\n🧪 TEST 5: SSL CONFIGURACIÓN WEBHOOK")
try:
    r5 = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        username=REDIS_USERNAME,
        decode_responses=True,
        ssl=True,
        ssl_cert_reqs='none',
        ssl_check_hostname=False,
        ssl_ca_certs=None,
        socket_connect_timeout=15,
        socket_timeout=15,
        retry_on_timeout=True
    )
    result5 = r5.ping()
    print(f"✅ SSL CONFIGURACIÓN WEBHOOK: {result5}")
except Exception as e:
    print(f"❌ SSL CONFIGURACIÓN WEBHOOK: {e}")

# Test 6: URL rediss:// con SSL automático
print("\n🧪 TEST 6: URL REDISS CON SSL_CERT_REQS=NONE")
try:
    redis_url = f"rediss://{REDIS_USERNAME}:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"
    r6 = redis.from_url(
        redis_url, 
        decode_responses=True,
        ssl_cert_reqs=ssl.CERT_NONE,
        ssl_check_hostname=False
    )
    result6 = r6.ping()
    print(f"✅ URL REDISS CON SSL_CERT_REQS=NONE: {result6}")
except Exception as e:
    print(f"❌ URL REDISS CON SSL_CERT_REQS=NONE: {e}")

# Test 7: Conexión sin SSL (para comparar)
print("\n🧪 TEST 7: SIN SSL (PARA COMPARAR)")
try:
    r7 = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        username=REDIS_USERNAME,
        decode_responses=True,
        ssl=False
    )
    result7 = r7.ping()
    print(f"✅ SIN SSL: {result7}")
except Exception as e:
    print(f"❌ SIN SSL: {e}")

print("\n" + "="*50)
print("🎯 PRUEBA TODAS LAS CONFIGURACIONES ARRIBA")
print("✅ La que muestre '✅' es la configuración correcta")
print("="*50)
