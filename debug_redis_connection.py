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

# Test 1: Conexión simple sin SSL
print("\n🧪 TEST 1: SIN SSL")
try:
    r1 = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        username=REDIS_USERNAME,
        decode_responses=True,
        ssl=False
    )
    result1 = r1.ping()
    print(f"✅ SIN SSL: {result1}")
except Exception as e:
    print(f"❌ SIN SSL: {e}")

# Test 2: SSL básico
print("\n🧪 TEST 2: SSL BÁSICO")
try:
    r2 = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        username=REDIS_USERNAME,
        decode_responses=True,
        ssl=True
    )
    result2 = r2.ping()
    print(f"✅ SSL BÁSICO: {result2}")
except Exception as e:
    print(f"❌ SSL BÁSICO: {e}")

# Test 3: SSL sin verificación
print("\n🧪 TEST 3: SSL SIN VERIFICACIÓN")
try:
    r3 = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        username=REDIS_USERNAME,
        decode_responses=True,
        ssl=True,
        ssl_cert_reqs=ssl.CERT_NONE,
        ssl_check_hostname=False
    )
    result3 = r3.ping()
    print(f"✅ SSL SIN VERIFICACIÓN: {result3}")
except Exception as e:
    print(f"❌ SSL SIN VERIFICACIÓN: {e}")

# Test 4: URL rediss://
print("\n🧪 TEST 4: URL REDISS://")
try:
    redis_url = f"rediss://{REDIS_USERNAME}:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"
    r4 = redis.from_url(redis_url, decode_responses=True)
    result4 = r4.ping()
    print(f"✅ URL REDISS: {result4}")
except Exception as e:
    print(f"❌ URL REDISS: {e}")

# Test 5: URL redis:// (sin SSL)
print("\n🧪 TEST 5: URL REDIS:// (SIN SSL)")
try:
    redis_url = f"redis://{REDIS_USERNAME}:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"
    r5 = redis.from_url(redis_url, decode_responses=True)
    result5 = r5.ping()
    print(f"✅ URL REDIS: {result5}")
except Exception as e:
    print(f"❌ URL REDIS: {e}")

# Test 6: Conexión manual SSL con timeout largo
print("\n🧪 TEST 6: SSL CON TIMEOUT LARGO")
try:
    r6 = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        username=REDIS_USERNAME,
        decode_responses=True,
        ssl=True,
        ssl_cert_reqs=None,
        ssl_check_hostname=False,
        ssl_ca_certs=None,
        socket_connect_timeout=60,
        socket_timeout=60
    )
    result6 = r6.ping()
    print(f"✅ SSL TIMEOUT LARGO: {result6}")
except Exception as e:
    print(f"❌ SSL TIMEOUT LARGO: {e}")

print("\n" + "="*50)
print("🎯 PRUEBA TODAS LAS CONFIGURACIONES ARRIBA")
print("✅ La que muestre '✅' es la configuración correcta")
print("="*50)
