from flask import Flask, request
from main import Chatbot, load_vector_store
from dotenv import load_dotenv
import os
import requests

load_dotenv()

app = Flask(__name__)

VERIFY_TOKEN = "xtalento2024"  # Usalo también en Meta Developer
vectorstore = load_vector_store()
bot = Chatbot(vectorstore)

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")  # Usa el token de acceso de la app de WhatsApp Cloud
PHONE_NUMBER_ID = "624826184058222"  # Asegurate de que sea string

def send_whatsapp_message(to, message):
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message}
    }
    response = requests.post(url, headers=headers, json=payload)
    print("Respuesta de WhatsApp:", response.status_code, response.text)

@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    if request.method == "GET":
        token_sent = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")
        if token_sent == VERIFY_TOKEN:
            return str(challenge)
        return "Token de verificación inválido"

    if request.method == "POST":
        data = request.get_json()
        print("[MENSAJE RECIBIDO]", data)

        try:
            message_text = data['entry'][0]['changes'][0]['value']['messages'][0]['text']['body']
            sender = data['entry'][0]['changes'][0]['value']['messages'][0]['from']

            # Procesar mensaje con chatbot
            respuesta = bot.process_message(message_text)

            # Enviar respuesta real al usuario por WhatsApp
            send_whatsapp_message(sender, respuesta)

        except Exception as e:
            print("[ERROR EN PROCESAMIENTO]", e)

        return "EVENT_RECEIVED", 200

if __name__ == "__main__":
    app.run(port=5000, debug=True)
