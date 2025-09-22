



import os
import time
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from redis_state import set_conversation_state_from_bot, get_conversation_state_from_bot, STATE_HUMANO

# Cargar variables de entorno. Asegúrate de tener un archivo .env con tu OPENAI_API_KEY
load_dotenv(override=True)

# --- Constantes ---     
DOCUMENTS_PATH = "documents"
VECTORSTORE_PATH = "vectorstore"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
OPENAI_MODEL = "gpt-4o-mini"
PAYMENT_FORM_URL = "https://forms.gle/vBDAguF19cSaDhAK6"
CALENDAR_LINK = "https://n9.cl/fa5tz3"

# --- Configuración para estado "escribiendo" ---
BOT_TYPING_DELAY = 15  # segundos
EVO_API_URL = os.getenv('EVO_API_URL', 'http://localhost:8080')
EVO_APIKEY = os.getenv('EVO_APIKEY', '')
EVO_INSTANCE = os.getenv('EVO_INSTANCE', '')

# --- Estados de Conversación ---
class ConversationState:
    AWAITING_GREETING = "AWAITING_GREETING"
    AWAITING_NAME_CITY = "AWAITING_NAME_CITY"
    AWAITING_ROLE_INPUT = "AWAITING_ROLE_INPUT"
    AWAITING_SERVICE_CHOICE = "AWAITING_SERVICE_CHOICE"
    AWAITING_CONTINUE_CHOICE = "AWAITING_CONTINUE_CHOICE"
    PROVIDING_INFO = "PROVIDING_INFO"

# --- Funciones Auxiliares para Estado "Escribiendo" ---
def _auth_headers():
    """Devuelve headers de autenticación para Evolution API."""
    return {
        'Content-Type': 'application/json',
        'apikey': EVO_APIKEY
    }

def send_typing_indicator(number: str) -> bool:
    """Envía indicador de 'escribiendo...' a WhatsApp."""
    try:
        url = f"{EVO_API_URL}/chat/presence/{EVO_INSTANCE}"
        payload = {
            "number": number,
            "presence": "composing"  # 'composing' = escribiendo
        }
        
        print(f"[TYPING] Enviando indicador 'escribiendo' a {number}")
        response = requests.post(url, headers=_auth_headers(), json=payload, timeout=10)
        
        if response.status_code in (200, 201):
            print(f"[TYPING] ✅ Indicador enviado exitosamente")
            return True
        else:
            print(f"[TYPING] ❌ Error al enviar indicador: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"[TYPING ERROR] No se pudo enviar indicador de escritura: {e}")
        return False

def simulate_typing_delay(number: str, delay_seconds: int = BOT_TYPING_DELAY):
    """Simula que el bot está escribiendo por un tiempo determinado."""
    try:
        print(f"[TYPING] Iniciando simulación de escritura por {delay_seconds} segundos...")
        
        # Enviar indicador de 'escribiendo'
        send_typing_indicator(number)
        
        # Esperar el tiempo especificado
        time.sleep(delay_seconds)
        
        print(f"[TYPING] ✅ Simulación de escritura completada")
        
    except Exception as e:
        print(f"[TYPING ERROR] Error durante simulación de escritura: {e}")

# --- Lógica del Chatbot ---
class Chatbot:
    def __init__(self, vectorstore, chat_id=None):
        self.state = ConversationState.AWAITING_GREETING
        self.user_data = {}
        self.user_data['name'] = "" # Se inicializa el nombre del usuario
        self.chat_history = []
        self.chat_id = chat_id  # Identificador del chat para Redis
        # Sistema de memoria para confirmaciones de pasos
        self.confirmed_steps = {
            'paso1': False,  # Formulario completado
            'paso3': False   # Pago realizado
        }
        self.llm = ChatOpenAI(model_name=OPENAI_MODEL, max_tokens=500, temperature=0.1)
        
        system_prompt = """
        Actuás como Xtalento Bot, un asistente profesional cálido, claro y experto que guía a personas a potenciar su perfil laboral y encontrar empleo más rápido.
        El nombre del usuario es {user_name}. Cuando sea natural y amigable, utiliza su nombre para personalizar la conversación. Si no sabes el nombre (porque está vacío), no intentes inventarlo.
        Importante: No utilices la palabra 'Hola' en ninguna de tus respuestas, ya que el saludo inicial ya fue dado. siempre trata de no sobrepasar los 200 tokens.

        🔎 Cuando el usuario mencione un servicio, responde incluyendo:
        - Qué incluye el servicio.
        - Cuál es el precio para su nivel de cargo.
        - Cómo se agenda o paga.

        📌 Cuando el usuario mencione su cargo o el último empleo:
        - Clasificalo automáticamente según la guía de cargos (operativo, táctico o estratégico).
        - Usa esa clasificación para sugerir servicios y precios.

        🎯 Siempre destacá:
        - La mentoría virtual personalizada (45 a 60 minutos).
        - La entrega rápida (40 a 120 minutos en optimización HV).
        - La garantía de superar filtros ATS.
        - El respaldo de casos reales y experiencia.

        🚨 POLÍTICA DE CONOCIMIENTO ESTRICTA:
        - SOLO habla de lo que tienes conocimiento confirmado en el contexto proporcionado (RAG).
        - NO inventes información, servicios, precios o datos que no estén en tu base de conocimiento.
        - Si no tienes conocimiento suficiente sobre algo que te preguntan, responde honestamente: "Actualmente no tengo conocimiento sobre esto. Si quieres comunicarte con un humano, menciona la palabra 'agente' en el chat."
        - Si el usuario menciona "agente" en cualquier momento, conecta inmediatamente con un agente humano.
        - Si el usuario decide seguir con el bot después de no tener información, tu objetivo principal es vender un servicio disponible en tu conocimiento y proponer agendar una sesión virtual.
        - Usa emojis con calidez, sin perder profesionalismo. Sé concreto y con orientación clara a la acción.
        """
        
        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = """Dada una conversación y una pregunta de seguimiento, reformula la pregunta de seguimiento para que sea una pregunta independiente, en su idioma original. El nombre del usuario es {user_name}. IMPORTANTE: Solo utiliza información que esté confirmada en el contexto de la conversación. Si no tienes conocimiento suficiente, indica que no tienes esa información y que el cliente se puede comunicar con un agente humano. copiando la palabra agente en el chat"""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        qa_system_prompt = system_prompt + """

        Contexto: {context}
        Respuesta:"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    def _apply_typing_delay(self, user_number: str = None):
        """Aplica retraso con indicador de escritura si hay un número de usuario disponible."""
        if user_number and EVO_API_URL and EVO_APIKEY and EVO_INSTANCE:
            simulate_typing_delay(user_number, BOT_TYPING_DELAY)
        else:
            # Si no hay configuración de API, simplemente esperar sin indicador
            print(f"[TYPING] API no configurada, aplicando solo retraso de {BOT_TYPING_DELAY}s")
            time.sleep(BOT_TYPING_DELAY)

    def _extract_role_from_text(self, text):
        """Extrae el cargo o rol laboral de un texto complejo."""
        extraction_prompt = f"""
        Analiza el siguiente texto y extrae ÚNICAMENTE el cargo o rol laboral mencionado. 

        Ejemplos:
        - "Estoy desempleado antes era analista de datos" → "analista de datos"
        - "Soy gerente de ventas en una empresa" → "gerente de ventas"
        - "Trabajo como desarrollador frontend" → "desarrollador frontend"
        - "Fui coordinador de proyectos" → "coordinador de proyectos"
        - "Me desempeño como CEO" → "CEO"
        - "Quiero trabajar de marketing" → "marketing"

        IMPORTANTE: 
        - Extrae solo el cargo/rol, sin contexto adicional
        - Si hay múltiples cargos, extrae el más relevante o reciente
        - Si no hay un cargo claro, responde 'no_identificable'
        - No agregues explicaciones

        Texto: "{text}"
        Cargo extraído:
        """
        response = self.llm.invoke(extraction_prompt)
        extracted_role = response.content.strip()
        
        if extracted_role.lower() in ['no_identificable', 'no identificable', '']:
            return None
        return extracted_role

    def _classify_role(self, role_description):
        """Usa el LLM para extraer y clasificar un rol de texto complejo en operativo, táctico o estratégico."""
        # Paso 1: Extraer el cargo del texto complejo
        extracted_role = self._extract_role_from_text(role_description)
        
        if not extracted_role:
            print(f"[DEBUG] No se pudo extraer un cargo del texto: {role_description}")
            return None
            
        print(f"[DEBUG] Cargo extraído: '{extracted_role}' del texto: '{role_description}'")
        
        # Paso 2: Clasificar el cargo extraído
        classification_prompt_text = f"""
        Clasifica el siguiente cargo laboral en UNO de estos tres niveles jerárquicos.
        Responde ÚNICAMENTE con la palabra: operativo, táctico o estratégico.

        NIVELES JERÁRQUICOS:

        OPERATIVO: Cargos de ejecución directa y técnicos
        - Analistas, desarrolladores, asistentes, operarios, técnicos
        - Especialistas junior, consultores junior
        - Ejecutivos de cuenta, vendedores

        TÁCTICO: Cargos de supervisión y coordinación media
        - Coordinadores, especialistas senior, jefes de área
        - Supervisores, team leads, líderes de equipo
        - Gerentes de área específica

        ESTRATÉGICO: Cargos de alta dirección y toma de decisiones
        - CEO, presidente, vicepresidente, director general
        - Directores de área, gerentes generales
        - VP (vicepresidente), fundadores

        Cargo a clasificar: "{extracted_role}"

        Respuesta (solo la palabra):"""
        response = self.llm.invoke(classification_prompt_text)
        # Aseguramos que la respuesta sea una de las tres opciones válidas
        classification_raw = response.content.strip().lower()
        
        print(f"[DEBUG] Clasificación raw obtenida: '{classification_raw}' para cargo: '{extracted_role}'")
        
        # Extraer la clasificación real del texto (por si el LLM agrega palabras extra)
        if 'operativo' in classification_raw:
            classification = 'operativo'
        elif 'táctico' in classification_raw or 'tactico' in classification_raw:
            classification = 'táctico'
        elif 'estratégico' in classification_raw or 'estrategico' in classification_raw:
            classification = 'estratégico'
        else:
            print(f"[DEBUG] No se pudo extraer clasificación válida de: '{classification_raw}'")
            return None
        
        print(f"[DEBUG] Clasificación final: '{classification}' para cargo: '{extracted_role}'")
        return classification

    def _extract_name(self, name_city_text):
        """Usa el LLM para extraer el nombre de pila del usuario de un texto."""
        extraction_prompt_text = f"""
        De la siguiente frase, extrae únicamente el nombre de pila del usuario.
        Ejemplo: si la frase es "Soy Carlos de Lima", la respuesta debe ser "Carlos".
        IMPORTANTE: Solo extrae el nombre si está claramente presente. Si no puedes identificar un nombre con certeza, responde 'no_identificable'.
        Devuelve solo el nombre, sin explicaciones ni texto adicional.
        Frase: "{name_city_text}"
        Nombre de pila:
        """
        response = self.llm.invoke(extraction_prompt_text)
        name = response.content.strip()
        
        # Si el LLM no devuelve nada, usamos la primera palabra como fallback.
        if not name:
            return name_city_text.split()[0]
        return name

    def _generate_response(self, prompt_text):
        """Genera una respuesta directa del LLM para mensajes conversacionales."""
        # Usamos el mismo LLM pero sin el contexto de RAG
        response = self.llm.invoke(prompt_text)
        return response.content.strip()

    def _safe_rag_answer(self, query_text: str) -> str:
        """Intenta responder vía RAG; si falla, continúa la conversación con conocimiento general (LLM)."""
        try:
            response = self.rag_chain.invoke({"input": query_text, "chat_history": self.chat_history, "user_name": self.user_data.get('name', '')})
            answer_text = response.get('answer') or ""
            if not answer_text.strip():
                return self._build_unknown_options_message()
            return answer_text
        except Exception:
            return self._build_unknown_options_message()

    def _build_unknown_options_message(self) -> str:
        """Devuelve el mensaje estándar de opciones cuando no hay suficiente información."""
        return (
            "Actualmente no tengo conocimiento sobre esto. Si quieres comunicarte con un humano, menciona la palabra 'agente' en el chat. "
            "¿Qué prefieres que hagamos?\n\n"
            "1) Hablar con un agente humano (menciona 'agente' para conectarte inmediatamente).\n"
            "2) Seguir conmigo y explorar otros servicios de Xtalento que sí conozco."
        )

    def _continue_conversation(self, user_text: str, guidance: str) -> str:
        """Responde cuando no se detecta una respuesta válida, ofreciendo opciones claras al usuario."""
        return (
            "No se detectó una respuesta válida. "
            "¿Qué prefieres?\n\n"
            "1) Te remito con un agente de ventas humano\n"
            "2) Seguir hablando conmigo\n\n"
            "Si quieres hablar con un agente, escribe 'agente'."
        )
    
    def _is_agent_request(self, user_input: str) -> bool:
        """Detecta si el usuario está solicitando específicamente un agente humano."""
        text = user_input.lower().strip()
        words = text.split()
        
        # Caso 1: Palabra exacta "agente"
        if len(words) == 1 and words[0] == 'agente':
            return True
            
        # Caso 2: Frases específicas de solicitud de agente
        agent_request_phrases = [
            'Quiero un agente', 'Necesito un agente', 'Conectame con un agente',
            'Hablar con un agente', 'Contactar un agente', 'Agente humano',
            'Quiero agente', 'Necesito agente', 'Conectame con agente',
            'Quiero hablar con un agente', 'Necesito hablar con un agente'
        ]
        
        return any(phrase in text for phrase in agent_request_phrases)

    def _is_returning_after_agent(self, user_input: str) -> bool:
        """Detecta si el cliente está retomando la conversación después de haber hablado con un agente humano."""
        try:
            prompt = f"""
            Analiza si el siguiente mensaje parece ser de un cliente que está retomando o continuando una conversación después de haber hablado con un agente humano.

            MENSAJE DEL USUARIO: "{user_input}"

            ¿Este mensaje indica que el cliente está retomando la conversación o necesita más ayuda?

            Responde ÚNICAMENTE:
            - "SÍ" si parece que retoma conversación, saluda de nuevo, o busca más ayuda
            - "NO" si es una respuesta específica a opciones del bot

            Indicadores de retomar conversación:
            - Saludos nuevos: "Hola", "Buenos días", "Qué tal"
            - Frases de continuación: "Necesito más información", "Tengo otra pregunta"
            - Agradecimientos y nuevas consultas: "Gracias, pero también quería..."
            - Mensajes naturales que no son respuestas a opciones específicas

            Ejemplos:
            - "Hola, tengo una pregunta adicional" → SÍ
            - "Buenos días, necesito más información" → SÍ  
            - "Gracias por la atención, pero quería saber..." → SÍ
            - "¿Tienen otros servicios?" → SÍ
            - "1" → NO (respuesta a opción)
            - "agente" → NO (solicitud específica)
            """
            
            response = self.llm.invoke(prompt).content.strip().upper()
            
            if "SÍ" in response or "SI" in response:
                print(f"[DEBUG] LLM detectó cliente retomando conversación: {response}")
                return True
            else:
                print(f"[DEBUG] LLM NO detectó retomar conversación: {response}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Error en _is_returning_after_agent: {e}")
            # Fallback: detectar saludos básicos y frases de retomar
            return any(phrase in user_input.lower() for phrase in [
                'hola', 'buenos días', 'buenas tardes', 'qué tal', 'necesito más', 'tengo otra', 'quería saber'
            ])

    def _generate_welcome_back_message(self, user_input: str) -> str:
        """Genera un mensaje de bienvenida cálido para clientes que retoman la conversación."""
        user_name = self.user_data.get('name', '')
        name_part = f"{user_name}, " if user_name and user_name != 'no_identificable' else ""
        
        try:
            prompt = f"""
            Genera un mensaje de bienvenida cálido y profesional para un cliente que está retomando la conversación con Xtalento Bot después de haber hablado con un agente humano.

            NOMBRE DEL CLIENTE: {name_part}
            MENSAJE DEL CLIENTE: "{user_input}"

            El mensaje debe:
            1. Dar la bienvenida de vuelta de forma cálida
            2. Agradecer por su regreso
            3. Preguntar en qué podemos ayudarle
            4. Ser profesional pero amigable
            5. Usar el nombre si está disponible
            6. Máximo 150 palabras

            Ejemplos de tono:
            - "¡{name_part}es un gusto tenerte de vuelta! Agradezco que regreses a conversar conmigo..."
            - "¡Qué bueno verte de nuevo{name_part}! Me da mucho gusto que hayas decidido continuar..."

            Genera el mensaje de bienvenida:
            """
            
            response = self.llm.invoke(prompt).content.strip()
            return response
            
        except Exception as e:
            print(f"[ERROR] Error en _generate_welcome_back_message: {e}")
            # Fallback mensaje estático
            return f"¡{name_part}es un gusto tenerte de vuelta! 😊 Agradezco que regreses a conversar conmigo. Me da mucho gusto que hayas decidido continuar con nuestros servicios. ¿En qué más puedo ayudarte hoy?"

    def _is_natural_conversation_resumption(self, user_input: str) -> bool:
        """Detecta si el usuario está retomando una conversación de forma natural."""
        try:
            prompt = f"""
            Analiza si el siguiente mensaje parece ser una conversación natural donde alguien está:
            - Saludando normalmente
            - Haciendo una pregunta sobre servicios laborales
            - Retomando una conversación previa
            - Escribiendo algo natural (no solo números o palabras clave)

            MENSAJE: "{user_input}"

            ¿Este mensaje parece una conversación natural que debería ser respondida normalmente?

            Responde ÚNICAMENTE:
            - "SÍ" si es conversación natural
            - "NO" si parece ser respuesta a opciones específicas (1, 2, agente, etc.)

            Ejemplos:
            - "Hola estoy haciendo pruebas" → SÍ
            - "¿Qué servicios tienen?" → SÍ
            - "Me interesa mejorar mi CV" → SÍ
            - "1" → NO
            - "agente" → NO
            - "seguir hablando" → NO
            """
            
            response = self.llm.invoke(prompt).content.strip().upper()
            
            if "SÍ" in response or "SI" in response:
                print(f"[DEBUG] LLM detectó conversación natural: {response}")
                return True
            else:
                print(f"[DEBUG] LLM detectó respuesta específica: {response}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Error en _is_natural_conversation_resumption: {e}")
            # Fallback: si tiene más de 10 caracteres y no son números, probablemente es natural
            return len(user_input.strip()) > 10 and not user_input.strip().isdigit()

    def _detect_service_intention(self, user_input: str) -> bool:
        """Usa LLM para detectar si el usuario quiere seleccionar un servicio específico."""
        try:
            prompt = f"""
            Analiza si el siguiente mensaje del usuario indica que quiere seleccionar UN SERVICIO ESPECÍFICO de esta lista:

            SERVICIOS DISPONIBLES:
            1. Optimización de Hoja de vida (ATS)
            2. Mejora de perfil en plataformas de empleo  
            3. Preparación para Entrevistas
            4. Estrategia de búsqueda de empleo
            5. Simulación de entrevista con feedback
            6. Método X (servicio premium)
            7. Test EPI - Evaluación de Personalidad Integral
            8. Todos los servicios

            MENSAJE DEL USUARIO: "{user_input}"

            ¿El usuario está intentando seleccionar un servicio específico de la lista?

            Responde ÚNICAMENTE:
            - "SÍ" si claramente quiere un servicio específico (números, nombres, o intención clara)
            - "NO" si es una pregunta general, saludo, o no menciona servicios específicos

            Ejemplos:
            - "Quiero el 3" → SÍ
            - "Hoja de vida" → SÍ  
            - "¿Qué incluye el método X?" → NO (es pregunta, no selección)
            - "Hola, estoy interesado" → NO (muy general)
            - "Todos" → SÍ
            """
            
            response = self.llm.invoke(prompt).content.strip().upper()
            
            # Si dice SÍ, el usuario quiere seleccionar un servicio
            if "SÍ" in response or "SI" in response:
                print(f"[DEBUG] LLM detectó intención de servicio: {response}")
                return True
            else:
                print(f"[DEBUG] LLM NO detectó selección de servicio: {response}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Error en _detect_service_intention: {e}")
            # Fallback a detección básica por keywords
            service_keywords = ['1', '2', '3', '4', '5', '6', '7', '8', 'todos', 'todo', 'hoja de vida', 'cv', 'método x', 'metodo x']
            return any(keyword in user_input.lower() for keyword in service_keywords)

    def _detect_scheduling_request(self, user_input: str) -> bool:
        """Usa LLM para detectar si el usuario quiere agendar una cita o sesión."""
        try:
            prompt = f"""
            Analiza si el siguiente mensaje del usuario indica que quiere AGENDAR, PROGRAMAR o SOLICITAR una cita, sesión, reunión o consulta.

            MENSAJE DEL USUARIO: "{user_input}"

            ¿El usuario está intentando agendar o programar algo?

            Responde ÚNICAMENTE:
            - "SÍ" si claramente quiere agendar, programar o solicitar una cita/sesión
            - "NO" si solo está preguntando sobre horarios, precios, o haciendo consultas generales

            Ejemplos:
            - "agenda" → SÍ
            - "Quiero agendar una cita" → SÍ
            - "¿Cuándo puedo programar?" → SÍ
            - "Disponibilidad para sesión" → SÍ
            - "¿A qué hora trabajan?" → NO (solo pregunta horarios)
            - "¿Cuánto cuesta la consulta?" → NO (pregunta precio)
            - "¿Qué incluye la sesión?" → NO (pregunta información)
            """
            
            response = self.llm.invoke(prompt).content.strip().upper()
            
            if "SÍ" in response or "SI" in response:
                print(f"[DEBUG] LLM detectó solicitud de agendamiento: {response}")
                return True
            else:
                print(f"[DEBUG] LLM NO detectó solicitud de agendamiento: {response}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Error en _detect_scheduling_request: {e}")
            # Fallback a detección básica por keywords críticas
            scheduling_keywords = ['agendar', 'agenda', 'agendo', 'programar', 'cita', 'sesión', 'reunión']
            return any(keyword in user_input.lower() for keyword in scheduling_keywords)
    
    def _provide_calendar_link(self) -> str:
        """Proporciona el enlace del calendario para agendar citas."""
        return (
            f"¡Perfecto! Para agendar tu sesión personalizada, entra a nuestro calendario online:\n\n"
            f"🗓️ {CALENDAR_LINK}\n\n"
            f"Ahí puedes elegir el día y horario que mejor te convenga. "
            f"¡Te esperamos!"
        )

    def _handle_continue_choice(self, user_input: str) -> str:
        """Maneja la respuesta del usuario después de _continue_conversation."""
        text_lower = user_input.lower().strip()
        
        # Opción 1: Quiere agente humano
        if any(x in text_lower for x in ["1", "uno", "humano", "ventas"]) or self._is_agent_request(user_input):
            return "Perfecto. Te conecto con un agente humano inmediatamente. Pauso este chat y un agente de ventas te contactará en este mismo canal."
        
        # Opción 2: Quiere seguir hablando
        elif any(x in text_lower for x in ["2", "dos", "seguir", "continuar", "hablar", "preguntas"]):
            return "Perfecto, sigamos hablando. ¿Qué preguntas tienes sobre nuestros servicios de potenciación laboral?"
        
        # Si no detecta ninguna opción clara
        else:
            return (
                "No entendí tu respuesta. Por favor elige:\n\n"
                "Escribe '1' para hablar con un agente humano\n"
                "Escribe '2' si quieres seguir hablando conmigo\n"
                "O escribe 'agente' para conectarte directamente."
            )

    def process_message(self, user_input, user_number: str = None):
        try:
            # PRIORIDAD MÁXIMA: Detectar solicitud de agente humano ANTES de cualquier procesamiento
            # Solo activar con palabra exacta "agente" o frases específicas de solicitud
            if (user_input and self._is_agent_request(user_input) and 
                self.state != ConversationState.AWAITING_ROLE_INPUT):
                # Agregar el mensaje del usuario al historial antes de responder
                self.chat_history.append(HumanMessage(content=user_input))
                response = "Te conecto con un agente humano inmediatamente. Un agente de ventas te contactará en este mismo canal."
                self.chat_history.append(AIMessage(content=response))
                
                # Cambiar estado a HUMANO en Redis si tenemos chat_id
                if self.chat_id:
                    success = set_conversation_state_from_bot(self.chat_id, STATE_HUMANO, "CLIENTE_SOLICITA_AGENTE")
                    if success:
                        print(f"[BOT->REDIS] Estado cambiado a HUMANO para {self.chat_id}")
                    else:
                        print(f"[BOT->REDIS ERROR] No se pudo cambiar estado para {self.chat_id}")
                
                return response
            
            # SEGUNDA PRIORIDAD: Detectar solicitudes de agendamiento
            if user_input and self._detect_scheduling_request(user_input):
                # Agregar el mensaje del usuario al historial antes de responder
                self.chat_history.append(HumanMessage(content=user_input))
                response = self._provide_calendar_link()
                self.chat_history.append(AIMessage(content=response))
                return response
            
            # Los saludos iniciales no necesitan memoria ni RAG
            if self.state == ConversationState.AWAITING_GREETING:
                # Aplicar retraso con indicador de escritura
                self._apply_typing_delay(user_number)
                
                self.state = ConversationState.AWAITING_NAME_CITY
                prompt = "Actúas como Xtalento Bot. Genera un saludo inicial cálido y profesional que comience exactamente con la palabra '¡Hola! 👋'. A continuación, preséntate brevemente y pide al usuario su nombre y la ciudad desde la que escribe. IMPORTANTE: Solo habla de servicios y información que tienes conocimiento confirmado en tu base de datos."
                response_text = self._generate_response(prompt)
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            # Guardamos la entrada del usuario en el historial
            self.chat_history.append(HumanMessage(content=user_input))

            if self.state == ConversationState.AWAITING_NAME_CITY:
                is_question = '?' in user_input or (user_input.lower().split() and user_input.lower().split()[0] in [
                    'qué', 'cómo', 'cuándo', 'dónde', 'cuál', 'por', 'quién', 'what', 
                    'how', 'when', 'where', 'which', 'why', 'who', 'do', 'is', 'are'
                ])
                if is_question:
                    print(f"[DEBUG] Se detectó una pregunta en lugar de nombre/ciudad. Respondiendo sin interrumpir la conversación.")
                    # Aplicar retraso con indicador de escritura
                    self._apply_typing_delay(user_number)
                    answer = self._safe_rag_answer(user_input)
                    self.chat_history.append(AIMessage(content=answer))
                    return answer
                
                self.user_data['name_city'] = user_input
                user_name = self._extract_name(user_input)
                self.user_data['name'] = user_name
                self.state = ConversationState.AWAITING_ROLE_INPUT
                
                # Aplicar retraso con indicador de escritura
                self._apply_typing_delay(user_number)
                
                prompt = f"Actúas como Xtalento Bot. El usuario se llama {user_name}. Dale una bienvenida personalizada (sin usar la palabra 'Hola') y luego pregúntale sobre su cargo actual o al que aspira para poder darle una mejor asesoría. IMPORTANTE: Solo habla de servicios que tienes conocimiento confirmado. Si no sabes algo específico, di 'Actualmente no tengo conocimiento sobre esto. Si quieres comunicarte con un humano, menciona la palabra agente en el chat.'"
                response_text = self._generate_response(prompt)
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            elif self.state == ConversationState.AWAITING_ROLE_INPUT:
                role_classification = self._classify_role(user_input)
                if not role_classification:
                    print(f"[DEBUG] No se pudo clasificar el rol. Continuando sin clasificación específica.")
                    # En lugar de ir a _continue_conversation, continuar sin rol específico
                    role_classification = 'general'  # Rol por defecto

                self.user_data['role'] = role_classification
                self.state = ConversationState.AWAITING_SERVICE_CHOICE
                
                # Aplicar retraso con indicador de escritura
                self._apply_typing_delay(user_number)
                
                prompt = f"""
                Actúas como Xtalento Bot. Presenta los siguientes servicios en una lista numerada sin mencionar ni revelar la categoría/nivel del usuario:
                Gracias por tu interés en Xtalento. Te contamos que ayudamos a personas como tú a potenciar su perfil profesional y conseguir trabajo más rápido.

                Nuestros servicios son:

                1. *Optimización de hoja de vida)(formato ATS)*: Adaptamos tu HV para que supere filtros digitales y capte la atención de los reclutadores.
                2. *Mejora de perfil en plataformas de empleo*: Potenciamos tu perfil para que se vea profesional, tenga mayor visibilidad y atraiga más oportunidades.
                3. *Preparación de entrevistas laborales*: Te entrenamos con preguntas reales, retroalimentación y técnicas para responder con seguridad y generar impacto.
                4. *Estrategia personalizada de búsqueda de empleo)*: Creamos un plan contigo para que busques trabajo de forma más efectiva, enfocada y con objetivos claros.
                5. *Simulación de entrevistas laborales con feedback*: Simulación de entrevistas reales, recomendaciones personalizadas y retroalimentación para que te prepares mejor para la entrevista.
                6. *Metodo X (recomendado)*: Programa 1:1 de 5 sesiones para diagnosticar tu perfil, optimizar CV/LinkedIn, entrenarte en liderazgo y entrevistas, y cerrar con un plan de acción sostenible para ascender o moverte con estrategia.
                7. *Test EPI (Evaluación de Personalidad Integral)*: Aplicamos el Test EPI (Evaluación de Personalidad Integral), una herramienta diseñada para conocerte en profundidad y descubrir tu potencial personal y profesional.

                Libros y recursos en: https://xtalento.com.co

                Nota: escribe "Metodo X" en negrilla. Si el canal lo soporta, muestra la palabra "recomendado" en color gris junto al nombre; si no es posible, déjalo como (recomendado).
                Usa un emoji como 🚀 al final de la introducción.
                sin usar la palabra Hola de nuevo, recuerda que el usuario ya te saludó.
                Dile que puede elegir uno o varios servicios, marcando el número del servicio y que si quiere escoger todos marque en el char la palabara Todos
                IMPORTANTE: Solo presenta estos servicios que tienes en tu conocimiento confirmado. Si el usuario pregunta por servicios no listados, di 'Actualmente no tengo conocimiento sobre esto. Si quieres comunicarte con un humano, menciona la palabra agente en el chat.'
                """
                response_text = self._generate_response(prompt)
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            elif self.state == ConversationState.AWAITING_SERVICE_CHOICE:
                # Usar LLM para detectar intención de servicio
                service_detected = self._detect_service_intention(user_input)
                
                if not service_detected:
                    print(f"[DEBUG] No se detectó intención de servicio específico. Usando RAG para responder.")
                    # En lugar de ir a _continue_conversation, usar RAG para responder naturalmente
                    answer = self._safe_rag_answer(user_input)
                    self.chat_history.append(AIMessage(content=answer))
                    return answer

                self.user_data['service'] = user_input
                self.state = ConversationState.PROVIDING_INFO
                
                # Si el usuario elige TODOS los servicios, ofrecer diagnóstico gratuito
                normalized_choice = (user_input or "").strip().lower()
                if normalized_choice in {"Todos", "Todos los servicios", "Lista completa", "Opciones disponibles"}:
                    response_text = (
                        "¡Nos encantaría conocerte y trabajar contigo! 🎉\n\n"
                        "Te ofrecemos un diagnóstico virtual gratuito para revisar tu perfil y a partir de este diagnóstico generar junto contigo una Estrategia Laboral Personalizada.\n\n"
                        "Marca 'agenda' en el chat para que escojas tu horario disponible ⏰"
                    )
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text
                
                # Si el usuario selecciona explícitamente Metodo X, responder sin precios antes de construir el query general
                elif normalized_choice in {"metodo x", "método x", "metodo", "método", "6"}:
                    mx_prompt = (
                        "Usa EXCLUSIVAMENTE el contexto de tu conocimiento confirmado. Brinda información clara pero corta en un maximo de 200 tokens sobre 'Metodo X' SIN INCLUIR precios: "
                        "qué es, para quién aplica, beneficios, cómo funciona y resultados esperables. "
                        "IMPORTANTE: Solo habla de información que tienes confirmada en tu base de conocimiento. Si no tienes conocimiento suficiente sobre algún aspecto del Método X, di 'Actualmente no tengo conocimiento completo sobre esto. Si quieres comunicarte con un humano, menciona la palabra agente en el chat.' "
                        f"Cierra invitando a agendar una asesoría personalizada gratuita. Incluye este enlace para agendar: {CALENDAR_LINK}. Recuerda que si el cliente dice que le interesa o quiere agendar una asesoria no le digas nada sobre pagos porque esta asesoria es gratuita"
                    )
                    mx_answer = self._safe_rag_answer(mx_prompt)
                    self.chat_history.append(AIMessage(content=mx_answer))
                    return mx_answer

                # PASO 1: Enviar mensaje de diagnóstico gratuito primero
                # Aplicar retraso con indicador de escritura
                self._apply_typing_delay(user_number)
                
                diagnostic_message = (
                    f"¡Nos encantaría conocerte y trabajar contigo! 🎉\n\n"
                    f"Te ofrecemos un diagnóstico virtual (sin costo) para revisar tu perfil y a partir de este diagnóstico generar junto contigo una Estrategia Laboral Personalizada.\n\n"
                    f"Separa un espacio en el siguiente link:\n"
                    f"🗓️ {CALENDAR_LINK}\n\n"
                    f"A continuación te comparto la información detallada del servicio que seleccionaste: 👇"
                )
                self.chat_history.append(AIMessage(content=diagnostic_message))
                
                # PASO 2: Esperar un poco más antes del segundo mensaje (información detallada)
                self._apply_typing_delay(user_number)
                
                # PASO 3: Enviar información detallada con precios
                user_role = self.user_data.get('role', 'táctico')

                query = (
                    f"""
                    Usa EXCLUSIVAMENTE el contexto de tu conocimiento confirmado para responder, excepto en la política de precios indicada abajo.
                    Servicios escogidos por el usuario: "{user_input}".

                    🚨 POLÍTICA DE CONOCIMIENTO ESTRICTA:
                    - SOLO proporciona información que tienes confirmada en tu base de conocimiento.
                    - Si no tienes información completa sobre algún servicio solicitado, di: "Actualmente no tengo conocimiento completo sobre este servicio. Si quieres comunicarte con un humano, menciona la palabra 'agente' en el chat."
                    - NO inventes detalles sobre servicios, tiempos o características.
                    - NUNCA reveles o menciones la clasificación de nivel del usuario.

                    📊 POLÍTICA DE PRECIOS POR NIVEL JERÁRQUICO:
                    
                    Para el nivel OPERATIVO (cargos de ejecución directa y técnicos: analistas, desarrolladores, asistentes, operarios, técnicos, especialistas junior, consultores junior, ejecutivos de cuenta, vendedores):
                    - Hoja de vida/CV/ATS: 50.000$ (precio fijo)
                    - Mejora de perfil en plataformas: 80.000$ (precio fijo)  
                    - Para otros servicios: busca en tu base de conocimiento los precios específicos para nivel operativo
                    
                    Para el nivel TÁCTICO (cargos de supervisión y coordinación media: coordinadores, especialistas senior, jefes de área, supervisors, team leads, líderes de equipo, gerentes de área específica):
                    - Hoja de vida/CV/ATS: 50.000$ (precio fijo)
                    - Mejora de perfil en plataformas: 80.000$ (precio fijo)
                    - Para otros servicios: busca en tu base de conocimiento los precios específicos para nivel táctico
                    
                    Para el nivel ESTRATÉGICO (cargos de alta dirección y toma de decisiones: CEO, presidente, vicepresidente, director general, directores de área, gerentes generales, VP, fundadores):
                    - Hoja de vida/CV/ATS: 50.000$ (precio fijo)
                    - Mejora de perfil en plataformas: 80.000$ (precio fijo)
                    - Para otros servicios: busca en tu base de conocimiento los precios específicos para nivel estratégico
                    
                    El usuario está clasificado como nivel {user_role.upper()}. Busca los precios correspondientes a este nivel en tu base de conocimiento, excepto para los dos servicios con precio fijo mencionados arriba.
                  
                    Formato de salida (en español, claro y consistente). Sigue estos encabezados en este orden, en texto plano:
                    
                    Servicio o servicios escogidos: <lista breve de los servicios tal como aparecen en el contexto>
                    Información sobre el servicio o servicios: <qué incluye, cómo funciona y tiempos si están en contexto - SOLO si tienes la información confirmada>
                    Precio del servicio o servicios: <aplica precios específicos para nivel {user_role} según tu base de conocimiento, excepto hoja de vida=50.000$ y mejora de perfil=80.000$ que son fijos>
                    
                    - Paso 1: llenar el formulario {PAYMENT_FORM_URL} (indica que este paso es fundamental para poder seguir)

                    - Paso 2: SOLO si entre los servicios hay 'hoja de vida'/'cv'/'currículum'/'ATS'/'1'/Hoja de vida/ Hoja/ hoja/Elaboración: pedir la hoja de vida actual; si no la tiene, pedir documento con nombres, cédula, estudios y experiencias laborales. Si NO aplica, escribe: 'paso 2: (no aplica)'

                    - Paso 3: formas de pagar y confirmar pago: incluye las cuentas/medios de pago que estan en el RAG SI no acá están Banco: bancolmbia \n tipo: ahorros \n numero: 10015482343 \n titular: gina paola cano \n nequi: 3128186587.

                    Cierra indicando: 'Confirma cuando completes el formulario (paso 1) y cuando realices el pago (paso 3)'. Evita saludos iniciales. Por favor trata de no sobrepasar los 400 tokens.
                    """
                )
                answer = self._safe_rag_answer(query)
                self.chat_history.append(AIMessage(content=answer))
                
                # Retornar ambos mensajes concatenados para el sistema de logging
                return f"{diagnostic_message}\n\n{answer}"
            
            elif self.state == ConversationState.AWAITING_CONTINUE_CHOICE:
                # PRIORIDAD 1: Verificar si es un cliente retomando después de agente humano
                if self._is_returning_after_agent(user_input):
                    print(f"[DEBUG] Detectado cliente retomando conversación después de agente.")
                    # Cambiar a estado de información y dar bienvenida cálida
                    self.state = ConversationState.PROVIDING_INFO
                    welcome_message = self._generate_welcome_back_message(user_input)
                    self.chat_history.append(AIMessage(content=welcome_message))
                    return welcome_message
                
                # PRIORIDAD 2: Verificar si es una conversación natural (retomar después de pausa)
                elif self._is_natural_conversation_resumption(user_input):
                    print(f"[DEBUG] Detectado retomar conversación natural. Reseteando flujo.")
                    # Resetear el flujo como si fuera una nueva conversación
                    self.state = ConversationState.PROVIDING_INFO
                    answer = self._safe_rag_answer(user_input)
                    self.chat_history.append(AIMessage(content=answer))
                    return answer
                else:
                    # Manejar como elección 1/2/agente
                    response_text = self._handle_continue_choice(user_input)
                    self.chat_history.append(AIMessage(content=response_text))
                    
                    # Si el usuario eligió agente, mantener el estado para que el webhook detecte el bloqueo
                    if "Te conecto con un agente humano" in response_text:
                        return response_text
                    # Si eligió seguir hablando, cambiar a estado de información
                    elif "¿Qué preguntas tienes" in response_text:
                        self.state = ConversationState.PROVIDING_INFO
                        return response_text
                    # Si no entendió, mantener el mismo estado para volver a preguntar
                    else:
                        return response_text
                
            elif self.state == ConversationState.PROVIDING_INFO:
                # PRIORIDAD 1: Verificar si es un cliente retomando después de agente humano
                if self._is_returning_after_agent(user_input):
                    print(f"[DEBUG] Detectado cliente retomando conversación después de agente en PROVIDING_INFO.")
                    welcome_message = self._generate_welcome_back_message(user_input)
                    self.chat_history.append(AIMessage(content=welcome_message))
                    return welcome_message
                
                # PRIORIDAD 2: Detectar confirmación de pasos 1 y 3 usando memoria persistente
                payment_status = self._detect_payment_confirmation(user_input)
                
                # Actualizar memoria de confirmaciones
                if payment_status['paso1']:
                    self.confirmed_steps['paso1'] = True
                    print(f"[DEBUG] Actualizando memoria: Paso 1 confirmado")
                
                if payment_status['paso3']:
                    self.confirmed_steps['paso3'] = True
                    print(f"[DEBUG] Actualizando memoria: Paso 3 confirmado")
                
                # Verificar si ambos pasos están confirmados (usando memoria)
                if self.confirmed_steps['paso1'] and self.confirmed_steps['paso3']:
                    response_text = self._send_calendar_for_confirmed_payment()
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text
                
                # Manejo de confirmaciones individuales usando memoria
                elif payment_status['paso1'] and not self.confirmed_steps['paso3']:
                    response_text = (
                        "¡Confirmado! ✅ Has completado el formulario (paso 1).\n\n"
                        "Ahora te falta completar el paso 3 (realizar Y COMPLETAR el pago/transferencia) para poder agendar tu sesión virtual.\n\n"
                        "⚠️ **IMPORTANTE:** Solo confírmalo cuando YA hayas terminado de hacer el pago completamente.\n\n"
                        "✅ **Confirma SOLO cuando hayas completado el pago:**\n"
                        "• 'Realicé el pago'\n"
                        "• 'Ya pagué'\n"
                        "• 'Terminé la transferencia'\n"
                        "• 'Pago completado'\n\n"
                        "❌ **NO confirmes si solo vas a pagar o tienes preguntas.**\n\n"
                        "Una vez COMPLETADO el pago, confírmalo y te enviaré inmediatamente el link del calendario. 😊"
                    )
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text
                
                elif payment_status['paso3'] and not self.confirmed_steps['paso1']:
                    response_text = (
                        "¡Confirmado! ✅ Has completado el pago (paso 3).\n\n"
                        "Ahora te falta completar el paso 1 (llenar Y ENVIAR completamente el formulario) para poder agendar tu sesión virtual.\n\n"
                        "⚠️ **IMPORTANTE:** Solo confírmalo cuando YA hayas terminado de llenar y enviar el formulario.\n\n"
                        "✅ **Confirma SOLO cuando hayas completado el formulario:**\n"
                        "• 'Completé el formulario'\n"
                        "• 'Ya llené el formulario'\n"
                        "• 'Envié el formulario'\n"
                        "• 'Formulario terminado'\n\n"
                        "❌ **NO confirmes si solo vas a llenarlo o tienes preguntas.**\n\n"
                        "Una vez COMPLETADO el formulario, confírmalo y te enviaré inmediatamente el link del calendario. 😊"
                    )
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text
                
                # Si ya confirmó pasos previamente, recordárselo
                elif (self.confirmed_steps['paso1'] and not payment_status['paso3'] and 
                      not self.confirmed_steps['paso3']):
                    response_text = (
                        "Recuerda que ya confirmaste el formulario ✅\n\n"
                        "Solo falta que confirmes el PAGO cuando ya lo hayas completado:\n"
                        "• 'Realicé el pago'\n"
                        "• 'Ya pagué'\n"
                        "• 'Pago completado'\n\n"
                        "Una vez confirmes el pago, te envío el calendario inmediatamente. 😊"
                    )
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text
                    
                elif (self.confirmed_steps['paso3'] and not payment_status['paso1'] and 
                      not self.confirmed_steps['paso1']):
                    response_text = (
                        "Recuerda que ya confirmaste el pago ✅\n\n"
                        "Solo falta que confirmes el FORMULARIO cuando ya lo hayas completado:\n"
                        "• 'Completé el formulario'\n"
                        "• 'Ya llené el formulario'\n"
                        "• 'Formulario terminado'\n\n"
                        "Una vez confirmes el formulario, te envío el calendario inmediatamente. 😊"
                    )
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text
                
                # Opción específica SOLO para cuando el usuario explícitamente quiere ver la lista completa
                text_l = (user_input or "").strip().lower()
                
                # Usar LLM para detectar si quiere ver la lista de servicios
                if self._wants_to_see_services_list(user_input):
                    response_text = (
                        "Perfecto, sigamos. Te recuerdo nuestros servicios disponibles:\n\n"
                        "1. Optimización de Hoja de Vida (ATS)\n"
                        "2. Mejora de perfil en plataformas de empleo\n"
                        "3. Preparación para Entrevistas\n"
                        "4. Estrategia de búsqueda de empleo\n"
                        "5. Simulación de entrevista con feedback\n"
                        "6. **Método X** (recomendado)\n"
                        "7. Test EPI (Evaluación de Personalidad Integral)\n\n"
                        "¿Cuál te interesa? Puedes elegir por número o nombre del servicio."
                    )
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text

                # Si menciona temas de pago/formulario pero no confirma claramente, pedir clarificación
                if self._is_payment_related_query(user_input):
                    response_text = self._send_step_clarification_message()
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text

                # Responder vía RAG; si RAG no sabe, devolver opciones 1/2
                answer = self._safe_rag_answer(user_input)
                self.chat_history.append(AIMessage(content=answer))
                return answer

        except Exception as e:
            print(f"\n[ERROR] Ha ocurrido un problema, continuo la conversación: {e}")
            guidance = "hubo un inconveniente interno; responde de forma útil a lo último que dijo el usuario y mantén la conversación en marcha. IMPORTANTE: Solo habla de información que tienes conocimiento confirmado. Si no sabes algo específico, di 'Actualmente no tengo conocimiento sobre esto. Si quieres comunicarte con un humano, menciona la palabra agente en el chat.'"
            return self._continue_conversation(str(user_input), guidance)

    def _detect_payment_confirmation(self, user_input: str) -> dict:
        """Usa LLM para detectar si el usuario REALMENTE confirma haber completado el formulario y/o realizado el pago."""
        try:
            prompt = f"""
            Analiza MUY CUIDADOSAMENTE si el usuario está CONFIRMANDO COMPLETAMENTE haber realizado estas acciones específicas:

            PASO 1: LLENAR Y ENVIAR completamente un formulario de Google Forms
            PASO 3: REALIZAR Y COMPLETAR un pago/transferencia bancaria

            MENSAJE DEL USUARIO: "{user_input}"

            CRITERIOS ESTRICTOS:
            
            Para PASO1=SÍ (formulario):
            - Debe confirmar que YA llenó/completó/envió/terminó el formulario
            - Debe usar verbos de completitud: "completé", "llené", "envié", "terminé", "ya hice"
            - NO confirmar si solo dice "voy a llenar", "necesito llenar", "¿cómo lleno?"
            
            Para PASO3=SÍ (pago):
            - Debe confirmar que YA realizó/hizo/envió/completó el pago/transferencia
            - Debe usar verbos de completitud: "realicé", "pagué", "transferí", "ya hice", "envié"
            - NO confirmar si solo dice "voy a pagar", "necesito pagar", "¿cómo pago?"

            Responde en formato exacto:
            PASO1: SÍ/NO
            PASO3: SÍ/NO

            Ejemplos CORRECTOS:
            - "Ya completé el formulario" → PASO1: SÍ, PASO3: NO
            - "Realicé el pago" → PASO1: NO, PASO3: SÍ
            - "Terminé el formulario y pagué" → PASO1: SÍ, PASO3: SÍ
            - "Listo, envié el formulario" → PASO1: SÍ, PASO3: NO

            Ejemplos INCORRECTOS (NO confirmar):
            - "Voy a llenar el formulario" → PASO1: NO, PASO3: NO (futuro, no completado)
            - "¿Cómo pago?" → PASO1: NO, PASO3: NO (pregunta, no confirmación)
            - "Necesito hacer el pago" → PASO1: NO, PASO3: NO (necesidad, no completitud)
            - "El formulario está difícil" → PASO1: NO, PASO3: NO (comentario, no confirmación)
            - "¿Dónde está el formulario?" → PASO1: NO, PASO3: NO (pregunta ubicación)
            """
            
            response = self.llm.invoke(prompt).content.strip().upper()
            print(f"[DEBUG] LLM respuesta ESTRICTA de confirmación: {response}")
            
            # Extraer respuestas
            paso1_confirmed = "PASO1: SÍ" in response or "PASO1: SI" in response
            paso3_confirmed = "PASO3: SÍ" in response or "PASO3: SI" in response
            
            print(f"[DEBUG] Confirmaciones ESTRICTAS detectadas - Paso1: {paso1_confirmed}, Paso3: {paso3_confirmed}")
            
            return {
                'paso1': paso1_confirmed,
                'paso3': paso3_confirmed,
                'both_confirmed': paso1_confirmed and paso3_confirmed
            }
                
        except Exception as e:
            print(f"[ERROR] Error en _detect_payment_confirmation: {e}")
            # Fallback MÁS ESTRICTO con keywords de completitud únicamente
            text_lower = user_input.lower().strip()
            
            # Solo keywords que indican COMPLETITUD, no intención
            paso1_keywords = ['completé el formulario', 'llené el formulario', 'envié el formulario', 'terminé el formulario', 'formulario listo', 'formulario enviado']
            paso3_keywords = ['realicé el pago', 'hice el pago', 'pagué', 'transferí', 'pago listo', 'pago realizado', 'ya pagué']
            
            paso1_confirmed = any(keyword in text_lower for keyword in paso1_keywords)
            paso3_confirmed = any(keyword in text_lower for keyword in paso3_keywords)
            
            return {
                'paso1': paso1_confirmed,
                'paso3': paso3_confirmed,
                'both_confirmed': paso1_confirmed and paso3_confirmed
            }
    
    def _send_calendar_for_confirmed_payment(self) -> str:
        """Envía el link del calendario cuando se confirman ambos pasos."""
        return (
            f"¡Excelente! ✅ Has completado tanto el formulario como el pago.\n\n"
            f"Ahora puedes agendar tu sesión virtual de 60 minutos directamente en nuestro calendario:\n\n"
            f"🗓️ {CALENDAR_LINK}\n\n"
            f"Elige el día y horario que mejor te convenga. Una vez agendado, recibirás los detalles de confirmación.\n\n"
            f"¡Estamos listos para acompañarte en este proceso! 😊"
        )

    def _reset_confirmation_memory(self):
        """Resetea la memoria de confirmaciones de pasos."""
        self.confirmed_steps = {
            'paso1': False,
            'paso3': False
        }
        print(f"[DEBUG] Memoria de confirmaciones reseteada")

    def _get_confirmation_status_summary(self) -> str:
        """Retorna un resumen del estado actual de confirmaciones."""
        status_paso1 = "✅" if self.confirmed_steps['paso1'] else "⏳"
        status_paso3 = "✅" if self.confirmed_steps['paso3'] else "⏳"
        return f"Estado: Formulario {status_paso1} | Pago {status_paso3}"

    def _wants_to_see_services_list(self, user_input: str) -> bool:
        """Usa LLM para detectar si el usuario quiere ver la lista completa de servicios."""
        try:
            prompt = f"""
            Analiza si el siguiente mensaje del usuario indica que quiere VER, MOSTRAR o CONOCER la lista completa de servicios disponibles.

            MENSAJE DEL USUARIO: "{user_input}"

            ¿El usuario está pidiendo específicamente ver la lista de servicios disponibles?

            Responde ÚNICAMENTE:
            - "SÍ" si claramente quiere ver/mostrar/conocer todos los servicios o la lista completa
            - "NO" si pregunta sobre un servicio específico, precio, o hace otra consulta

            Ejemplos:
            - "Mostrar servicios" → SÍ
            - "¿Qué servicios tienen?" → SÍ
            - "Lista completa" → SÍ
            - "Ver opciones disponibles" → SÍ
            - "¿Cuánto cuesta la hoja de vida?" → NO (pregunta específica)
            - "Quiero el método X" → NO (selección específica)
            - "¿Cómo funciona?" → NO (pregunta general)
            """
            
            response = self.llm.invoke(prompt).content.strip().upper()
            
            if "SÍ" in response or "SI" in response:
                print(f"[DEBUG] LLM detectó solicitud de lista de servicios: {response}")
                return True
            else:
                print(f"[DEBUG] LLM NO detectó solicitud de lista de servicios: {response}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Error en _wants_to_see_services_list: {e}")
            # Fallback a detección básica por keywords
            service_list_keywords = ['mostrar servicios', 'ver servicios', 'lista de servicios', 'que servicios', 'opciones disponibles']
            return any(keyword in user_input.lower() for keyword in service_list_keywords)

    def _is_payment_related_query(self, user_input: str) -> bool:
        """Usa LLM para detectar si el usuario está haciendo consultas sobre pagos/pasos pero no confirmando."""
        try:
            prompt = f"""
            Analiza si el siguiente mensaje del usuario está haciendo una CONSULTA o PREGUNTA sobre:
            - Formularios, pasos, pagos, transferencias
            - Información sobre cómo pagar, dónde pagar, precios
            - Datos bancarios, cuentas, métodos de pago
            - Proceso de completar formularios

            Pero NO está confirmando haber completado algo.

            MENSAJE DEL USUARIO: "{user_input}"

            ¿Es una consulta/pregunta sobre temas de pago o formularios (pero no una confirmación)?

            Responde ÚNICAMENTE:
            - "SÍ" si pregunta sobre pagos/formularios pero no confirma
            - "NO" si no es relacionado con pagos/formularios, o si está confirmando

            Ejemplos:
            - "¿Cómo pago?" → SÍ (pregunta sobre pago)
            - "¿Dónde está el formulario?" → SÍ (pregunta sobre formulario)
            - "Ya pagué" → NO (es confirmación, no pregunta)
            - "¿Qué servicios tienen?" → NO (no relacionado con pagos)
            - "Información sobre precios" → SÍ (pregunta sobre pagos)
            """
            
            response = self.llm.invoke(prompt).content.strip().upper()
            
            if "SÍ" in response or "SI" in response:
                print(f"[DEBUG] LLM detectó consulta de pago: {response}")
                return True
            else:
                print(f"[DEBUG] LLM NO detectó consulta de pago: {response}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Error en _is_payment_related_query: {e}")
            # Fallback a detección básica por keywords
            query_keywords = ['formulario', 'pago', 'paso', 'transferencia', 'banco', 'cuenta', 'cómo pago', 'precio']
            return any(keyword in user_input.lower() for keyword in query_keywords)
    
    def _send_step_clarification_message(self) -> str:
        """Mensaje para clarificar los pasos cuando el usuario no confirma claramente."""
        # Mostrar estado actual de confirmaciones
        status_paso1 = "✅ COMPLETADO" if self.confirmed_steps['paso1'] else "⏳ PENDIENTE"
        status_paso3 = "✅ COMPLETADO" if self.confirmed_steps['paso3'] else "⏳ PENDIENTE"
        
        return (
            f"**ESTADO ACTUAL DE TUS PASOS:**\n"
            f"📋 **Paso 1 (Formulario):** {status_paso1}\n"
            f"💳 **Paso 3 (Pago):** {status_paso3}\n\n"
            f"Para poder enviarte el calendario, necesito que confirmes ÚNICAMENTE cuando hayas COMPLETADO totalmente cada paso:\n\n"
            f"📋 **Paso 1:** Llenar Y ENVIAR el formulario de Google Forms\n"
            f"💳 **Paso 3:** Realizar Y COMPLETAR el pago/transferencia\n\n"
            f"⚠️ **IMPORTANTE:** Solo confirma cuando ya hayas terminado completamente la acción.\n\n"
            f"✅ **Ejemplos de confirmación válida:**\n"
            f"• 'Ya completé el formulario'\n"
            f"• 'Realicé el pago'\n"
            f"• 'Terminé el formulario y pagué'\n"
            f"• 'Listo, envié el formulario'\n\n"
            f"Si necesitas ayuda personalizada, escribe **'agente'** para comunicarte con un agente de ventas. 👥"
        )


# --- Funciones de Soporte ---
def load_documents():
    """Carga los documentos desde el directorio especificado."""
    loader = DirectoryLoader(DOCUMENTS_PATH, glob="**/*.*", loader_cls=lambda p: UnstructuredFileLoader(p))
    return loader.load()

def create_vector_store(documents):
    """Crea y guarda el almacén de vectores FAISS."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore

def load_vector_store():
    """Carga el almacén de vectores FAISS si existe."""
    if os.path.exists(VECTORSTORE_PATH):
        embeddings = OpenAIEmbeddings()
        return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

def main():
    """Función principal para ejecutar el chatbot."""
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] La variable de entorno OPENAI_API_KEY no fue encontrada.")
        print("Por favor, asegúrate de tener un archivo .env en la raíz del proyecto con la línea: OPENAI_API_KEY='tu_clave_aqui'")
        return

    vectorstore = load_vector_store()
    if not vectorstore:
        print("Creando almacén de vectores por primera vez...")
        documents = load_documents()
        vectorstore = create_vector_store(documents)
        print("Almacén de vectores creado y guardado.")

    chatbot = Chatbot(vectorstore)
    
    print(f"Xtalento: {chatbot.process_message(None)}")
    
    while True:
        user_input = input("\nTú: ")
        if user_input.lower() == 'salir':
            print("¡Hasta luego!")
            break
        
        response = chatbot.process_message(user_input)
        print(f"\nXtalento: {response}")

if __name__ == "__main__":
    main()
