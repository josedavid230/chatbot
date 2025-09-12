



import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

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

# --- Estados de Conversación ---
class ConversationState:
    AWAITING_GREETING = "AWAITING_GREETING"
    AWAITING_NAME_CITY = "AWAITING_NAME_CITY"
    AWAITING_ROLE_INPUT = "AWAITING_ROLE_INPUT"
    AWAITING_SERVICE_CHOICE = "AWAITING_SERVICE_CHOICE"
    AWAITING_CONTINUE_CHOICE = "AWAITING_CONTINUE_CHOICE"
    PROVIDING_INFO = "PROVIDING_INFO"

# --- Lógica del Chatbot ---
class Chatbot:
    def __init__(self, vectorstore):
        self.state = ConversationState.AWAITING_GREETING
        self.user_data = {}
        self.user_data['name'] = "" # Se inicializa el nombre del usuario
        self.chat_history = []
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
    
    def _detect_scheduling_request(self, user_input: str) -> bool:
        """Detecta si el usuario quiere agendar una cita o sesión."""
        scheduling_keywords = [
            'agendar', 'agenda', 'agendo', 'sesión', 'sesion', 'cita', 'reunión', 'reunion',
            'calendario', 'hora', 'horario', 'cuando', 'cuándo', 'disponible', 'disponibilidad',
            'programar', 'programa', 'appointment', 'meeting', 'schedule', 'virtual',
            'asesoría', 'asesoria', 'consulta', 'mentoria', 'mentoría'
        ]
        text_lower = user_input.lower().strip()
        return any(keyword in text_lower for keyword in scheduling_keywords)
    
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
        if any(x in text_lower for x in ["1", "uno", "agente", "humano", "ventas"]):
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

    def process_message(self, user_input):
        try:
            # PRIORIDAD MÁXIMA: Detectar solicitud de agente humano ANTES de cualquier procesamiento
            # EXCEPCIÓN: No detectar "agente" cuando el usuario está describiendo su cargo laboral
            if (user_input and "agente" in user_input.lower() and 
                self.state != ConversationState.AWAITING_ROLE_INPUT):
                # Agregar el mensaje del usuario al historial antes de responder
                self.chat_history.append(HumanMessage(content=user_input))
                response = "Perfecto. Te conecto con un agente humano inmediatamente. Pauso este chat y un agente de ventas te contactará en este mismo canal."
                self.chat_history.append(AIMessage(content=response))
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
                    answer = self._safe_rag_answer(user_input)
                    self.chat_history.append(AIMessage(content=answer))
                    return answer
                
                self.user_data['name_city'] = user_input
                user_name = self._extract_name(user_input)
                self.user_data['name'] = user_name
                self.state = ConversationState.AWAITING_ROLE_INPUT
                prompt = f"Actúas como Xtalento Bot. El usuario se llama {user_name}. Dale una bienvenida personalizada (sin usar la palabra 'Hola') y luego pregúntale sobre su cargo actual o al que aspira para poder darle una mejor asesoría. IMPORTANTE: Solo habla de servicios que tienes conocimiento confirmado. Si no sabes algo específico, di 'Actualmente no tengo conocimiento sobre esto. Si quieres comunicarte con un humano, menciona la palabra agente en el chat.'"
                response_text = self._generate_response(prompt)
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            elif self.state == ConversationState.AWAITING_ROLE_INPUT:
                role_classification = self._classify_role(user_input)
                if not role_classification:
                    print(f"[DEBUG] No se pudo clasificar el rol. Continuando la conversación sin error.")
                    self.state = ConversationState.AWAITING_CONTINUE_CHOICE
                    response_text = self._continue_conversation(user_input, "")
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text

                self.user_data['role'] = role_classification
                self.state = ConversationState.AWAITING_SERVICE_CHOICE
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
                service_keywords = ['hoja de vida','Hoja', 'Hoja de vida', 'Optimización', 'Optimización de Hoja de vida', 'ats','Optimización de Hoja de vida (ATS)','Mejora de perfil en plataformas de empleo','Preparación para Entrevistas','Preparacion para Entrevistas','Estrategia de búsqueda de empleo','Estrategia de busqueda de empleo','Simulación de entrevista con feedback','Simulacion de entrevista con feedback','Metodo X','Test EPI','Evaluación de Personalidad Integral','1', '2', '3', '4', '5', '6','7', 'mejora','mejorar','preparación', 'metodo x', 'método x'  ]
                is_service_choice = any(keyword in user_input.lower() for keyword in service_keywords)

                if not is_service_choice:
                    print(f"[DEBUG] No se detectó una selección de servicio. Continuando sin interrumpir.")
                    self.state = ConversationState.AWAITING_CONTINUE_CHOICE
                    response_text = self._continue_conversation(user_input, "")
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text

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
                return answer
            
            elif self.state == ConversationState.AWAITING_CONTINUE_CHOICE:
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
                # PRIORIDAD: Detectar confirmación de pasos 1 y 3 para enviar calendario
                payment_status = self._detect_payment_confirmation(user_input)
                if payment_status['both_confirmed']:
                    response_text = self._send_calendar_for_confirmed_payment()
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text
                
                # Detectar confirmación individual de pasos para dar retroalimentación
                elif payment_status['paso1'] and not payment_status['paso3']:
                    response_text = (
                        "¡Perfecto! ✅ Has completado el formulario (paso 1).\n\n"
                        "Ahora solo falta confirmar el pago (paso 3) para poder agendar tu sesión virtual.\n\n"
                        "Una vez realices el pago, confírmalo aquí para enviarte el link del calendario. 😊"
                    )
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text
                
                elif payment_status['paso3'] and not payment_status['paso1']:
                    response_text = (
                        "¡Excelente! ✅ Has confirmado el pago (paso 3).\n\n"
                        "Ahora solo falta completar el formulario (paso 1) para poder agendar tu sesión virtual.\n\n"
                        "Una vez llenes el formulario, confírmalo aquí para enviarte el link del calendario. 😊"
                    )
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text
                
                # Opción específica SOLO para cuando el usuario explícitamente quiere ver la lista completa
                text_l = (user_input or "").strip().lower()
                
                # Frases más específicas que realmente indican que quiere ver todos los servicios
                show_services_phrases = [
                    "mostrar servicios", "ver servicios", "lista de servicios", "todos los servicios",
                    "que servicios tienen", "cuales servicios", "opciones disponibles", 
                    "mostrar opciones", "ver opciones", "lista completa"
                ]
                
                # Solo activar la plantilla si es muy específico
                if any(phrase in text_l for phrase in show_services_phrases):
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
        """Detecta si el usuario confirma el paso 1 (formulario) y/o paso 3 (pago)."""
        text_lower = user_input.lower().strip()
        
        # Palabras clave para confirmar paso 1 (formulario)
        paso1_keywords = [
            'Completé el formulario', 'Llené el formulario', 'Formulario listo', 
            'Formulario completo', 'Ya llené', 'Ya completé', 'Paso 1 listo',
            'Paso uno listo', 'Formulario enviado', 'Envié el formulario', 'Listo paso uno',
            'Listo paso 1', 'Confirmo paso 1', 'Confirmo paso uno'
        ]
        
        # Palabras clave para confirmar paso 3 (pago)
        paso3_keywords = [
            'Realicé el pago', 'Hice el pago', 'Pago realizado', 'Pago listo',
            'Ya pagué', 'Pagué', 'Transferencia realizada', 'Paso 3 listo',
            'Paso tres listo', 'Pago confirmado', 'Envié el pago'
        ]
        
        # Detectar confirmaciones
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
    
    def _is_payment_related_query(self, user_input: str) -> bool:
        """Detecta si el usuario está haciendo una consulta relacionada con pagos/pasos pero no confirmando."""
        text_lower = user_input.lower().strip()
        
        # Palabras que indican que está preguntando sobre los pasos pero no confirmando
        query_keywords = [
            'formulario', 'pago', 'paso', 'transferencia', 'banco', 'cuenta',
            'cómo pago', 'donde pago', 'cuánto cuesta', 'precio', 'valor',
            'información', 'datos', 'llenar', 'completar', 'enviar'
        ]
        
        return any(keyword in text_lower for keyword in query_keywords)
    
    def _send_step_clarification_message(self) -> str:
        """Mensaje para clarificar los pasos cuando el usuario no confirma claramente."""
        return (
            "Para continuar con tu proceso, necesito que confirmes los pasos completados:\n\n"
            "📋 **Paso 1:** Llenar formulario\n"
            "💳 **Paso 3:** Realizar pago\n\n"
            "Por favor confirma cuáles has completado, ejemplo:\n"
            "• 'Completé el formulario'\n"
            "• 'Realicé el pago'\n"
            "• 'Completé formulario y pago'\n\n"
            "Si necesitas ayuda personalizada, escribe **'agente'** para comunicarte con un agente de ventas. 👥"
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
