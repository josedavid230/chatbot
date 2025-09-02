



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

# --- Estados de Conversación ---
class ConversationState:
    AWAITING_GREETING = "AWAITING_GREETING"
    AWAITING_NAME_CITY = "AWAITING_NAME_CITY"
    AWAITING_ROLE_INPUT = "AWAITING_ROLE_INPUT"
    AWAITING_SERVICE_CHOICE = "AWAITING_SERVICE_CHOICE"
    PROVIDING_INFO = "PROVIDING_INFO"

# --- Lógica del Chatbot ---
class Chatbot:
    def __init__(self, vectorstore):
        self.state = ConversationState.AWAITING_GREETING
        self.user_data = {}
        self.user_data['name'] = "" # Se inicializa el nombre del usuario
        self.chat_history = []
        self.llm = ChatOpenAI(model_name=OPENAI_MODEL, max_tokens=400, temperature=0.2)
        
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

        Política de conocimiento y ventas:
        - Responde solo con información que tengas en tu conocimiento y en el contexto proporcionado (RAG). No inventes servicios ni datos.
        - Si no cuentas con información suficiente para responder con precisión, NO improvises.
        - En su lugar, ofrece dos opciones:
          1) Derivar a un agente de ventas humano (indicando que el agente responderá en ~3 horas y que se pausará este chat)
          2) Explorar otros servicios disponibles (estrictamente los que aparezcan en el contexto)
        - Si el usuario decide seguir con el bot, tu objetivo principal es vender un servicio y proponer agendar una sesión virtual.
        - Usa emojis con calidez, sin perder profesionalismo. Sé concreto y con orientación clara a la acción.
        """
        
        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = """Dada una conversación y una pregunta de seguimiento, reformula la pregunta de seguimiento para que sea una pregunta independiente, en su idioma original. El nombre del usuario es {user_name}."""
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

    def _classify_role(self, role_description):
        """Usa el LLM para clasificar un rol en operativo, táctico o estratégico."""
        classification_prompt_text = f"""
        Clasifica el siguiente cargo únicamente como 'operativo', 'táctico' o 'estratégico'. No agregues ninguna otra palabra o explicación.
        Cargo: "{role_description}"
        Clasificación:
        """
        response = self.llm.invoke(classification_prompt_text)
        # Aseguramos que la respuesta sea una de las tres opciones válidas
        classification = response.content.strip().lower()
        if classification in ['operativo', 'táctico', 'estratégico']:
            return classification
        return None # Devolvemos None si la clasificación falla

    def _extract_name(self, name_city_text):
        """Usa el LLM para extraer el nombre de pila del usuario de un texto."""
        extraction_prompt_text = f"""
        De la siguiente frase, extrae únicamente el nombre de pila del usuario.
        Ejemplo: si la frase es "Soy Carlos de Lima", la respuesta debe ser "Carlos".
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
            "No tengo suficiente información para darte una respuesta precisa ahora mismo. "
            "¿Qué prefieres que hagamos?\n\n"
            "1) Hablar con un agente de ventas humano (pausamos este chat y te contactarán en ~3 horas).\n"
            "2) Seguir conmigo y explorar otros servicios de Xtalento disponibles."
        )

    def _continue_conversation(self, user_text: str, guidance: str) -> str:
        """Genera una respuesta que mantenga la conversación sin depender de RAG cuando algo no se puede clasificar o detectar."""
        prompt = (
            "Actúas como Xtalento Bot. "
            f"Mensaje del usuario: '{user_text}'. "
            f"Objetivo: {guidance}. "
            "Responde de forma clara, útil y breve; si corresponde, haz una pregunta para avanzar."
        )
        return self._generate_response(prompt)

    def process_message(self, user_input):
        try:
            # Los saludos iniciales no necesitan memoria ni RAG
            if self.state == ConversationState.AWAITING_GREETING:
                self.state = ConversationState.AWAITING_NAME_CITY
                prompt = "Actúas como Xtalento Bot. Genera un saludo inicial cálido y profesional que comience exactamente con la palabra '¡Hola! 👋'. A continuación, preséntate brevemente y pide al usuario su nombre y la ciudad desde la que escribe."
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
                prompt = f"Actúas como Xtalento Bot. El usuario se llama {user_name}. Dale una bienvenida personalizada (sin usar la palabra 'Hola') y luego pregúntale sobre su cargo actual o al que aspira para poder darle una mejor asesoría."
                response_text = self._generate_response(prompt)
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            elif self.state == ConversationState.AWAITING_ROLE_INPUT:
                role_classification = self._classify_role(user_input)
                if not role_classification:
                    print(f"[DEBUG] No se pudo clasificar el rol. Continuando la conversación sin error.")
                    self.state = ConversationState.AWAITING_SERVICE_CHOICE
                    guidance = (
                        "no se pudo clasificar el cargo; presenta la lista de servicios (1-4) y pide al usuario elegir uno, "
                        "o que cuente brevemente su cargo para recomendarle mejor"
                    )
                    response_text = self._continue_conversation(user_input, guidance)
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text

                self.user_data['role'] = role_classification
                self.state = ConversationState.AWAITING_SERVICE_CHOICE
                prompt = f"""
                Actúas como Xtalento Bot. Presenta los siguientes servicios en una lista numerada sin mencionar ni revelar la categoría/nivel del usuario:
                1. Optimización de Hoja de Vida (ATS)
                2. Mejora de perfil en plataformas de empleo
                3. Preparación para Entrevistas
                4. Estrategia de búsqueda de empleo
                5. Simulación de entrevista con feedback
                6. *Metodo X* (recomendado)
                Nota: escribe "Metodo X" en negrilla. Si el canal lo soporta, muestra la palabra "recomendado" en color gris junto al nombre; si no es posible, déjalo como (recomendado).
                Usa un emoji como 🚀 al final de la introducción.
                sin usar la palabra Hola de nuevo, recuerda que el usuario ya te saludó.
                Dile que puede elegir uno o varios servicios, marcando el número o diciendo el nombre del servicio.
                """
                response_text = self._generate_response(prompt)
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            elif self.state == ConversationState.AWAITING_SERVICE_CHOICE:
                service_keywords = ['hoja de vida', 'ats', 'perfil', 'plataforma', 'entrevista', 'estrategia', 'búsqueda', '1', '2', '3', '4', '5', '6', 'mejora','mejorar','preparación', 'metodo x', 'método x']
                is_service_choice = any(keyword in user_input.lower() for keyword in service_keywords)

                if not is_service_choice:
                    print(f"[DEBUG] No se detectó una selección de servicio. Continuando sin interrumpir.")
                    guidance = (
                        "orienta brevemente sobre los servicios disponibles (1-6) y solicita elegir uno o varios; "
                        "si el usuario hizo una pregunta, respóndela con tu conocimiento general y vuelve a ofrecer la lista"
                    )
                    response_text = self._continue_conversation(user_input, guidance)
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text

                self.user_data['service'] = user_input
                self.state = ConversationState.PROVIDING_INFO
                
                # Si el usuario selecciona explícitamente Metodo X, responder sin precios antes de construir el query general
                normalized_choice = (user_input or "").strip().lower()
                if normalized_choice in {"metodo x", "método x", "metodo", "método", "6"}:
                    mx_prompt = (
                        "Usa EXCLUSIVAMENTE el contexto. Brinda información clara pero corta en un maximo de 200 tokens sobre 'Metodo X' SIN INCLUIR precios: "
                        "qué es, para quién aplica, beneficios, cómo funciona y resultados esperables. "
                        "Cierra invitando a agendar una asesoría personalizada gratuita con un asesor para conocer por qué este paquete sería adecuado y el beneficio de comprarlo. recuerda que si el cliente dice que le interesa o quiere agendar una asesoria no le digas nada sobre pagos porque esta asesoria es gratuita"
                    )
                    mx_answer = self._safe_rag_answer(mx_prompt)
                    self.chat_history.append(AIMessage(content=mx_answer))
                    return mx_answer

                query = (
                    f"""
                    Usa EXCLUSIVAMENTE el contexto para responder, excepto en la política de precios indicada abajo.
                    Servicios escogidos por el usuario: "{user_input}".
                    Nivel de cargo del usuario: "{self.user_data.get('role', 'táctico')}".

                    Política de precios (aplica SIEMPRE, independientemente del rol):
                    - Para servicios relacionados con 'hoja de vida'/'HV'/'CV'/'currículum'/'ATS' (o el servicio 1): el precio es 50.000$.
                    - Para 'Mejora de perfil en plataformas de empleo' (o el servicio 2): el precio es 80.000$.
                    - Si el contexto muestra otros precios para esos dos servicios, IGNÓRALOS y aplica esta política.

                  
                    Formato de salida (en español, claro y consistente). Sigue estos encabezados en este orden, en texto plano:
                    - servicio o servicios escogidos: <lista breve de los servicios tal como aparecen en el contexto>
                    - informacion sobre el servicio o servicios: <qué incluye, cómo funciona y tiempos si están en contexto>
                    - precio del servicio o servicios: <aplica la política de precios arriba descrita para (1) Hoja de vida = 50.000$ y (2) Mejora de perfil = 80.000$; para el resto usa el contexto o indica si falta>
                    - paso 1: llenar el formulario {PAYMENT_FORM_URL} (indica que este paso es fundamental para poder seguir)
                    - paso 2: SOLO si entre los servicios hay 'hoja de vida'/'cv'/'currículum'/'ATS'/'1'/Elaboración: pedir la hoja de vida actual; si no la tiene, pedir documento con nombres, cédula, estudios y experiencias laborales. Si NO aplica, escribe: 'paso 2: (no aplica)'
                    - paso 3: formas de pagar y confirmar pago: incluye las cuentas/medios de pago que estan en el RAG SI no acá están Banco: bancolmbia \n tipo: ahorros \n numero: 10015482343 \n titular: gina paola cano \n nequi: 3128186587.

                    Cierra indicando: 'Confirma cuando completes el formulario (paso 1) y cuando realices el pago (paso 3)'. Evita saludos iniciales. Por favor trata de no sobrepasar los 400 tokens.
                    """
                )
                answer = self._safe_rag_answer(query)
                self.chat_history.append(AIMessage(content=answer))
                return answer
                
            elif self.state == ConversationState.PROVIDING_INFO:
                # Detectar elección del usuario cuando no sabemos responder
                text_l = (user_input or "").strip().lower()
                if any(x in text_l for x in ["agente", "humano", "ventas", "persona", "asesor"]) and any(x in text_l for x in ["hablar", "quiero", "prefiero", "conectar", "contacto"]):
                    self.state = ConversationState.PROVIDING_INFO
                    return (
                        "Perfecto. Pauso este chat y un agente de ventas te contactará en este mismo canal. "
                        "Si deseas retomar con el bot más tarde, inicia una nueva conversación."
                    )

                # Opción de seguir con el bot y explorar servicios
                if any(x in text_l for x in ["seguir", "continuar", "bot", "opciones", "servicios", "2", "dos"]):
                    guidance = (
                        "presenta servicios disponibles de Xtalento (solo los que estén en el contexto del RAG) y guía a escoger uno; "
                        "propón agendar una sesión virtual como siguiente paso, quiero que si el cliente dice que le interesa el metodo x  (solo con el metodo x) y quiere agendar una asesoria no le digas nada sobre pagos porque esta asesoria es gratuita"
                    )
                    response_text = self._continue_conversation(user_input, guidance)
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text

                # Responder vía RAG; si RAG no sabe, devolver opciones 1/2
                answer = self._safe_rag_answer(user_input)
                self.chat_history.append(AIMessage(content=answer))
                return answer

        except Exception as e:
            print(f"\n[ERROR] Ha ocurrido un problema, continuo la conversación: {e}")
            guidance = "hubo un inconveniente interno; responde de forma útil a lo último que dijo el usuario y mantén la conversación en marcha"
            return self._continue_conversation(str(user_input), guidance)


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
