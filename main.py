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
load_dotenv()

# --- Constantes ---     
DOCUMENTS_PATH = "documents"
VECTORSTORE_PATH = "vectorstore"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 64
OPENAI_MODEL = "gpt-4o-mini"

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
        self.llm = ChatOpenAI(model_name=OPENAI_MODEL, max_tokens=200) # Aumentamos un poco por si acaso
        
        system_prompt = """
        Actuás como Xtalento Bot, un asistente profesional cálido, claro y experto que guía a personas a potenciar su perfil laboral y encontrar empleo más rápido.
        El nombre del usuario es {user_name}. Cuando sea natural y amigable, utiliza su nombre para personalizar la conversación. Si no sabes el nombre (porque está vacío), no intentes inventarlo.
        Importante: No utilices la palabra 'Hola' en ninguna de tus respuestas, ya que el saludo inicial ya fue dado.

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

        Usá emojis con calidez, sin perder profesionalismo. Tus respuestas deben ser concretas, persuasivas y con orientación clara a la acción.
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

    def _generate_response(self, prompt_text):
        """Genera una respuesta directa del LLM para mensajes conversacionales."""
        # Usamos el mismo LLM pero sin el contexto de RAG
        response = self.llm.invoke(prompt_text)
        return response.content.strip()

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
                    print(f"[DEBUG] Se detectó una pregunta en lugar de nombre/ciudad. Usando RAG.")
                    response = self.rag_chain.invoke({"input": user_input, "chat_history": self.chat_history, "user_name": self.user_data['name']})
                    self.chat_history.append(AIMessage(content=response["answer"]))
                    return response['answer']
                
                self.user_data['name_city'] = user_input
                user_name = user_input.split()[0]
                self.user_data['name'] = user_name
                self.state = ConversationState.AWAITING_ROLE_INPUT
                prompt = f"Actúas como Xtalento Bot. El usuario se llama {user_name}. Dale una bienvenida personalizada (sin usar la palabra 'Hola') y luego pregúntale sobre su cargo actual o al que aspira para poder darle una mejor asesoría."
                response_text = self._generate_response(prompt)
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            elif self.state == ConversationState.AWAITING_ROLE_INPUT:
                role_classification = self._classify_role(user_input)
                if not role_classification:
                    print(f"[DEBUG] La clasificación del rol para '{user_input}' falló. Asumiendo que es una pregunta para RAG.")
                    response = self.rag_chain.invoke({"input": user_input, "chat_history": self.chat_history, "user_name": self.user_data['name']})
                    self.chat_history.append(AIMessage(content=response["answer"]))
                    return response['answer']

                self.user_data['role'] = role_classification
                self.state = ConversationState.AWAITING_SERVICE_CHOICE
                prompt = f"""
                Actúas como Xtalento Bot. Acabas de clasificar el cargo del usuario como '{role_classification}'.
                Confirma esta clasificación de forma amigable y luego presenta los siguientes servicios en una lista numerada:
                1. Optimización de Hoja de Vida (ATS)
                2. Mejora de perfil en plataformas de empleo
                3. Preparación para Entrevistas
                4. Estrategia de búsqueda de empleo
                Usa un emoji como 🚀 al final de la introducción.
                """
                response_text = self._generate_response(prompt)
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            elif self.state == ConversationState.AWAITING_SERVICE_CHOICE:
                service_keywords = ['hoja de vida', 'ats', 'perfil', 'plataforma', 'entrevista', 'estrategia', 'búsqueda', '1', '2', '3', '4']
                is_service_choice = any(keyword in user_input.lower() for keyword in service_keywords)

                if not is_service_choice:
                    print(f"[DEBUG] No se detectó una selección de servicio en '{user_input}'. Usando RAG.")
                    response = self.rag_chain.invoke({"input": user_input, "chat_history": self.chat_history, "user_name": self.user_data['name']})
                    self.chat_history.append(AIMessage(content=response["answer"]))
                    return response['answer']

                self.user_data['service'] = user_input
                self.state = ConversationState.PROVIDING_INFO
                
                query = f"Dime el precio y los detalles del servicio '{user_input}' para un cargo de nivel '{self.user_data.get('role', 'táctico')}'"
                response = self.rag_chain.invoke({"input": query, "chat_history": self.chat_history, "user_name": self.user_data['name']})
                self.chat_history.append(AIMessage(content=response["answer"]))
                return response['answer']
                
            elif self.state == ConversationState.PROVIDING_INFO:
                response = self.rag_chain.invoke({"input": user_input, "chat_history": self.chat_history, "user_name": self.user_data['name']})
                self.chat_history.append(AIMessage(content=response["answer"]))
                return response['answer']

        except Exception as e:
            print(f"\n[ERROR] Ha ocurrido un error al procesar el mensaje: {e}")
            return "Lo siento, estoy teniendo un problema técnico en este momento. Por favor, intenta de nuevo en unos segundos."


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
