import os
from dotenv import load_dotenv

# Cargar automáticamente la API Key desde el archivo .env
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def inicializar_agente(ruta_pdf: str):
    if not os.environ.get("COHERE_API_KEY"):
        raise ValueError("Por favor, configura la variable de entorno COHERE_API_KEY en el archivo .env")

    # 1. Cargar y segmentar el PDF
    loader = PyPDFLoader(ruta_pdf)
    paginas = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    fragmentos = text_splitter.split_documents(paginas)

    # 2. Crear los Embeddings y almacenar en la Base de Datos Vectorial (Chroma)
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
    vector_store = Chroma.from_documents(fragmentos, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 3. Configurar el Modelo de Lenguaje vigente de Cohere
    llm = ChatCohere(model="command-r7b-12-2024", temperature=0.3)

    # 4. Diseñar el Sistema de Preguntas y Respuestas (Prompt)
    system_prompt = (
        "Eres un asistente experto en responder preguntas sobre documentos internos de la empresa.\n"
        "Utiliza los siguientes fragmentos de contexto recuperados para responder la pregunta.\n"
        "Si no sabes la respuesta o no está en el documento, di claramente que no dispones de esa información.\n"
        "Mantén la respuesta concisa y profesional.\n\n"
        "Contexto:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 5. Crear la cadena final usando LCEL
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

if __name__ == "__main__":
    PATH_PDF = "data/politicas_empresa.pdf"
    
    if os.path.exists(PATH_PDF):
        print("🤖 Inicializando agente con tu documento local...")
        agente = inicializar_agente(PATH_PDF)
        
        pregunta = "¿Cuáles son los puntos clave o políticas mencionadas en el documento?"
        print(f"\nPregunta ejecutada por defecto: {pregunta}")
        print("-" * 40)
        
        respuesta = agente.invoke(pregunta)
        print(f"Respuesta de la IA:\n{respuesta}")
    else:
        print(f"⚠️ Archivo no encontrado. Por favor coloca tu PDF en la ruta: {PATH_PDF}")