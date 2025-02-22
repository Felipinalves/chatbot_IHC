__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb
import time
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma.base import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import google.generativeai as genai

# Configuração da página
st.set_page_config(
    page_title="IAHC Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

# Título do app
st.title("Chatbot Especialista em IHC 🤖")
st.info("Este chatbot utiliza RAG (Retrieval Augmented Generation) para fornecer respostas precisas sobre IHC.", icon="ℹ️")

# Inicialização das configurações e estados
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Olá! Sou um especialista em IHC. Como posso ajudar você hoje?"
        }
    ]

@st.cache_resource(show_spinner=False)
def initialize_system():
    try:
        # Configurar o Gemini
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        
        # Configurar o modelo de embedding
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        Settings.llm = None
        
        # Inicializar ChromaDB
        db = chromadb.PersistentClient(path="chroma_db")
        chroma_collection = db.get_or_create_collection("quickstart")
        
        # Configurar vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Carregar documentos e criar índice
        documents = SimpleDirectoryReader("./arquivosFormatados").load_data()
        index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context,
            show_progress=True
        )
        
        return index.as_query_engine()
    except Exception as e:
        st.error(f"Erro na inicialização do sistema: {str(e)}")
        return None

def generate_response_with_gemini(prompt, max_retries=3):
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }
    
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config
    )
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            if response.text:
                return response.text
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            st.error(f"Erro ao gerar resposta: {str(e)}")
            return None

# Main execution
try:
    # Initialize the system
    query_engine = initialize_system()
    
    if query_engine is None:
        st.error("Não foi possível inicializar o sistema. Por favor, verifique os logs e tente novamente.")
    else:
        # Interface do chat
        if prompt := st.chat_input("Faça uma pergunta sobre IHC"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Mostrar indicador de "pensando"
            with st.status("Processando sua pergunta...", expanded=True) as status:
                st.write("Buscando informações relevantes...")
                
                # Recuperar contexto
                context = str(query_engine.query(prompt))
                
                st.write("Gerando resposta...")
                # Gerar prompt completo
                full_prompt = f"""[INSTRUÇÕES DETALHADAS]
    Você é um especialista em IHC (Interação Humano-Computador) com vasta experiência acadêmica e prática.
    Sua tarefa é fornecer respostas precisas e bem fundamentadas, baseadas exclusivamente no contexto fornecido.
    [DIRETRIZES DE RESPOSTA]
    1. Linguagem: Use português brasileiro formal
    2. Termos técnicos: Mantenha termos técnicos consolidados em inglês
    3. Estrutura: Organize a resposta em parágrafos claros e concisos
    4. Citações: Mencione autores/fontes quando relevante
    5. Exemplos: Inclua exemplos práticos quando apropriado
    [CONTEXTO ACADÊMICO]
    {context}
    [PERGUNTA]
    {prompt}
    [FORMATO ESPERADO]
    1. Inicie com uma resposta direta à pergunta
    2. Desenvolva a explicação com detalhes relevantes
    3. Conclua com uma síntese prática
    4. Se houver divergências na literatura, apresente as diferentes visões
    [RESPOSTA EM PORTUGUÊS]
    """
                # Gerar resposta
                response = generate_response_with_gemini(full_prompt)
                
                if response:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                status.update(label="Resposta gerada!", state="complete", expanded=False)
        
        # Exibir histórico de mensagens
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

except Exception as e:
    st.error(f"Erro durante a execução: {str(e)}")
