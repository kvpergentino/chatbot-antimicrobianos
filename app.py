# update teste v2  
import streamlit as st  
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


# FUNÇÕES E CONFIGURAÇÕES DO BACKEND

# Usamos o cache do streamlit para carregar os modelos e clientes
@st.cache_resource
def carregar_recursos():
    """
    Carrega os recursos essenciais (chaves de API, modelos de IA)
    que não mudam entre as execuções.
    """
    # st.secrets para o deploy no streamlit community cloud
    try:
        # Tenta pegar dos segredos do streamlit
        google_api_key = st.secrets["GOOGLE_API_KEY"]
        qdrant_url = st.secrets["QDRANT_URL"]
        qdrant_api_key = st.secrets["QDRANT_API_KEY"]
    except Exception:
        # Fallback para userdata do colab
        from google.colab import userdata
        google_api_key = userdata.get('GOOGLE_API_KEY')
        qdrant_url = userdata.get('QDRANT_URL')
        qdrant_api_key = userdata.get('QDRANT_API_KEY')

    # Inicializa os modelos com a chave carregada
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )
    llm_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key=google_api_key
    )
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    return embeddings_model, llm_model, qdrant_client

def formatar_documentos(docs):
    """
    Formata a lista de documentos recuperados para serem inseridos no prompt.
    """
    return "\n\n".join(f"Fonte: {doc.metadata.get('fonte', 'N/A')}\nConteúdo: {doc.page_content}" for doc in docs)

@st.cache_resource
def criar_pipeline_rag(_qdrant_client, _embeddings_model, _llm_model, collection_name="chatbot_antimicrobianos_v1", k_retriever=3):
    """
    Cria e retorna o pipeline RAG completo usando os recursos já carregados.
    O decorator _qdrant_client garante que o pipeline seja criado apenas uma vez.
    """
    vector_store = Qdrant(
        client=_qdrant_client,
        collection_name=collection_name,
        embeddings=_embeddings_model,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": k_retriever})

    template = """
    Você é um assistente de IA especializado em fornecer informações sobre a prescrição de antimicrobianos para médicos.
    Sua tarefa é responder à pergunta do usuário de forma clara, concisa e precisa, baseando-se EXCLUSIVAMENTE no contexto fornecido.
    Se a informação necessária para responder à pergunta não estiver no contexto, responda exatamente: "A informação para responder a esta pergunta não foi encontrada na base de dados."
    Não adicione nenhuma informação que não esteja explicitamente no texto de contexto. Cite a fonte da informação se ela estiver disponível no contexto.

    Contexto:
    {context}

    Pergunta do Usuário:
    {question}

    Resposta do Assistente:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    rag_chain = (
        {"context": retriever | formatar_documentos, "question": RunnablePassthrough()}
        | prompt
        | _llm_model
        | StrOutputParser()
    )
    return rag_chain


# INTERFACE DO USUÁRIO - STREAMLIT


# Título da aplicação
st.set_page_config(page_title="PseudomonIA", page_icon="🦠") # Um apelido carinhoso #teamPseudomonas
st.title("🦠 Chatbot para Suporte à Prescrição de Antimicrobianos")
st.info("Este chatbot foi desenvolvido como um projeto acadêmico e utiliza a arquitetura RAG para responder perguntas acerca do tratamento de infeções com base em uma fonte de conhecimento específica, neste caso, o 'The WHO AWaRe (Access, Watch, Reserve) antibiotic book', da Ogrnização Mundial da Saúde. As respostas não substituem o julgamento clínico.")

# Carrega os recursos uma única vez
try:
    embeddings, llm, qdrant_client = carregar_recursos()
    
    # Cria o pipeline uma única vez
    rag_chain = criar_pipeline_rag(qdrant_client, embeddings, llm)

    # Inicializa o histórico do chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibe as mensagens do histórico
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input do usuário
    if prompt_usuario := st.chat_input("Qual sua dúvida sobre prescrição de antimicrobianos?"):
        # Adiciona a mensagem do usuário ao histórico e exibe
        st.session_state.messages.append({"role": "user", "content": prompt_usuario})
        with st.chat_message("user"):
            st.markdown(prompt_usuario)

        # Gera e exibe a resposta do chatbot
        with st.chat_message("assistant"):
            with st.spinner("Analisando a base de conhecimento..."):
                resposta = rag_chain.invoke(prompt_usuario)
                st.markdown(resposta)
        
        # Adiciona a resposta do assistente ao histórico
        st.session_state.messages.append({"role": "assistant", "content": resposta})

except Exception as e:
    st.error(f"Ocorreu um erro ao carregar a aplicação: {e}. Verifique as chaves de API e a configuração.")
