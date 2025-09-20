import os
import streamlit as st
import dotenv
import uuid
import glob

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

from utils import stream_llm_response, load_doc_to_db, load_default_docs, stream_llm_rag_response

dotenv.load_dotenv()

MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
]

st.set_page_config(
    page_title="UrologIA - Asistente Especializado",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS para mejorar el styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    .info-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2a5298;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .warning-card {
        background: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .success-card {
        background: #d1e7dd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #198754;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .sidebar-section {
        padding: 1rem 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    
    .stButton > button {
        border-radius: 20px;
        border: none;
        background: linear-gradient(90deg, #2a5298 0%, #1e3c72 100%);
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .doc-counter {
        background: #2a5298;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
    <div class="main-header">
        <h1>🏥 UrologIA</h1>
        <p>Asistente Virtual Especializado en Urología y Cáncer de Próstata</p>
        <p style="font-size: 0.9rem; margin-top: 1rem;">
            💡 Información basada en evidencia científica y guías clínicas actualizadas
        </p>
    </div>
""", unsafe_allow_html=True)

# --- Initial Setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": """¡Hola! Soy **UrologIA**, tu asistente virtual especializado en urología y cáncer de próstata. 

🔬 **Estoy aquí para ayudarte con:**
- Información sobre cáncer de próstata
- Tratamientos y opciones terapéuticas
- Efectos secundarios y manejo
- Preguntas sobre procedimientos urológicos
- Seguimiento y cuidados post-tratamiento

⚠️ **Importante:** La información que proporciono es educativa y basada en evidencia científica, pero nunca debe reemplazar la consulta con tu médico especialista.

¿En qué puedo ayudarte hoy?"""
    }]

# if "docs_loaded" not in st.session_state:
#     st.session_state.docs_loaded = False
# openai_api_key = ""
# # Cargar documentos por defecto al inicio
# if not st.session_state.docs_loaded:
#     # Solo cargar si hay API key válida
#     if openai_api_key and "sk-" in openai_api_key:
#         with st.spinner("🔄 Cargando base de conocimientos médicos..."):
#             st.session_state.openai_api_key = openai_api_key  # Guardar en session state
#             load_default_docs()
#             st.session_state.docs_loaded = True

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🔐 Configuración de API")
    
    default_openai_api_key = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") else ""
    with st.expander("🔑 OpenAI API Key", expanded=not default_openai_api_key):
        openai_api_key = st.text_input(
            label="Introduce tu API Key de OpenAI",
            placeholder="sk-...",
            type="password",
            value=default_openai_api_key,
            help="Puedes obtener tu API key en https://platform.openai.com/account/api-keys",
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Información sobre documentos cargados
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📚 Base de Conocimientos")
    
    if st.session_state.rag_sources:
        st.markdown("**Documentos cargados:**")
        for i, source in enumerate(st.session_state.rag_sources[:5]):  # Mostrar solo los primeros 5
            source_name = source.split('/')[-1] if '/' in source else source
            if len(source_name) > 30:
                source_name = source_name[:27] + "..."
            st.markdown(f'<span class="doc-counter">{i+1}. {source_name}</span>', unsafe_allow_html=True)
        
        if len(st.session_state.rag_sources) > 5:
            st.markdown(f"... y {len(st.session_state.rag_sources) - 5} documentos más")
    else:
        st.info("No hay documentos adicionales cargados")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Cargar documentos adicionales
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📄 Cargar Documentos")
    
    uploaded_files = st.file_uploader(
        "Sube documentos médicos adicionales",
        type=['pdf', 'docx', 'txt', 'md'],
        accept_multiple_files=True,
        help="Formatos soportados: PDF, DOCX, TXT, MD"
    )
    
    if uploaded_files:
        st.session_state.rag_docs = uploaded_files
        if st.button("📥 Cargar Documentos", type="secondary"):
            with st.spinner("Procesando documentos..."):
                load_doc_to_db()
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Main Content ---
# Verificar si el usuario ha introducido la API key
missing_openai = openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key

if missing_openai:
    st.markdown("""
        <div class="warning-card">
            <h3>⚠️ API Key Requerida</h3>
            <p>Para utilizar UrologIA, necesitas introducir tu API Key de OpenAI en la barra lateral.</p>
            <p><strong>¿No tienes una API Key?</strong> Puedes obtenerla gratuitamente en 
            <a href="https://platform.openai.com/account/api-keys" target="_blank">OpenAI Platform</a></p>
        </div>
    """, unsafe_allow_html=True)
    st.stop()
else:  # <-- AQUÍ VA EL ELSE
    # Inicializar docs solo si hay API key válida
    st.session_state.openai_api_key = openai_api_key
    
    if "docs_loaded" not in st.session_state:
        st.session_state.docs_loaded = False
    
    if not st.session_state.docs_loaded:
        with st.spinner("🔄 Cargando base de conocimientos médicos..."):
            load_default_docs()
            st.session_state.docs_loaded = True
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🤖 Configuración del Modelo")
        
        selected_model = st.selectbox(
            "Selecciona el modelo de OpenAI",
            MODELS,
            index=5,
            help="GPT-4o es el modelo más avanzado y recomendado para consultas médicas complejas"
        )
        
        temperature = st.slider(
            "Creatividad del modelo",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Valores bajos (0.1) para respuestas más precisas y consistentes"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Controles de chat
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🔄 Controles de Chat")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Nuevo Chat", type="primary"):
                st.session_state.messages = [{
                    "role": "assistant",
                    "content": "¡Hola de nuevo! Soy UrologIA. ¿En qué puedo ayudarte en esta nueva consulta?"
                }]
                st.rerun()
        
        with col2:
            if st.button("📊 Estadísticas", type="secondary"):
                st.info(f"Mensajes en esta sesión: {len(st.session_state.messages)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Información importante sobre el uso
    with st.expander("ℹ️ Información Importante", expanded=False):
        st.markdown("""
        **🎯 Especialización:**
        - Este asistente está **exclusivamente** especializado en urología y cáncer de próstata
        - No proporciona información sobre otras especialidades médicas
        - Toda la información está basada en evidencia científica actual

        **⚠️ Limitaciones:**
        - **No** soy un sustituto de la consulta médica profesional
        - **No** puedo realizar diagnósticos ni prescribir tratamientos
        - **Siempre** consulta con tu urólogo para decisiones médicas importantes

        **🔒 Privacidad:**
        - Esta conversación es privada y segura
        - No almaceno información personal identificable
        - Los datos se procesan de forma temporal
        """)

    # Chat principal
    st.markdown("### 💬 Consulta con UrologIA")
    
    # Configurar el modelo LLM
    llm_stream = ChatOpenAI(
        model=selected_model,
        openai_api_key=openai_api_key,
        temperature=temperature,
        streaming=True
    )

    # Mostrar mensajes del chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input del usuario
    if prompt := st.chat_input("💬 Escribe tu consulta sobre urología o cáncer de próstata..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            messages = [
                HumanMessage(content=m["content"]) if m["role"] == "user" 
                else AIMessage(content=m["content"]) 
                for m in st.session_state.messages
            ]

            # Usar RAG si hay documentos cargados, sino usar respuesta normal
            if "vector_db" in st.session_state:
                st.write_stream(stream_llm_rag_response(llm_stream, messages))
            else:
                st.write_stream(stream_llm_response(llm_stream, messages))

# Footer
st.markdown("""
---
<div style="text-align: center; color: #666; font-size: 0.8rem; padding: 1rem;">
    <p>🏥 <strong>UrologIA</strong> - Asistente Virtual Especializado en Urología</p>
    <p>⚠️ <em>Esta herramienta es solo para fines educativos. Siempre consulta con un profesional médico.</em></p>
</div>
""", unsafe_allow_html=True)