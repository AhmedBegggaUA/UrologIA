import os
import glob
import dotenv
from time import time
import streamlit as st

from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
    PyMuPDFLoader,
)
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

dotenv.load_dotenv()

os.environ["USER_AGENT"] = "UrologIA_Agent"
DB_DOCS_LIMIT = 50

# Función para stream de respuestas del LLM
def stream_llm_response(llm_stream, messages):
    """Stream respuestas estándar del LLM con prompt especializado"""
    
    # Sistema prompt altamente especializado en urología
    system_prompt = """
    Eres UrologIA, un asistente de inteligencia artificial altamente especializado en urología y específicamente en cáncer de próstata. Tu expertise abarca todos los aspectos de esta especialidad médica.

    ## IDENTIDAD Y ESPECIALIZACIÓN:
    - Eres un asistente médico virtual especializado EXCLUSIVAMENTE en urología
    - Tu conocimiento se enfoca principalmente en cáncer de próstata, pero también cubres toda la urología
    - Tienes acceso a la literatura médica más reciente y guías clínicas actualizadas
    - Tu comunicación es profesional, empática y siempre basada en evidencia científica

    ## ÁREAS DE CONOCIMIENTO ESPECÍFICO:
    
    ### CÁNCER DE PRÓSTATA:
    - Epidemiología, factores de riesgo y prevención
    - Síntomas y signos clínicos tempranos y tardíos
    - Métodos de detección: PSA, tacto rectal, resonancia magnética multiparamétrica
    - Técnicas de biopsia: transrectal, transperineal, biopsia de fusión
    - Sistemas de clasificación: Gleason, ISUP, TNM, D'Amico
    - Estadificación y estratificación de riesgo
    - Tratamientos curativos: prostatectomía radical, radioterapia externa, braquiterapia
    - Tratamientos paliativos y para enfermedad avanzada
    - Terapia hormonal: análogos de LHRH, antiandrógenos, nuevos agentes
    - Quimioterapia: docetaxel, cabazitaxel
    - Nuevas terapias: enzalutamida, abiraterona, sipuleucel-T
    - Efectos secundarios y manejo: incontinencia, disfunción eréctil, fatiga
    - Seguimiento post-tratamiento y recidiva bioquímica
    - Aspectos psicológicos y calidad de vida

    ### UROLOGÍA GENERAL:
    - Infecciones del tracto urinario
    - Litiasis renal y ureteral
    - Hiperplasia benigna de próstata
    - Disfunción eréctil
    - Incontinencia urinaria
    - Cáncer renal, vesical y testicular
    - Malformaciones urogenitales
    - Trauma urológico
    - Andrología e infertilidad masculina

    ## RESTRICCIONES ESTRICTAS:
    
    ### LO QUE NO PUEDES HACER:
    - NO proporcionar información sobre especialidades médicas distintas a la urología
    - NO responder preguntas sobre cardiología, dermatología, neurología, etc.
    - NO hablar de política, deportes, entretenimiento o temas no médicos
    - NO realizar diagnósticos definitivos
    - NO prescribir medicamentos específicos
    - NO dar fechas exactas de supervivencia o pronósticos específicos
    - NO crear pánico o desesperanza en el paciente
    - NO inventar información médica
    - NO contradecir las indicaciones del médico tratante del paciente

    ### LO QUE SÍ DEBES HACER:
    - Responder SOLO preguntas relacionadas con urología
    - Proporcionar información educativa basada en evidencia
    - Ser empático y ofrecer esperanza realista
    - Enfatizar que la medicina avanza constantemente
    - Recalcar la importancia del seguimiento médico profesional
    - Usar lenguaje claro y comprensible para el paciente
    - Mencionar opciones de tratamiento disponibles
    - Hablar sobre la importancia del apoyo familiar y psicológico

    ## PROTOCOLO DE RESPUESTA:

    ### SI LA PREGUNTA ES SOBRE UROLOGÍA:
    1. Proporciona información precisa y actualizada
    2. Explica de manera comprensible evitando jerga médica excesiva
    3. Ofrece esperanza mencionando los avances médicos
    4. Recomienda consultar con el médico especialista
    5. Si es apropiado, menciona recursos de apoyo

    ### SI LA PREGUNTA NO ES SOBRE UROLOGÍA:
    "Lo siento, soy un asistente especializado exclusivamente en urología y cáncer de próstata. No puedo ayudarte con consultas sobre [tema mencionado]. Para esas consultas, te recomiendo consultar con el especialista correspondiente o un médico de familia. ¿Hay alguna pregunta sobre urología en la que sí pueda ayudarte?"

    ### SI LA PREGUNTA ES MÉDICA PERO NO UROLÓGICA:
    "Aunque tu consulta es médica, mi especialización se limita estrictamente a urología y cáncer de próstata. Para consultas sobre [especialidad], te recomiendo acudir a un [especialista correspondiente] o tu médico de familia. ¿Tienes alguna pregunta relacionada con urología que pueda resolver?"

    ## TONO Y COMUNICACIÓN:
    - Siempre empático y profesional
    - Optimista pero realista
    - Claro y educativo
    - Respetuoso con las preocupaciones del paciente
    - Enfocado en dar esperanza y tranquilidad

    ## MENSAJES DE ESPERANZA A INCLUIR:
    - "La medicina urológica ha avanzado significativamente en los últimos años"
    - "Existen múltiples opciones de tratamiento disponibles"
    - "La mayoría de pacientes con cáncer de próstata tienen un pronóstico favorable"
    - "El diagnóstico temprano permite mejores resultados de tratamiento"
    - "Los efectos secundarios pueden ser manejados efectivamente"
    - "Es importante mantener una comunicación abierta con tu equipo médico"

    Recuerda: SOLO respondes sobre urología. Cualquier otro tema debe ser redirigido cortésmente.
    """

    # Crear mensajes con el sistema prompt
    full_messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m.type, "content": m.content} for m in messages
    ]

    response_message = ""
    for chunk in llm_stream.stream(full_messages):
        response_message += chunk.content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})


# --- FUNCIONES DE CARGA DE DOCUMENTOS ---

def load_default_docs():
    """Carga automáticamente todos los documentos de la carpeta docs/"""
    docs_folder = "docs"
    
    if not os.path.exists(docs_folder):
        os.makedirs(docs_folder, exist_ok=True)
        st.warning(f"📁 Carpeta '{docs_folder}' creada. Coloca aquí tus documentos médicos de referencia.")
        return
    
    # Buscar todos los archivos soportados en la carpeta docs
    supported_extensions = ['*.pdf', '*.docx', '*.txt', '*.md']
    docs_to_load = []
    
    for extension in supported_extensions:
        docs_to_load.extend(glob.glob(os.path.join(docs_folder, extension)))
    
    if not docs_to_load:
        st.info("📚 No se encontraron documentos en la carpeta 'docs'. Puedes agregar documentos PDF, DOCX, TXT o MD.")
        return
    
    docs = []
    loaded_count = 0
    
    for file_path in docs_to_load:
        file_name = os.path.basename(file_path)
        
        if file_name not in st.session_state.rag_sources:
            try:
                if file_path.endswith('.pdf'):
                    loader = PyMuPDFLoader(file_path)
                elif file_path.endswith('.docx'):
                    loader = Docx2txtLoader(file_path)
                elif file_path.endswith(('.txt', '.md')):
                    loader = TextLoader(file_path, encoding='utf-8')
                else:
                    continue
                
                file_docs = loader.load()
                docs.extend(file_docs)
                st.session_state.rag_sources.append(file_name)
                loaded_count += 1
                
            except Exception as e:
                st.error(f"❌ Error cargando {file_name}: {str(e)}")
    
    if docs:
        _split_and_load_docs(docs)
        if loaded_count > 0:
            st.success(f"✅ Base de conocimientos cargada: {loaded_count} documentos procesados exitosamente")


def load_doc_to_db():
    """Carga documentos adicionales subidos por el usuario"""
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = []
        loaded_count = 0
        
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    # Crear directorio temporal si no existe
                    os.makedirs("temp_uploads", exist_ok=True)
                    file_path = f"./temp_uploads/{doc_file.name}"
                    
                    try:
                        # Guardar archivo temporalmente
                        with open(file_path, "wb") as file:
                            file.write(doc_file.read())

                        # Cargar según el tipo de archivo
                        if doc_file.type == "application/pdf":
                            loader = PyMuPDFLoader(file_path)
                        elif doc_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        elif doc_file.type in ["text/plain", "text/markdown"]:
                            loader = TextLoader(file_path, encoding='utf-8')
                        else:
                            st.warning(f"⚠️ Tipo de documento {doc_file.type} no soportado.")
                            continue

                        file_docs = loader.load()
                        docs.extend(file_docs)
                        st.session_state.rag_sources.append(doc_file.name)
                        loaded_count += 1

                    except Exception as e:
                        st.error(f"❌ Error procesando {doc_file.name}: {str(e)}")
                    
                    finally:
                        # Limpiar archivo temporal
                        if os.path.exists(file_path):
                            os.remove(file_path)
                else:
                    st.error(f"❌ Límite máximo de documentos alcanzado ({DB_DOCS_LIMIT}).")
                    break

        if docs and loaded_count > 0:
            _split_and_load_docs(docs)
            st.success(f"✅ {loaded_count} documento(s) adicional(es) procesados exitosamente")


def initialize_vector_db(docs):
    """Inicializa la base de datos vectorial"""
    
    # Validar documentos
    if not docs or len(docs) == 0:
        st.error("❌ No hay documentos para procesar")
        return None
    
    try:
        # Verificar configuración de embeddings
        if "AZ_OPENAI_API_KEY" in os.environ:
            st.info("🔧 Usando Azure OpenAI para embeddings")
            embedding = AzureOpenAIEmbeddings(
                api_key=os.getenv("AZ_OPENAI_API_KEY"), 
                azure_endpoint=os.getenv("AZ_OPENAI_ENDPOINT"),
                model="text-embedding-3-large",
                openai_api_version="2024-02-15-preview",
            )
        else:
            # Usar la API key de OpenAI del estado de la sesión o variable de entorno
            api_key = st.session_state.get('openai_api_key') or os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                st.error("❌ No se encontró API key de OpenAI. Configura OPENAI_API_KEY o ingrésala en la interfaz.")
                return None
                
            st.info("🔧 Usando OpenAI para embeddings")
            embedding = OpenAIEmbeddings(
                api_key=api_key,
                model="text-embedding-3-small"
            )

        # Crear progreso para el usuario
        with st.spinner("🔄 Creando base de datos vectorial..."):
            vector_db = Chroma.from_documents(
                documents=docs,
                embedding=embedding,
                collection_name=f"urologia_{str(time()).replace('.', '')[:14]}_{st.session_state['session_id'][:8]}",
            )

        # Verificar que se creó correctamente
        if vector_db is None:
            st.error("❌ Error: La base de datos vectorial se creó como None")
            return None

        st.success(f"✅ Base de datos vectorial creada con {len(docs)} documentos")

        # Gestión de colecciones (máximo 30 para evitar problemas de memoria)
        try:
            chroma_client = vector_db._client
            collection_names = sorted([collection.name for collection in chroma_client.list_collections()])
            
            if len(collection_names) > 30:
                st.info(f"🧹 Limpiando colecciones antiguas ({len(collection_names)} > 30)")
                while len(collection_names) > 30:
                    chroma_client.delete_collection(collection_names[0])
                    collection_names.pop(0)
                    
        except Exception as e:
            st.warning(f"⚠️ Advertencia en gestión de colecciones: {str(e)}")

        return vector_db
    
    except Exception as e:
        st.error(f"❌ Error inicializando base de datos vectorial: {str(e)}")
        st.error(f"Tipo de error: {type(e).__name__}")
        
        # Información adicional para debugging
        if "api_key" in str(e).lower():
            st.error("🔑 Problema con API key. Verifica tu configuración.")
        elif "embedding" in str(e).lower():
            st.error("📊 Problema con el modelo de embeddings.")
        elif "chroma" in str(e).lower():
            st.error("💾 Problema con la base de datos Chroma.")
            
        return None


def _split_and_load_docs(docs):
    """Divide los documentos en chunks y los carga en la base vectorial"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,      # Chunks más pequeños para mejor precisión
        chunk_overlap=800,    # Mayor overlap para mejor contexto
        separators=["\n\n", "\n", ". ", ".", " ", ""],
        length_function=len,
    )

    document_chunks = text_splitter.split_documents(docs)

    if "vector_db" not in st.session_state:
        st.session_state.vector_db = initialize_vector_db(document_chunks)
    else:
        try:
            st.session_state.vector_db.add_documents(document_chunks)
        except Exception as e:
            st.error(f"❌ Error agregando documentos a la base vectorial: {str(e)}")


# --- FUNCIONES RAG (Retrieval Augmented Generation) ---

def _get_context_retriever_chain(vector_db, llm):
    """Crea la cadena de recuperación de contexto"""
    retriever = vector_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 8,              # Número de documentos a recuperar
            "score_threshold": 0.25  # Umbral de similitud
        }
    )
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", """Basándote en la conversación anterior, genera una consulta de búsqueda precisa para encontrar información médica relevante en la base de documentos de urología. 
        
        INSTRUCCIONES:
        - Si la pregunta es sobre cáncer de próstata, incluye términos como: "próstata", "PSA", "Gleason", "tratamiento", "radioterapia", "prostatectomía"
        - Si es sobre otros temas urológicos, usa términos específicos de la condición
        - Si la pregunta no es clara, reformula para buscar información general de urología
        - Mantén la consulta en español y enfócate en términos médicos precisos
        
        Genera solo la consulta de búsqueda, sin explicaciones adicionales."""),
    ])
    
    retriever_chain = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=prompt
    )
    return retriever_chain


def get_conversational_rag_chain(vector_db, llm):
    """Crea la cadena RAG conversacional"""
    retriever_chain = _get_context_retriever_chain(vector_db, llm)

    # Prompt system altamente especializado para RAG
    system_prompt = """
    Eres UrologIA, un asistente de inteligencia artificial altamente especializado en urología y cáncer de próstata. Tienes acceso a una base de conocimientos médicos especializada y debes usar esta información para proporcionar respuestas precisas y actualizadas.

    ## IDENTIDAD Y MISIÓN:
    - Asistente médico virtual EXCLUSIVAMENTE especializado en urología
    - Experto principal en cáncer de próstata y todas las subespecialidades urológicas
    - Comunicación empática, profesional y basada en evidencia científica sólida
    - Promotor de esperanza realista y adherencia al tratamiento médico

    ## USO DE LA DOCUMENTACIÓN:
    - SIEMPRE prioriza la información de los documentos médicos proporcionados
    - Combina el conocimiento de los documentos con tu expertise en urología
    - Si hay conflictos entre fuentes, menciona las diferentes perspectivas
    - NUNCA inventes información que no esté en los documentos o tu conocimiento base
    - Cita de manera natural cuando uses información específica de los documentos

    ## ESPECIALIZACIÓN ESTRICTA EN UROLOGÍA:

    ### CÁNCER DE PRÓSTATA (Área principal):
    - Epidemiología, genética y factores de riesgo
    - Screening: PSA, PSA libre, PHI, 4Kscore, SelectMDx
    - Imagen: ecografía transrectal, RM multiparamétrica, PET-PSMA
    - Biopsia: técnicas, sistemas de clasificación (Gleason/ISUP)
    - Estadificación TNM y estratificación de riesgo (D'Amico, NCCN, CAPRA)
    - Vigilancia activa: criterios, seguimiento, ansiedad del paciente
    - Cirugía: prostatectomía radical (abierta, laparoscópica, robótica)
    - Radioterapia: externa (IMRT, VMAT), braquiterapia, radiocirugía
    - Terapia hormonal: castración, resistencia, secuencial vs. combinada
    - Enfermedad metastásica: quimioterapia, nuevos antihormonales
    - Efectos adversos: incontinencia, disfunción eréctil, toxicidad intestinal
    - Recidiva bioquímica: definición, manejo, terapias de rescate
    - Soporte psicológico y calidad de vida

    ### UROLOGÍA GENERAL:
    - Hiperplasia benigna de próstata: médico, quirúrgico, nuevas tecnologías
    - Infecciones urinarias: cistitis, pielonefritis, prostatitis
    - Litiasis: composición, tratamiento conservador vs. intervencionista
    - Cáncer renal: localizados, avanzados, terapias dirigidas
    - Cáncer vesical: superficial, músculo-invasivo, inmunoterapia
    - Disfunción eréctil: orgánica vs. psicogénica, tratamientos
    - Incontinencia: tipos, evaluación, manejo conservador y quirúrgico
    - Andrología: infertilidad, hipogonadismo, Peyronie

    ## PROTOCOLO DE RESPUESTA CON RAG:

    ### CUANDO HAY INFORMACIÓN RELEVANTE EN LOS DOCUMENTOS:
    1. Usar la información de los documentos como base principal
    2. Complementar con conocimiento médico establecido
    3. Proporcionar contexto clínico adicional cuando sea apropiado
    4. Explicar implicaciones prácticas para el paciente
    5. Ofrecer perspectiva esperanzadora basada en evidencia

    ### CUANDO NO HAY INFORMACIÓN SUFICIENTE EN LOS DOCUMENTOS:
    1. Usar tu conocimiento base de urología
    2. Ser transparente sobre las limitaciones de la información disponible
    3. Sugerir consultar con el médico especialista para información específica
    4. Proporcionar información general educativa cuando sea apropiado

    ### RESTRICCIONES ABSOLUTAS:
    - NO responder consultas no urológicas (rechazar cortésmente)
    - NO hacer diagnósticos definitivos
    - NO prescribir medicamentos específicos
    - NO dar pronósticos de supervivencia específicos
    - NO crear alarma innecesaria
    - NO contradecir indicaciones del médico tratante

    ## ESTILO DE COMUNICACIÓN:

    ### TONO EMPÁTICO Y PROFESIONAL:
    - Reconocer las preocupaciones emocionales del paciente
    - Usar lenguaje comprensible evitando jerga médica excesiva
    - Proporcionar explicaciones claras y estructuradas
    - Transmitir confianza en los avances médicos actuales

    ### MENSAJES DE ESPERANZA CONSTANTES:
    - "Los tratamientos actuales para el cáncer de próstata son muy efectivos"
    - "La detección temprana permite excelentes resultados de tratamiento"
    - "Existe una amplia gama de opciones terapéuticas disponibles"
    - "Los efectos secundarios pueden manejarse efectivamente"
    - "La investigación en urología avanza constantemente"
    - "La mayoría de hombres con cáncer de próstata viven vidas plenas"

    ### ESTRUCTURA DE RESPUESTA IDEAL:
    1. **Reconocimiento:** Validar la pregunta/preocupación
    2. **Información:** Datos precisos basados en documentos y evidencia
    3. **Contexto:** Explicación práctica y relevancia clínica
    4. **Esperanza:** Perspectiva positiva realista
    5. **Recomendación:** Sugerencia de seguimiento médico apropiado

    ## CASOS ESPECIALES:

    ### SI LA CONSULTA NO ES SOBRE UROLOGÍA:
    "Comprendo tu preocupación, pero mi especialización se limita estrictamente a urología y cáncer de próstata. Para consultas sobre [área mencionada], te recomiendo consultar con un especialista en [especialidad correspondiente] o tu médico de familia. ¿Hay alguna pregunta relacionada con urología en la que sí pueda ayudarte?"

    ### SI HAY ANSIEDAD O MIEDO EVIDENTE:
    - Reconocer y validar las emociones
    - Proporcionar información tranquilizadora basada en hechos
    - Enfatizar los avances médicos y opciones de tratamiento
    - Sugerir recursos de apoyo psicológico cuando sea apropiado
    - Recalcar la importancia del equipo médico multidisciplinario

    Recuerda: Tu misión es educar, tranquilizar y empoderar a los pacientes con información precisa y esperanza realista, siempre dentro del ámbito estricto de la urología.

    INFORMACIÓN DE LOS DOCUMENTOS MÉDICOS:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )
    
    return create_retrieval_chain(
        retriever_chain,
        document_chain,
    )


def stream_llm_rag_response(llm_stream, messages):
    """Stream respuestas RAG del LLM usando documentos médicos"""
    if "vector_db" not in st.session_state:
        # Si no hay base vectorial, usar respuesta estándar
        return stream_llm_response(llm_stream, messages)
    
    try:
        conversation_rag_chain = get_conversational_rag_chain(st.session_state.vector_db, llm_stream)
        
        response_message = ""
        
        # Preparar mensajes para el chain
        formatted_messages = []
        for msg in messages[:-1]:  # Todos excepto el último
            if hasattr(msg, 'type'):
                formatted_messages.append({"role": msg.type, "content": msg.content})
            else:
                formatted_messages.append(msg)
        
        # El último mensaje es el input del usuario
        user_input = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
        
        # Stream de la respuesta RAG
        for chunk in conversation_rag_chain.pick("answer").stream({
            "messages": formatted_messages, 
            "input": user_input
        }):
            response_message += chunk
            yield chunk

        # Agregar prefijo para indicar que es respuesta RAG
        full_response = f"📚 *Respuesta basada en documentos médicos*\n\n{response_message}"
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
    except Exception as e:
        st.error(f"❌ Error en respuesta RAG: {str(e)}")
        # Fallback a respuesta estándar
        return stream_llm_response(llm_stream, messages)


# --- FUNCIONES DE UTILIDAD ADICIONALES ---

def clear_vector_db():
    """Limpia la base de datos vectorial"""
    if "vector_db" in st.session_state:
        try:
            # Intentar eliminar la colección actual
            collection_name = st.session_state.vector_db._collection.name
            st.session_state.vector_db._client.delete_collection(collection_name)
            del st.session_state.vector_db
            st.session_state.rag_sources.clear()
            st.success("✅ Base de conocimientos reiniciada exitosamente")
        except Exception as e:
            st.error(f"❌ Error reiniciando base de conocimientos: {str(e)}")


def get_rag_stats():
    """Obtiene estadísticas de la base RAG"""
    if "vector_db" in st.session_state:
        try:
            collection = st.session_state.vector_db._collection
            count = collection.count()
            return {
                "documentos_cargados": len(st.session_state.rag_sources),
                "chunks_procesados": count,
                "estado": "✅ Activa"
            }
        except:
            return {
                "documentos_cargados": len(st.session_state.rag_sources),
                "chunks_procesados": "No disponible",
                "estado": "⚠️ Error"
            }
    else:
        return {
            "documentos_cargados": 0,
            "chunks_procesados": 0,
            "estado": "❌ No inicializada"
        }


def validate_medical_query(query):
    """Valida si la consulta es relacionada con urología"""
    urology_keywords = [
        # Cáncer de próstata
        'prostata', 'próstata', 'psa', 'gleason', 'prostatectomia', 'prostatectomía',
        'radioterapia', 'braquiterapia', 'hormonal', 'antiandrógeno', 'antiandrogeno',
        
        # Urología general
        'urolog', 'urin', 'riñon', 'riñón', 'vejiga', 'uretra', 'uréter',
        'litiasis', 'cálculo', 'calculo', 'piedra', 'infección', 'infeccion',
        'cistitis', 'pielonefritis', 'prostatitis',
        
        # Síntomas urológicos
        'incontinencia', 'disfuncion', 'disfunción', 'erectil', 'eréctil',
        'orina', 'micción', 'miccion', 'sangre', 'hematuria',
        
        # Anatomía urológica
        'testiculo', 'testículo', 'escroto', 'pene', 'uretra',
        'androlog', 'fertilidad', 'esperma'
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in urology_keywords)


# Función de limpieza al finalizar la sesión
def cleanup_temp_files():
    """Limpia archivos temporales"""
    temp_dirs = ["temp_uploads", "source_files"]
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass