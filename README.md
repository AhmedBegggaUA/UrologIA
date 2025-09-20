# 🏥 UrologIA - Asistente Virtual Especializado en Urología

## 📋 Descripción

UrologIA es un asistente de inteligencia artificial altamente especializado en urología y cáncer de próstata. Utiliza tecnología RAG (Retrieval Augmented Generation) para proporcionar información médica precisa basada en documentos científicos y guías clínicas.

## ✨ Características Principales

- **🎯 Especialización Exclusiva:** Solo responde consultas sobre urología y cáncer de próstata
- **📚 Base de Conocimientos:** Carga automática de documentos médicos de referencia
- **🤖 RAG Avanzado:** Respuestas basadas en documentos científicos actualizados
- **💬 Interfaz Amigable:** Diseño moderno y responsive con Streamlit
- **🔒 Seguridad:** Validación estricta de consultas médicas apropiadas
- **💚 Enfoque Empático:** Comunicación profesional con mensaje de esperanza

## 🚀 Instalación

### Prerrequisitos

- Python 3.8 o superior
- API Key de OpenAI

### Pasos de Instalación

1. **Clonar o descargar el proyecto:**
```bash
git clone <tu-repositorio>
cd urologia-chatbot
```

2. **Crear entorno virtual (recomendado):**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno:**
Crear archivo `.env` en la raíz del proyecto:
```
OPENAI_API_KEY=tu_api_key_aqui
```

5. **Crear estructura de carpetas:**
```bash
mkdir docs
mkdir temp_uploads
```

## 📁 Estructura del Proyecto

```
urologia-chatbot/
├── main.py                 # Aplicación principal
├── utils.py                # Funciones de utilidad y RAG
├── requirements.txt        # Dependencias
├── .env                   # Variables de entorno (crear)
├── docs/                  # Documentos médicos de referencia
├── temp_uploads/          # Archivos temporales
└── README.md             # Este archivo
```

## 🔧 Configuración

### Documentos de Referencia

1. **Coloca tus documentos médicos en la carpeta `docs/`:**
   - Formatos soportados: PDF, DOCX, TXT, MD
   - Ejemplo: guías clínicas, protocolos, artículos científicos

2. **Los documentos se cargan automáticamente al iniciar la aplicación**

### API de OpenAI

1. **Obtén tu API Key en:** https://platform.openai.com/account/api-keys
2. **Configúrala en el archivo `.env` o directamente en la aplicación**

## 🎮 Uso

### Ejecutar la Aplicación

```bash
streamlit run main.py
```

### Funcionalidades

#### 🤖 **Chat Especializado**
- Realiza consultas sobre urología y cáncer de próstata
- Recibe respuestas basadas en evidencia científica
- Información empática y profesional

#### 📚 **Gestión de Documentos**
- Carga automática de documentos de la carpeta `docs/`
- Subida de documentos adicionales desde la interfaz
- Procesamiento automático con RAG

#### ⚙️ **Configuración Avanzada**
- Selección de modelos OpenAI (GPT-4, GPT-3.5)
- Control de temperatura del modelo
- Estadísticas de uso

## 🔒 Restricciones de Seguridad

### ✅ **Lo que SÍ hace:**
- Responde consultas sobre urología y cáncer de próstata
- Proporciona información educativa basada en evidencia
- Ofrece apoyo emocional y esperanza realista
- Recomienda consultar con especialistas

### ❌ **Lo que NO hace:**
- No responde consultas de otras especialidades médicas
- No realiza diagnósticos definitivos
- No prescribe medicamentos
- No da pronósticos específicos de supervivencia
- No proporciona información alarmante sin fundamento

## 🎨 Personalización

### Modificar el Prompt

Edita la variable `system_prompt` en `utils.py` para ajustar:
- Tono de comunicación
- Áreas de especialización
- Restricciones adicionales
- Mensajes de esperanza

### Styling de la Interfaz

Modifica el CSS en `main.py` para cambiar:
- Colores y gradientes
- Tipografía
- Layouts y espaciado
- Efectos visuales

## 📊 Monitoreo

### Estadísticas Disponibles
- Número de documentos cargados
- Chunks procesados en la base vectorial
- Mensajes por sesión
- Estado del sistema RAG

## 🐛 Solución de Problemas

### Errores Comunes

1. **Error de API Key:**
   - Verifica que la API key sea válida
   - Revisa el formato (debe empezar con 'sk-')

2. **Error cargando documentos:**
   - Verifica que los archivos estén en formato soportado
   - Revisa permisos de lectura en la carpeta `docs/`

3. **Error de memoria:**
   - Reduce el número de documentos
   - Usa chunks más pequeños en `utils.py`

### Logs de Depuración

Para obtener más información sobre errores, ejecuta:
```bash
streamlit run main.py --logger.level=debug
```

## 🤝 Contribuciones

### Para Desarrolladores

1. **Fork el repositorio**
2. **Crea una rama para tu feature:** `git checkout -b feature/nueva-funcionalidad`
3. **Commit tus cambios:** `git commit -m 'Agregar nueva funcionalidad'`
4. **Push a la rama:** `git push origin feature/nueva-funcionalidad`
5. **Abre un Pull Request**

### Para Médicos/Especialistas

1. **Proporciona documentos médicos actualizados**
2. **Revisa la precisión de las respuestas**
3. **Sugiere mejoras en el prompt médico**
4. **Reporta inconsistencias o errores**

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para detalles.

## ⚠️ Disclaimer Médico

**IMPORTANTE:** Este asistente es solo para fines educativos e informativos. No debe ser usado como sustituto de la consulta médica profesional. Siempre consulta con un médico especialista para decisiones de salud importantes.

## 📞 Soporte

Para soporte técnico o preguntas:
- Crear un issue en el repositorio
- Contactar al equipo de desarrollo
- Revisar la documentación de troubleshooting

---

**🏥 UrologIA - Democratizando el acceso a información médica especializada**