# 🤖 Agente de Inteligencia Artificial para Documentos Internos (RAG)

Este proyecto consiste en un agente corporativo inteligente capaz de leer, procesar y responder preguntas en lenguaje natural basadas en la documentación interna de la empresa (manuales, políticas, PDFs), optimizando el tiempo de búsqueda de información.

## 🏗️ Arquitectura del Sistema

El flujo de datos del agente sigue la arquitectura de Generación Aumentada por Recuperación (RAG):

1. **Ingesta de Datos:** Extracción de texto desde archivos PDF corporativos mediante `PyPDFLoader`.
2. **Procesamiento de Texto:** Segmentación del contenido en fragmentos lógicos (chunks) de 1000 caracteres con solapamiento (`RecursiveCharacterTextSplitter`).
3. **Generación de Embeddings:** Conversión de texto a vectores densos mediante el modelo multilingüe de Cohere (`embed-multilingual-v3.0`).
4. **Almacenamiento Vectorial:** Indexación y persistencia de fragmentos vectorizados en una base de datos local `Chroma`.
5. **Orquestación RAG:** Recuperación contextual inteligente combinada con el modelo de lenguaje avanzado `command-r7b-12-2024` de Cohere para estructurar respuestas precisas sin alucinaciones.

## 🚀 Instrucciones de Ejecución (Local)

### Requisitos Previos
* Python 3.10 o superior instalado.
* Una cuenta en Cohere y una API Key activa.

### Instalación
1. Clona este repositorio.
2. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
