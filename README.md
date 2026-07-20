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

:camera_with_flash: Demostración en Acción
Aquí puedes ver cómo el agente responde a las preguntas del usuario basándose exclusivamente en el documento proporcionado:

Politica puntual sobre politica de RRHH
<img width="893" height="408" alt="image" src="https://github.com/user-attachments/assets/eecf8bd0-3f4f-459a-a9ac-7ec04f9ff466" />

Pregunta ramdom
<img width="881" height="266" alt="image" src="https://github.com/user-attachments/assets/b1371fe4-4076-4404-9958-9a119c3acca8" />
