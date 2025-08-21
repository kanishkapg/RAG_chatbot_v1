# 🚀 RAG Chatbot v1

This repository contains a Retrieval-Augmented Generation (RAG) chatbot designed to answer user queries about official documents, specifically company or institutional circulars. The system processes PDF documents, extracts text and metadata, and uses a vector database to retrieve relevant information for generating accurate, context-aware answers.

## 🏗️ System Architecture

The system follows a multi-stage pipeline to ingest, process, and query information from PDF documents:

1.  **PDF Ingestion & OCR:** PDF files placed in the `data` directory are processed. `pdf2image` converts each page into an image, and `pytesseract` performs Optical Character Recognition (OCR) to extract the raw text.
2.  **Data Persistence:** The extracted text and a hash of the source file are stored in a PostgreSQL database. This prevents reprocessing of unchanged documents.
3.  **Metadata Extraction:** A Groq-powered LLM analyzes the raw text to identify and extract key metadata such as circular numbers, titles, effective dates, and references to repealed documents. This metadata is also stored in PostgreSQL.
4.  **Chunking & Embedding:** The full document text is split into smaller, semantically coherent chunks using LangChain's text splitters. Each chunk is then converted into a vector embedding using the `BAAI/bge-m3` sentence transformer model.
5.  **Vector Indexing:** The text chunks and their corresponding vector embeddings are upserted into a **Qdrant** vector database for efficient similarity search.
6.  **Retrieval & Generation:** When a user asks a question, the system embeds the query and searches Qdrant for the most relevant document chunks. This retrieved context, along with document metadata, is passed to a Groq LLM (`llama-3.1-8b-instant`) which synthesizes a final answer, citing the source circulars.
7.  **User Interface:** A simple web interface is provided using **Streamlit**.

## 🛠️ Key Technologies

- **LLM:** Groq (for both metadata extraction and final response generation)
- **Vector Database:** Qdrant
- **Data Store:** PostgreSQL
- **Embedding Model:** `sentence-transformers` (BAAI/bge-m3)
- **OCR:** `pytesseract` & `pdf2image`
- **Core Frameworks:** LangChain
- **Web UI:** Streamlit

## ⚙️ Setup & Installation

### 📋 Prerequisites

- Python 3.10+
- A running **PostgreSQL** instance.
- A running **Qdrant** instance.
- Tesseract OCR installed on your system.

### 🔧 Installation Steps

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/kanishkapg/RAG_chatbot_v1.git
    cd RAG_chatbot_v1
    ```

2.  **Create an Environment File**
    Create a file named `.env` in the root of the project and add your credentials. The application uses these variables to connect to the required services.

    ```env
    GROQ_API_KEY="your_groq_api_key"
    QDRANT_URL="http://localhost:6333"
    QDRANT_API_KEY="your_qdrant_api_key_if_any"

    # These can be overridden by environment variables
    # Default values are in config.py
    POSTGRES_DB="slt_circulars_db"
    POSTGRES_USER="postgres"
    POSTGRES_PASSWORD="your_postgres_password"
    POSTGRES_HOST="localhost"
    POSTGRES_PORT="5432"
    ```

3.  **Install Dependencies**
    Install the required Python packages using `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
    Alternatively, if you use Conda, you can create an environment from `environment.yml`:
    ```bash
    conda env create -f environment.yml
    conda activate base
    ```

## ▶️ Usage

1.  **Add Documents**
    Place the PDF circulars you want to process into the `data/original/` directory. You can use the `data/dummy/` directory for testing.

2.  **Configure Data Source**
    In `config.py`, ensure the `DATA_DIR` variable points to the directory containing your PDFs (e.g., `./data/original`).

3.  **Run the Ingestion Pipeline**
    Execute the main processing script. This will perform OCR, extract metadata, create embeddings, and populate both the PostgreSQL and Qdrant databases. The script will automatically create the required tables in PostgreSQL if they don't exist.

    ```bash
    python test.py
    ```

4.  **Start the Chatbot**
    Launch the Streamlit web application to start interacting with your documents.
    ```bash
    streamlit run app.py
    ```
