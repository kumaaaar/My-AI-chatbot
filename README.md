# AI Chatbot with RAG

A beginner-friendly AI chatbot built using Python, Streamlit and Ollama, with support for document-based question answering using Retrieval-Augmented Generation (RAG).

## Features

- Chat with a local language model.
- Upload PDF and TXT documents for document-based question answering.
- Extract and split document text into chunks.
- Generate embeddings using `nomic-embed-text`.
- Retrieve relevant chunks using cosine similarity.
- Generate answers using retrieved document context.
- Maintain chat history locally.

## Tech Stack

- **Python** — application logic
- **Streamlit** — user interface
- **Ollama** — run language models locally
- **llama3.2:1b** — language model
- **nomic-embed-text** — embedding model
- **RAG** — retrieve relevant document content to help answer questions

## Prerequisites

- Python installed
- Ollama installed and running
- The required Ollama models downloaded
- Git installed

## Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/kumaaaar/My-AI-chatbot
   cd my-ai-chatbot
   ```

2. Download the required models:

   ```bash
   ollama pull llama3.2:1b
   ollama pull nomic-embed-text
   ```

3. Install the Python dependencies used by the project.

   If a `requirements.txt` file is available, run:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the application:

   ```bash
   python -m streamlit run app.py
   ```

5. Open the local URL displayed in the terminal.

## How RAG Works

1. Upload a supported document.
2. Extract its text and split it into chunks.
3. Generate embeddings for the chunks.
4. Compare the user's question with the document chunks using cosine similarity.
5. Retrieve relevant chunks.
6. Provide the retrieved context to the language model to generate an answer.

## Project Status

Project built for learning Python, local LLM applications, Streamlit and RAG fundamentals.

## Future Improvements

- Automated tests
- Improved retrieval evaluation
- More robust error handling
- FastAPI backend
- Deployment options

## Author

https://github.com/kumaaaar