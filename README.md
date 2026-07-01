# Local PDF RAG Chatbot

A lightweight, fully functional Retrieval-Augmented Generation (RAG) pipeline built in Python. This project allows you to chat with any PDF document using local vector embeddings and a cloud LLM (OpenAI or Groq), with a strict emphasis on **zero hallucinations** through citation-based prompting.

## 🚀 Features

* **Local Vectorization:** Uses Hugging Face's `sentence-transformers` (`all-MiniLM-L6-v2`) to embed documents locally, ensuring your private PDFs are not sent to third-party embedding APIs.
* **In-Memory Database:** Utilizes `ChromaDB` for fast, local vector storage and semantic search directly on your machine.
* **Anti-Hallucination Prompting:** The LLM is strictly constrained to answer *only* using the provided context and must cite its sources (e.g., `[SOURCE 1]`). If the answer isn't in the PDF, it refuses to answer instead of guessing.
* **Plug-and-Play LLM:** Uses the standard `openai` Python SDK, making it trivial to swap between OpenAI (GPT-4o) and Groq (Llama-3) for free, fast inference.

## 🛠️ Architecture

The pipeline follows the classic 5-step RAG architecture:
1. **Extract:** Parses raw text from PDFs using `PyMuPDF`.
2. **Chunk:** Splits text into overlapping 500-character segments to preserve context without losing sentence meaning.
3. **Embed & Store:** Converts text chunks into 384-dimensional vectors and stores them in a local ChromaDB collection.
4. **Retrieve:** Converts the user's question into a vector and performs a similarity search to find the most relevant document chunks.
5. **Synthesize:** Injects the retrieved chunks into a strict prompt and generates a factual, cited response.

## 📦 Installation

**Prerequisites:** Python 3.8+

1. Clone this repository:
```bash
git clone [https://github.com/yourusername/local-pdf-rag.git](https://github.com/yourusername/local-pdf-rag.git)
cd local-pdf-rag
