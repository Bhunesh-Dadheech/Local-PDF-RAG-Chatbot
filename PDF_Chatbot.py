import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI

# 1. Extract: Read PDF and extract raw text

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# 2. Chunk: Split text into 500-character chunks with 50-char overlap

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks


# 3. Embed & Store: Convert to vectors and save to ChromaDB

def setup_vector_database(chunks, db_path="./my_local_chromadb"):
    # Initialize the local persistent vector database
    client = chromadb.PersistentClient(path=db_path)
    
    # Create or load a collection (think of it as a table)
    collection = client.get_or_create_collection(name="pdf_chatbot")
    
    # Load the open-source embedding model
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings for all chunks
    print("Embedding chunks...")
    embeddings = model.encode(chunks).tolist()
    
    # Generate unique IDs for each chunk
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    # Store chunks, their vectors, and IDs in the database
    collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )
    print("Database populated successfully.")
    return collection, model

# 4. Query: Search the database for the most similar chunks

def retrieve_similar_chunks(question, collection, model, top_k=3):
    # Convert the question into a vector using the SAME model
    question_embedding = model.encode([question]).tolist()
    
    # Search the database for the closest semantic matches
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )
    
    # Return the raw text of the best matching chunks
    return results['documents'][0]


# 5. Synthesize: Pass context and question to the LLM
def generate_answer(question, context_chunks, api_key):
    # (Use your OpenAI or Groq client setup here)
    client = OpenAI(api_key=api_key) 
    
    # 1. Number the chunks so the AI has a reference point
    context_text = ""
    for i, chunk in enumerate(context_chunks):
        # We label them as SOURCE 1, SOURCE 2, etc.
        context_text += f"\n\n--- SOURCE {i+1} ---\n{chunk}\n"
    
    # 2. The Strict Citation Prompt
    prompt = f"""
    Use the provided context to answer the question. 
    If the answer is not in the context, say you don't know.

    Sources:
    {context_text}

    Question:
    {question}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini", # or your Groq model
        messages=[
            {"role": "system", "content": "You are a precise, data-driven assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

# Main Execution Pipeline

if __name__ == "__main__":
    # --- Configuration ---
    PDF_FILE = "C:\\Users\\Bhunesh\\Downloads\\BHUNESH DADHEECH DISSERTATIION.pdf" 
    OPENAI_API_KEY = "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" # Insert your API key
    USER_QUESTION = "What is the main topic of this document?"
    
    # --- The Loop ---
    print("1. Extracting text...")
    raw_text = extract_text_from_pdf(PDF_FILE)
    
    print("2. Chunking text...")
    text_chunks = chunk_text(raw_text)
    
    print("3. Embedding and storing...")
    collection, embed_model = setup_vector_database(text_chunks)
    
    print("4. Retrieving context...")
    retrieved_chunks = retrieve_similar_chunks(USER_QUESTION, collection, embed_model)
    
    print("5. Synthesizing answer...")
    final_answer = generate_answer(USER_QUESTION, retrieved_chunks, OPENAI_API_KEY)
    
    print("\n================ ANSWER ================\n")
    print(final_answer)
