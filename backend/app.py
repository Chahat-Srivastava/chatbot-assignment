import os
import json
import numpy as np
import faiss
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from fastapi.middleware.cors import CORSMiddleware

# Fix OpenMP issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Change this to your frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Load BAAI/bge-large-en model
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en")
model = AutoModel.from_pretrained("BAAI/bge-large-en")

# Load FAISS index
index = faiss.read_index("./faiss_indices/faiss_mparticle_index.bin")  

# Load cleaned documents
with open("./cleaned_data/cleaned_mparticle.json", "r") as f:
    docs = json.load(f)

# Function to generate embeddings
def get_embedding(text):
    """Generate embedding using BAAI/bge-large-en"""
    if isinstance(text, list):
        text = " ".join(map(str, text))
    elif not isinstance(text, str):
        text = str(text)

    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        output = model(**tokens)

    return output.pooler_output.squeeze().numpy()

# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Define request model
class QueryRequest(BaseModel):
    query: str

# Function to retrieve answers
def retrieve_answer(query, faiss_index, docs, threshold=0.6, top_k=3):
    query_embedding = get_embedding(query).reshape(1, -1)
    
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    answers = []
    for idx, score in zip(indices[0], distances[0]):
        if idx < len(docs):
            doc_entry = docs[idx]  # ✅ Get full document entry
            
            if isinstance(doc_entry, dict):
                answer_text = doc_entry.get("content", "No relevant answer found.")
                source_url = doc_entry.get("source", None)  # ✅ Extract source URL if available
            else:
                answer_text = str(doc_entry)
                source_url = None
            
            similarity = cosine_similarity(query_embedding.flatten(), get_embedding(answer_text).flatten())

            if similarity >= threshold:
                # ✅ Ensure clean and readable formatting
                formatted_text = answer_text.replace("\n", " ")  # Remove unnecessary newlines
                formatted_text = " ".join(formatted_text.split())  # Remove extra spaces

                answers.append((formatted_text, similarity, source_url))

    answers = sorted(answers, key=lambda x: x[1], reverse=True)
    
    return answers if answers else [("No relevant answer found.", 0, None)]


# API endpoint to handle queries
@app.post("/chat")
def chat(query_request: QueryRequest):
    query = query_request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    answers = retrieve_answer(query, index, docs, threshold=0.6, top_k=1)

    formatted_response = ""
    source_url = None  # Default source URL

    for i, (answer, similarity, source) in enumerate(answers):
        source_url = source  # Store the source URL if available

        # ✅ Properly format lists and links
        if answer.startswith("http"):
            formatted_response += f"🔗 [Reference Link]({answer})\n\n"
        elif answer.startswith("- "):  
            formatted_response += f"📌 {answer}\n"
        else:
            formatted_response += f"✅ **Answer {i+1} (Similarity: {similarity:.2f})**:\n\n📖 {answer}\n\n"

    print(source_url)

    return {"answer": formatted_response, "source_url": source_url}
