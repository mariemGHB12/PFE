import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from essai import build_index
# Importer la lib Gemini
import google.generativeai as genai
from typing import List

# Configurer l'API KEY
genai.configure(api_key="AIzaSyC_x5AGH-cdrq1cgZGaPJ7yQFZYzo2UAYg")

# === CONFIG ===
TOP_K = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "C:\\Users\\marie\\OneDrive\\Desktop\\test\\rag_output\\faiss.index"
SENTENCE_PATH = "C:\\Users\\marie\\OneDrive\\Desktop\\test\\rag_output\\sentences.pkl"
METADATA_PATH = "C:\\Users\\marie\\OneDrive\\Desktop\\test\\rag_output\\metadata.pkl"
JSON_PATH = "pristini_programs.json"
OUTPUT_DIR = "rag_output"
#build_index(JSON_PATH, OUTPUT_DIR)
# === Load Pristini Data ===
def load_data():
    with open(SENTENCE_PATH, "rb") as f:
        sentences = pickle.load(f)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    index = faiss.read_index(INDEX_PATH)
    return sentences, metadata, index


# === Retrieve Top-K Similar Transactions ===
def get_top_k_chunks(sentences, index, query_vec, k=TOP_K):
    D, I = index.search(np.array(query_vec), k)
    return [sentences[i] for i in I[0]]

def detect_needs(user_prompt):
    needs = []
    prompt_lower = user_prompt.lower()
    
    # Check for program types
    if "master" in prompt_lower or "masters" in prompt_lower:
        needs.append("MASTERS")
    if "bachelor" in prompt_lower or "licence" in prompt_lower:
        needs.append("BACHELOR")
    if "formation certifiante" in prompt_lower or "certification" in prompt_lower:
        needs.append("FORMATIONS CERTIFIANTES")
    if "innovation" in prompt_lower or "startup" in prompt_lower:
        needs.append("INNOVATIONS")
    
    # Check for technical domains
    if ("data" in prompt_lower or "donnée" in prompt_lower or 
        "big data" in prompt_lower or "data science" in prompt_lower):
        needs.append("Data Science")
    if ("ia" in prompt_lower or "intelligence artificielle" in prompt_lower or 
        "machine learning" in prompt_lower or "deep learning" in prompt_lower):
        needs.append("Intelligence Artificielle")
    if ("management" in prompt_lower or "gestion" in prompt_lower or 
        "business" in prompt_lower or "entreprise" in prompt_lower):
        needs.append("Management")
    if ("industrie 4.0" in prompt_lower or "i4.0" in prompt_lower or 
        "robotique" in prompt_lower or "iot" in prompt_lower):
        needs.append("Industry 4.0")
    
    # Check for specific programs
    if "machine learning for business" in prompt_lower:
        needs.append("Machine Learning For Business")
    if "data science ai industry 4.0" in prompt_lower:
        needs.append("Data Science, AI & Industry 4.0")
    if "applied artificial intelligence" in prompt_lower:
        needs.append("Applied Artificial Intelligence")
    if "business intelligence" in prompt_lower:
        needs.append("Business Intelligence")
    
    # If no specific needs detected, return general programs
    if not needs:
        needs = ["MASTERS", "BACHELOR", "FORMATIONS CERTIFIANTES", "INNOVATIONS"]
    
    return list(set(needs))  # Remove duplicatesje 
def generate_answer(context_chunks: List[str], user_prompt: str):
    # Joindre les contextes récupérés avec un séparateur
    context = "\n---\n".join(context_chunks)

    # Construire le prompt complet
    full_prompt = f"""
Tu es un assistant expert en orientation universitaire.
Utilise uniquement les informations suivantes sur les programmes Pristini.
si l utilistateur fournit un  message vide demande lui de ne pas mettre un message vide 

Donne une réponse claire, concise et résumée, en mettant en avant les points clés et sans détails superflus.et si tu connais pas la reponse excuse toi et demande de reformuler la question ou de donner plus de context  
tu dois te presenter et repondre au salutation de l'utilsateur 
Context:
{context}

User question:
{user_prompt}

Answer:
"""

    # Initialiser le modèle Gemini (vérifie bien ta version lib >= 1.0.0)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    # Générer la réponse
    response = model.generate_content(full_prompt)

    # Retourner la réponse texte
    return response.text



# === Main Query Function ===
def query_data(user_prompt):
    print(f"🔎 Query: {user_prompt}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    query_vec = embedder.encode([user_prompt])

    # Load sentence chunks and FAISS index
    sentences, metadata, index = load_data()

    # Retrieve top-K relevant transaction descriptions
    top_chunks = get_top_k_chunks(sentences, index, query_vec)

    # Save top chunks for inspection
    with open("pristini_top_chunks.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(top_chunks):
            f.write(f"🔹 Chunk {i + 1}:\n{chunk}\n{'-' * 80}\n")
    print("✅ Top chunks saved to pristini_top_chunks.txt")

    # Generate the answer using Gemma
    answer = generate_answer(top_chunks, user_prompt)
    
    return answer


# === Run It ===
if __name__ == "__main__":

    user_question = input("Ask any Pristini-related question: ")
    result = query_data(user_question)
    print("\n💡 LLM Response:\n", result)

    with open("pristini_llm_response.txt", "w", encoding="utf-8") as f:
        f.write("💡 LLM Response:\n")
        f.write(result)