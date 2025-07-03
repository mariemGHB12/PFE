# === Imports ===
import os
import pickle
import faiss
import numpy as np
import requests
from typing import List
from langdetect import detect
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from essai import build_index

# === Config API Gemini ===
API_KEY = "AIzaSyC_x5AGH-cdrq1cgZGaPJ7yQFZYzo2UAYg"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
genai.configure(api_key=API_KEY)

# === Paramètres RAG ===
TOP_K = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "C:\\Users\\marie\\OneDrive\\Desktop\\test\\rag_output\\faiss.index"
SENTENCE_PATH = "C:\\Users\\marie\\OneDrive\\Desktop\\test\\rag_output\\sentences.pkl"
METADATA_PATH = "C:\\Users\\marie\\OneDrive\\Desktop\\test\\rag_output\\metadata.pkl"

# === Fonctions Utilitaires ===

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'

def ask_gemini(prompt):
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"❌ Error: {response.status_code}, {response.text}"

def load_data():
    
    with open(SENTENCE_PATH, "rb") as f:
        sentences = pickle.load(f)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    index = faiss.read_index(INDEX_PATH)
    return sentences, metadata, index

def get_top_k_chunks(sentences, index, query_vec, k=TOP_K):
    D, I = index.search(np.array(query_vec), k)
    return [sentences[i] for i in I[0]]

def generate_answer(context_chunks: List[str], user_prompt: str):
    # Détecter la langue sur la question utilisateur uniquement
    lang = detect_language(user_prompt)
    if lang == 'en':
        instruction = "Reply in English."
    elif lang == 'fr':
        instruction = "Réponds en français."
    else:
        instruction = "Réponds en français même si la langue du prompt n'est pas le français."

    # Joindre les contextes récupérés
    context = "\n---\n".join(context_chunks)

    # Construire le prompt complet
    full_prompt = f"""
{instruction}

Tu es un assistant expert en orientation universitaire.
Utilise uniquement les informations suivantes sur les programmes Pristini.

Donne une réponse claire, concise et résumée, en mettant en avant les points clés et sans détails superflus.
Si tu ne connais pas la réponse, excuse-toi et demande à l'utilisateur de reformuler ou d'apporter plus de contexte.
Présente-toi et réponds aux salutations de l'utilisateur.

Context:
{context}

User question:
{user_prompt}

Answer:
"""

    response = ask_gemini(full_prompt)
    return response

def query_data(user_prompt):
    print(f"🔎 Query: {user_prompt}")

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    query_vec = embedder.encode([user_prompt])
    sentences, metadata, index = load_data()
    top_chunks = get_top_k_chunks(sentences, index, query_vec)
    with open("pristini_top_chunks.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(top_chunks):
            f.write(f"🔹 Chunk {i+1}:\n{chunk}\n{'-'*80}\n")
    print("✅ Top chunks saved to pristini_top_chunks.txt")
    answer = generate_answer(top_chunks, user_prompt)
    return answer

# === Run CLI ===
if __name__ == "__main__":
    user_question = input("Ask any Pristini-related question: ")
    result = query_data(user_question)
    print("\n💡 LLM Response:\n", result)
    with open("pristini_llm_response.txt", "w", encoding="utf-8") as f:
        f.write("💡 LLM Response:\n")
        f.write(result)
