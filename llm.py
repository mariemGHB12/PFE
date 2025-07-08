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
from pathlib import Path
from essai import build_index

# === Config API Gemini ===
API_KEY = "AIzaSyC_x5AGH-cdrq1cgZGaPJ7yQFZYzo2UAYg"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
genai.configure(api_key=API_KEY)

# === Paramètres RAG ===
TOP_K = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# === Chemins des fichiers ===
BASE_DIR = Path(__file__).parent  # dossier du script
INDEX_PATH = BASE_DIR / "rag_output" / "faiss.index"
SENTENCE_PATH = BASE_DIR / "rag_output" / "sentences.pkl"
METADATA_PATH = BASE_DIR / "rag_output" / "metadata.pkl"

# === Fonctions Utilitaires ===

def detect_language(text):
    """Détecte la langue d'un texte"""
    try:
        return detect(text)
    except:
        return 'unknown'

def ask_gemini(prompt):
    """Interroge l'API Gemini avec un prompt"""
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
    """Charge les données RAG : phrases, metadata et index FAISS"""
    # Vérifie si les fichiers existent
    if not INDEX_PATH.exists() or not SENTENCE_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Un ou plusieurs fichiers RAG sont introuvables.")

    # Chargement des phrases
    with open(str(SENTENCE_PATH), "rb") as f:
        sentences = pickle.load(f)
    # Chargement des métadonnées
    with open(str(METADATA_PATH), "rb") as f:
        metadata = pickle.load(f)
    # Chargement de l'index FAISS
    index = faiss.read_index(str(INDEX_PATH))

    return sentences, metadata, index

def get_top_k_chunks(sentences, index, query_vec, k=TOP_K):
    """Recherche les k chunks les plus proches"""
    D, I = index.search(np.array(query_vec), k)
    return [sentences[i] for i in I[0]]

def generate_answer(context_chunks: List[str], user_prompt: str):
    """Construit le prompt final et récupère la réponse Gemini"""
    lang = detect_language(user_prompt)
    if lang == 'en':
        instruction = "Reply in English."
    elif lang == 'fr':
        instruction = "Réponds en français."
    else:
        instruction = "Réponds en français même si la langue du prompt n'est pas le français."

    context = "\n---\n".join(context_chunks)

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
    """Effectue une requête : encode, récupère les chunks et génère la réponse"""
    print(f"🔎 Query: {user_prompt}")

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    query_vec = embedder.encode([user_prompt])

    # Chargement des données
    sentences, metadata, index = load_data()

    # Recherche des meilleurs chunks
    top_chunks = get_top_k_chunks(sentences, index, query_vec)

    # Sauvegarde des chunks récupérés pour debug
    with open("pristini_top_chunks.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(top_chunks):
            f.write(f"🔹 Chunk {i+1}:\n{chunk}\n{'-'*80}\n")

    print("✅ Top chunks saved to pristini_top_chunks.txt")

    # Génération de la réponse via Gemini
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
