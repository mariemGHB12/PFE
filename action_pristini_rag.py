import numpy as np
import faiss
import pickle
import os
import json
from sentence_transformers import SentenceTransformer

# Chemins des fichiers
JSON_PATH = "pristini_programs.json"
OUTPUT_DIR = "rag_output"
FAISS_INDEX_OUTPUT = os.path.join(OUTPUT_DIR, "faiss.index")
SENTENCES_OUTPUT = os.path.join(OUTPUT_DIR, "sentences.pkl")
METADATA_OUTPUT = os.path.join(OUTPUT_DIR, "metadata.pkl")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Correction : si data est une liste contenant une liste
if isinstance(data[0], list):
    data = data[0]

# Chargement du modèle d'embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = []
metadata = []

# Conversion des objets JSON en phrases et metadata
"""for program in data:
    desc = " ".join(program.get("description_blocks", []))
    modules = " ".join([f"{k} : {', '.join(v)}." for k, v in program.get("modules", {}).items()])

    # Construction de la phrase
    sentence = (
    f"🌟 Programme : {program.get('title', 'Titre inconnu')}\n"
    f"🏫 Établissement : {program.get('delivered_by', 'Établissement inconnu')}\n"
    f"⏱ Durée : {program.get('duration', 'Non spécifiée')}\n\n"
    f"📝 Description :\n{desc if desc else 'Aucune description disponible'}\n\n"
    f"📚 Modules clés :\n{modules if modules else 'Modules non disponibles'}\n\n"
    f"🔗 Lien vers le programme : {program.get('url', 'URL non disponible')}\n"
    f"📅 Prochaine session : {program.get('start_date', 'À définir')}"
)

sentences.append(sentence)

metadata.append({
    "Programme": program.get("title", "Titre inconnu"),
    "Établissement": program.get("delivered_by", "Établissement inconnu"),
    "Durée": program.get("duration", "Non spécifiée"),
    "Type": "Master" if "master" in program.get('title', '').lower() else "Autre",
    "URL": program.get("url", ""),
    "Langue": program.get("language", "Anglais"),
    "Modalité": program.get("modality", "Présentiel"),
    "Crédits ECTS": program.get("ects", "Non spécifié"),
    "Diplôme délivré": program.get("diploma", "Master"),
    "Admission": program.get("admission", "Sur dossier"),
    "Prix": program.get("price", "Non communiqué")
})
print("Génération des embeddings...")"""

for program in data:
    # Vérifie que program est bien un dict et qu'il contient la clé 'title'
    if isinstance(program, dict) and 'title' in program:
        desc = " ".join(program.get("description_blocks", []))
        modules = " ".join([f"{k} : {', '.join(v)}." for k, v in program.get("modules", {}).items()])

        sentence = (
            f"🌟 Programme : {program.get('title')}\n"
            f"🏫 Établissement : {program.get('delivered_by')}\n"
            f"⏱ Durée : {program.get('duration')}\n\n"
            f"📝 Description :\n{desc}\n\n"
            f"📚 Modules clés :\n{modules}\n\n"
            f"🔗 Lien vers le programme : {program.get('url')}\n"
            f"📅 Prochaine session : {program.get('start_date', 'À définir')}"
        )
        sentences.append(sentence)

        metadata.append({
            "Programme": program.get("title"),
            "Établissement": program.get("delivered_by"),
            "Durée": program.get("duration"),
            "Type": "Master" if "master" in program.get('title', '').lower() else "Autre",
            "URL": program.get("url", ""),
            "Langue": program.get("language", "Anglais"),
            "Modalité": program.get("modality", "Présentiel"),
            "Crédits ECTS": program.get("ects", "Non spécifié"),
            "Diplôme délivré": program.get("diploma", "Master"),
            "Admission": program.get("admission", "Sur dossier"),
            "Prix": program.get("price", "Non communiqué")
        })


# Calcul des embeddings
embeddings = model.encode(sentences, show_progress_bar=True)

# Construction de l'index FAISS
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Sauvegarde des fichiers
#os.makedirs(OUTPUT_DIR, exist_ok=True)
faiss.write_index(index, FAISS_INDEX_OUTPUT)

with open(SENTENCES_OUTPUT, "wb") as f:
    pickle.dump(sentences, f)
with open(METADATA_OUTPUT, "wb") as f:
    pickle.dump(metadata, f)

print("✅ Index FAISS, phrases et metadata sauvegardés dans 'rag_output/'")
