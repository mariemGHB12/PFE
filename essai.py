import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def build_index(json_path, output_dir):
    # Préparation des chemins de sortie
    faiss_index_output = os.path.join(output_dir, "faiss.index")
    sentences_output = os.path.join(output_dir, "sentences.pkl")
    metadata_output = os.path.join(output_dir, "metadata.pkl")

    # Chargement des données
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    programs = [
        p for p in raw_data
        if isinstance(p, dict) and "title" in p and "description_blocks" in p
    ]

    model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences = []
    metadata = []

    for program in programs:
        desc = " ".join(program.get("description_blocks", []))
        modules = " ".join([
            f"{k} : {', '.join(v)}." for k, v in program.get("modules", {}).items()
        ])

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
            "Langue": program.get("language", "Français"),
            "Modalité": program.get("modality", "Présentiel"),
            "Crédits ECTS": program.get("ects", "Non spécifié"),
            "Diplôme délivré": program.get("diploma", "Master"),
            "Admission": program.get("admission", "Sur dossier"),
            "Prix": program.get("price", "Non communiqué")
        })

    print("🔍 Génération des embeddings...")
    embeddings = model.encode(sentences, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    print("💾 Sauvegarde de l'index FAISS et des données...")
    os.makedirs(output_dir, exist_ok=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, faiss_index_output)

    with open(sentences_output, "wb") as f:
        pickle.dump(sentences, f)

    with open(metadata_output, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Index FAISS et données sauvegardées dans '{output_dir}/'")