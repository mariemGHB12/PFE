# Utilise une image Python officielle
FROM python:3.12-slim

# Définit le dossier de travail dans le conteneur
WORKDIR /app

# Copie les fichiers locaux dans le conteneur
COPY . .

# Installe les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Expose le port 5000
EXPOSE 5000

# Commande pour démarrer l'app
CMD ["python", "app.py"]
