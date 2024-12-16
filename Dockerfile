FROM python:3.9-slim

# Installer git
RUN apt-get update && \
    apt-get install -y --no-install-recommends git ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Créer le répertoire de travail
RUN mkdir /app
WORKDIR /app

# Cloner le repo
RUN git clone --depth=2 --branch=master https://github.com/FloWPs/PWML_Detection_and_Segmentation

# Changer le répertoire de travail
WORKDIR /app/PWML_Detection_and_Segmentation

# Créer les dossiers "models" et "output"
RUN mkdir models && mkdir output

# Copier les modèles (en local) dans le conteneur
COPY models ./models

# Installer les librairies
RUN pip install -r requirements.txt

