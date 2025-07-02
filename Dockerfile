FROM continuumio/miniconda3

# Installa git e pulisci cache
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Crea cartella di lavoro
WORKDIR /app

# Clona la repo
RUN git clone https://github.com/LearningForVerification/TripleAIPaper.git

# Crea ambiente conda e installa Python
RUN conda create -n myenv python=3.10.12 -y

# Aggiorna pip e installa i requirements SENZA dipendenze
RUN conda run -n myenv pip install --upgrade pip && \
    conda run -n myenv pip install -r /app/TripleAIPaper/requirements.txt --no-deps

# Installa PyTorch con CUDA 11.8
RUN conda run -n myenv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Installa auto-LiRPA da Git
RUN conda run -n myenv pip install git+https://github.com/Verified-Intelligence/auto_LiRPA.git

# Imposta shell per usare l'ambiente conda automaticamente
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

ENV PYTHONPATH=/app/TripleAIPaper
WORKDIR /app/TripleAIPaper

# ENTRYPOINT diretto su python
ENTRYPOINT ["conda", "run", "-n", "myenv", "python"]
