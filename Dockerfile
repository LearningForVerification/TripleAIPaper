FROM continuumio/miniconda3

# Installa git
RUN apt-get update && apt-get install -y git

WORKDIR /app

# Clona la repo TripleAIPaper
RUN git clone https://github.com/LearningForVerification/TripleAIPaper.git

# Copia requirements.txt dalla repo clonata
COPY TripleAIPaper/requirements.txt ./requirements.txt

# Crea ambiente conda Python 3.10.12
RUN conda create -n myenv python=3.10.12 -y

# Aggiorna pip e installa pacchetti da requirements.txt (escludendo torch/vision/audio per CUDA)
RUN conda run -n myenv pip install --upgrade pip && \
    conda run -n myenv pip install -r requirements.txt --no-deps

# Installa PyTorch, torchvision e torchaudio con supporto CUDA 11.8 da index PyTorch
RUN conda run -n myenv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Installa auto-LiRPA da git
RUN conda run -n myenv pip install git+https://github.com/Verified-Intelligence/auto_LiRPA.git

# Usa sempre l'ambiente conda
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Imposta la working directory nella repo
WORKDIR /app/TripleAIPaper

# Comando di default: esegue lo script con argomenti passati
ENTRYPOINT ["python", "training/one_rs_param/shallow_networks_script.py"]
CMD []
