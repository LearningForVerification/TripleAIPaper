FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Imposta la variabile per Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Installa dipendenze di sistema e Miniconda
RUN apt-get update && apt-get install -y \
    wget git bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -t -i -p -y && \
    ln -s $CONDA_DIR/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# Crea la cartella di lavoro
WORKDIR /app

# Clona il repository
RUN git clone https://github.com/LearningForVerification/TripleAIPaper.git

# Crea ambiente conda con Python 3.10.12
RUN conda create -n myenv python=3.10.12 -y

# Installa i pacchetti Python richiesti nell'ambiente
RUN conda run -n myenv pip install --upgrade pip && \
    conda run -n myenv pip install -r /app/TripleAIPaper/requirements.txt --no-deps && \
    conda run -n myenv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    conda run -n myenv pip install git+https://github.com/Verified-Intelligence/auto_LiRPA.git

# Imposta la shell per usare automaticamente l'ambiente conda
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Setta PYTHONPATH e cartella di lavoro finale
ENV PYTHONPATH=/app/TripleAIPaper
WORKDIR /app/TripleAIPaper

# Comando di default all'avvio del container
ENTRYPOINT ["conda", "run", "-n", "myenv", "python"]
