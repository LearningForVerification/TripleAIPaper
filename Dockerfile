FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN apt-get update && apt-get install -y \
    wget git bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -t -i -p -y && \
    ln -s $CONDA_DIR/etc/profile.d/conda.sh /etc/profile.d/conda.sh

WORKDIR /app

RUN git clone https://github.com/LearningForVerification/TripleAIPaper.git

RUN conda config --set channel_priority strict && \
    conda config --add channels defaults && \
    conda tos accept --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n myenv python=3.10 -y

RUN conda run -n myenv pip install --upgrade pip
RUN conda run -n myenv pip install -r /app/TripleAIPaper/requirements.txt --no-deps
RUN conda run -n myenv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN conda run -n myenv pip install git+https://github.com/Verified-Intelligence/auto_LiRPA.git

# Copia lo script di entrypoint nel container
COPY docker-entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Usa lo script come entrypoint
CMD ["/app/entrypoint.sh"]
