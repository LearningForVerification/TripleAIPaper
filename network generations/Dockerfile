# Immagine ufficiale PyTorch con CUDA e CUDNN (supporta GPU)
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Imposta la working directory
WORKDIR /app

# Copia tutti i file del progetto nella working directory
COPY . .

# Installa le librerie necessarie
RUN pip install --no-cache-dir \
    torchvision \
    onnx \
    pandas

# Crea directory per salvataggi se non esistono
RUN mkdir -p /app/models /app/data

# Comando di default: esegui lo script principale
CMD ["/bin/bash"]