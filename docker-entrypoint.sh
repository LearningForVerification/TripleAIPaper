#!/bin/bash

# Attiva conda
source /opt/conda/etc/profile.d/conda.sh
conda activate myenv

# Setta PYTHONPATH per rendere visibili i moduli locali
export PYTHONPATH=/app/TripleAIPaper

# Avvia una shell interattiva
exec /bin/bash
