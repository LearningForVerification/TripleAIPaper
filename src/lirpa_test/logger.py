import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Crea la directory dei log se non esiste
LOG_FILE = "application.log"

# Nome del file di log comune
LOG_FILE = "application.log"


def setup_logging():
    """
    Configura il logging a livello radice per inviare tutti i messaggi allo stesso file
    con livello INFO.
    """
    # Ottieni il logger root
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Controlla se è già stato configurato
    if not root_logger.handlers:
        # Crea il formatter comune
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Crea il file handler con rotazione
        file_handler = RotatingFileHandler(
            log_dir / LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # Aggiungi l'handler al logger root
        root_logger.addHandler(file_handler)

        # Aggiungi anche un handler per la console (opzionale)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        logging.info("Logging inizializzato. Tutti i messaggi di log verranno salvati in: %s", log_dir / LOG_FILE)


# Esegui la configurazione quando questo modulo viene importato
setup_logging()