"""Le socle commun des TP : ce qu'un TP réutilise sans le réécrire.

Rien ici n'est à modifier pendant un TP. Ce que le TP enseigne est écrit dans le
notebook ; ce dossier ne contient que le chargement des données, la connexion aux
modèles locaux et les objets dont un TP a besoin pour être indépendant des autres.
"""
import os
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
DONNEES = RACINE / "data" / "roissy"
RESULTATS = RACINE / "resultats"
RESULTATS.mkdir(exist_ok=True)

# Tout reste sur la machine : on coupe les statistiques d'usage que certains outils
# envoient par défaut à leurs développeurs.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("DO_NOT_TRACK", "true")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
