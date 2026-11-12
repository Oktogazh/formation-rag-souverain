"""Deux commandes.

    python tp.py check      vérifie que votre machine est prête pour les TP ;
                            envoyez la sortie complète au formateur avant la session.
    python tp.py corriger   remplace les six notebooks par leur version corrigée,
                            après avoir mis les vôtres de côté dans resultats/mes-notebooks/.
"""
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("DO_NOT_TRACK", "true")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

PAQUETS = ["openai", "numpy", "langchain", "langchain_ollama", "fastapi", "uvicorn",
           "httpx", "torch", "transformers", "peft", "mlflow", "jupyterlab"]
MODELES_OLLAMA = ["ministral-3:3b", "bge-m3"]
PETIT_MODELE = "HuggingFaceTB/SmolLM2-135M-Instruct"


def ligne(ok: bool, texte: str) -> bool:
    print(f"  {'✅' if ok else '❌'} {texte}")
    return ok


def check() -> int:
    bon = True
    print(f"Système : {platform.system()} {platform.release()} · {platform.machine()} · Python {platform.python_version()}")
    print("\nBibliothèques")
    bon &= ligne(sys.version_info[:2] == (3, 11), "Python 3.11 (environnement formation-rag-souverain)")
    for nom in PAQUETS:
        try:
            importlib.import_module(nom)
            ligne(True, nom)
        except ImportError:
            bon &= ligne(False, f"{nom} manquant : conda env update -f environment.yml")
    print("\nOllama")
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            presents = {m["name"] for m in json.load(r)["models"]}
        ligne(True, "Ollama répond sur localhost:11434")
        for m in MODELES_OLLAMA:
            bon &= ligne(any(p.split(":latest")[0] == m for p in presents), f"{m}" + ("" if m in presents or f"{m}:latest" in presents else f" manquant : ollama pull {m}"))
    except OSError:
        bon &= ligne(False, "Ollama ne répond pas : lancez l'application Ollama, ou « ollama serve »")
    print("\nPetit modèle du fine-tuning (TP 4 et 6)")
    try:
        from huggingface_hub import snapshot_download
        # Seulement les fichiers utiles (≈ 270 Mo), pas les variantes ONNX du dépôt.
        snapshot_download(PETIT_MODELE, allow_patterns=["*.json", "*.txt", "model.safetensors"])
        ligne(True, f"{PETIT_MODELE} présent sur la machine (téléchargé au besoin)")
    except Exception as erreur:  # noqa: BLE001 — on affiche la cause, quelle qu'elle soit
        bon &= ligne(False, f"{PETIT_MODELE} introuvable ({type(erreur).__name__})")
    print("\nJupyter")
    noyaux = subprocess.run([sys.executable, "-m", "jupyter", "kernelspec", "list"], capture_output=True, text=True).stdout
    bon &= ligne("formation-rag-souverain" in noyaux,
                 "noyau formation-rag-souverain" + ("" if "formation-rag-souverain" in noyaux else
                 " manquant : python -m ipykernel install --user --name formation-rag-souverain"))
    print("\n" + ("Tout est prêt." if bon else "Il manque quelque chose : voir les ❌ ci-dessus."))
    return 0 if bon else 1


def corriger() -> int:
    racine = Path(__file__).resolve().parent
    cote = racine / "resultats" / "mes-notebooks"
    cote.mkdir(parents=True, exist_ok=True)
    for corrige in sorted(racine.glob("tp0*/corrige/tp0*-corrige.ipynb")):
        notebook = corrige.parents[1] / corrige.name.replace("-corrige", "")
        # Déjà corrigé (commande relancée) : surtout ne pas écraser la copie de côté.
        if notebook.exists() and notebook.read_bytes() != corrige.read_bytes():
            shutil.copy2(notebook, cote / notebook.name)
        shutil.copy2(corrige, notebook)
        print(f"  ✅ {notebook.relative_to(racine)} : version corrigée")
    print(f"\nVos versions sont gardées dans {cote.relative_to(racine)}/.")
    return 0


if __name__ == "__main__":
    commandes = {"check": check, "corriger": corriger}
    if len(sys.argv) != 2 or sys.argv[1] not in commandes:
        print(__doc__)
        sys.exit(2)
    sys.exit(commandes[sys.argv[1]]())
