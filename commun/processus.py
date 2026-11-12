"""Lancer et arrêter les services des TP 5 et 6 depuis un notebook.

Rien ici n'est à apprendre : c'est de la plomberie, pour que tout se lance depuis
le notebook, sans second terminal.
"""
import atexit
import subprocess
import sys
import threading
import time

import httpx
import uvicorn

_serveurs: dict[int, uvicorn.Server] = {}
_processus: dict[str, subprocess.Popen] = {}


def servir(app, hote: str = "127.0.0.1", port: int = 8000) -> uvicorn.Server:
    """Démarre une application FastAPI dans un fil d'exécution à part. Relancer remplace l'ancienne."""
    arreter_serveur(port)
    serveur = uvicorn.Server(uvicorn.Config(app, host=hote, port=port, log_level="warning"))
    threading.Thread(target=serveur.run, daemon=True).start()
    for _ in range(100):
        if serveur.started:
            break
        time.sleep(0.05)
    _serveurs[port] = serveur
    return serveur


def arreter_serveur(port: int) -> None:
    serveur = _serveurs.pop(port, None)
    if serveur:
        serveur.should_exit = True
        time.sleep(0.5)


def lancer(nom: str, arguments: list[str], adresse: str, cwd=None, sante: str | None = None) -> str:
    """Lance une commande Python (mlflow) en arrière-plan et attend qu'elle réponde.

    sante : un chemin qui répond 200 seulement si c'est bien notre service (« /health »
    pour MLflow). S'il répond, le service tourne déjà — lancé par un noyau précédent,
    par exemple — et on le réutilise au lieu d'échouer.
    """
    arreter(nom)
    if sante and _repond(adresse + sante, exiger_200=True):
        print(f"{nom} tourne déjà sur {adresse} : on le réutilise.")
        return adresse
    if _repond(adresse):
        raise RuntimeError(f"Un autre programme occupe déjà {adresse} : impossible d'y lancer {nom}.")
    p = subprocess.Popen([sys.executable, "-m", *arguments], cwd=cwd,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _processus[nom] = p
    for _ in range(120):
        if p.poll() is not None:
            raise RuntimeError(f"{nom} s'est arrêté au démarrage (code {p.returncode}).")
        if _repond(adresse):
            return adresse
        time.sleep(0.5)
    raise RuntimeError(f"{nom} ne répond pas sur {adresse}")


def _repond(adresse: str, exiger_200: bool = False) -> bool:
    try:
        r = httpx.get(adresse, timeout=1)
        return r.status_code == 200 if exiger_200 else True
    except httpx.HTTPError:
        return False


def arreter(nom: str) -> None:
    p = _processus.pop(nom, None)
    if p:
        p.terminate()
        p.wait(timeout=10)


@atexit.register
def _tout_arreter():
    for nom in list(_processus):
        arreter(nom)
