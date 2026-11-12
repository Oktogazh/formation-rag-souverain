"""Fabrique les notebooks des TP depuis leur source unique, le corrigé annoté.

    python outils/fabriquer.py            # tous les TP
    python outils/fabriquer.py tp02       # un seul

Chaque source `sources/tpNN.py` est un notebook au format « percent » de jupytext,
qui contient le code juste et, pour chaque panne, sa version cassée :

    # <<<PANNE 2 | la recherche doit rendre les plus proches d'abord
    ordre = np.argsort(-scores)[:k]
    # ===
    ordre = np.argsort(scores)[:k]
    # >>>PANNE 2

Deux notebooks en sortent : celui du stagiaire (toutes les pannes, aucun marqueur)
et le corrigé, rangé dans le sous-dossier `corrige/` du TP (le code juste, chaque
correction signalée par un commentaire « ✔ panne n »).
"""
import re
import sys
from pathlib import Path

import jupytext

RACINE = Path(__file__).resolve().parents[1]
TP = {
    "tp01": "tp01-modele", "tp02": "tp02-rag", "tp03": "tp03-agent",
    "tp04": "tp04-fine-tuning", "tp05": "tp05-service", "tp06": "tp06-mlflow",
}
NOYAU = {"name": "formation-rag-souverain", "display_name": "Formation RAG souverain", "language": "python"}
DEBUT = re.compile(r"^(\s*)# <<<PANNE (\d+)(?: \| (.*))?$")


def appliquer(source: str, pannes: set[int] | str) -> str:
    """pannes : 'toutes', ou l'ensemble des numéros de pannes à laisser cassées."""
    sortie, etat, numero = [], None, None
    for ligne in source.splitlines():
        m = DEBUT.match(ligne)
        if m:
            etat, numero = "juste", int(m.group(2))
            casse = pannes == "toutes" or numero in pannes
            if not casse:
                sortie.append(f"{m.group(1)}# ✔ panne {numero}" + (f" : {m.group(3)}" if m.group(3) else ""))
            continue
        if etat and ligne.strip() == "# ===":
            etat = "casse"
            continue
        if etat and ligne.strip() == f"# >>>PANNE {numero}":
            etat = None
            continue
        if etat is None or (etat == "juste") != casse:
            sortie.append(ligne)
    return "\n".join(sortie) + "\n"


def numeros_de_pannes(source: str) -> list[int]:
    return sorted({int(m.group(2)) for l in source.splitlines() if (m := DEBUT.match(l))})


def notebook(texte: str):
    nb = jupytext.reads(texte, fmt="py:percent")
    nb.metadata["kernelspec"] = NOYAU
    nb.metadata.pop("jupytext", None)
    # Identifiants fixes : sinon chaque fabrication en tire au hasard et salit le diff git.
    for i, cellule in enumerate(nb.cells):
        cellule.id = f"cellule-{i:03d}"
    return nb


def fabriquer(code: str) -> None:
    source = (RACINE / "sources" / f"{code}.py").read_text(encoding="utf-8")
    dossier = RACINE / TP[code]
    (dossier / "corrige").mkdir(parents=True, exist_ok=True)
    jupytext.write(notebook(appliquer(source, "toutes")), dossier / f"{code}.ipynb")
    jupytext.write(notebook(appliquer(source, set())), dossier / "corrige" / f"{code}-corrige.ipynb")
    print(f"{TP[code]} : {len(numeros_de_pannes(source))} pannes")


if __name__ == "__main__":
    for code in (sys.argv[1:] or [c for c in TP if (RACINE / "sources" / f"{c}.py").exists()]):
        fabriquer(code)
