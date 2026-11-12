"""Vérifie qu'un TP tient sa promesse, en l'exécutant pour de vrai (modèles compris).

    python outils/verifier.py tp02              # le corrigé, puis chaque panne seule
    python outils/verifier.py tp02 --corrige    # le corrigé seulement

La propriété vérifiée : le corrigé passe tous ses contrôles ; et la panne n, injectée
seule dans le corrigé, arrête le notebook après le contrôle n-1 et avant le contrôle n.
Aucune panne n'en masque une autre, et chacune se voit là où le notebook l'annonce.
"""
import re
import sys
import time
from pathlib import Path

from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fabriquer import RACINE, TP, appliquer, notebook, numeros_de_pannes  # noqa: E402

CONTROLE_OK = re.compile(r"✅ Contrôle (\d+)")


def executer(texte: str, dossier: Path) -> tuple[list[int], int, bool, str]:
    """Contrôles réussis, nombre total de contrôles du notebook, arrêt sur erreur, dernier message."""
    nb = notebook(texte)
    total = len(set(re.findall(r"controle\((\d+),", texte)))
    erreur, message = False, ""
    try:
        NotebookClient(nb, timeout=1800, kernel_name="formation-rag-souverain",
                       resources={"metadata": {"path": str(dossier)}}).execute()
    except CellExecutionError as e:
        lignes = str(e).strip().splitlines()
        erreur, message = True, next((l for l in lignes if "❌" in l), lignes[-1])[:160]
    reussis = [int(n) for cell in nb.cells if cell.cell_type == "code"
               for sortie in cell.get("outputs", []) for n in CONTROLE_OK.findall(sortie.get("text", ""))]
    return reussis, total, erreur, message


def verifier(code: str, corrige_seul: bool = False) -> bool:
    source = (RACINE / "sources" / f"{code}.py").read_text(encoding="utf-8")
    dossier, bon = RACINE / TP[code], True
    debut = time.time()
    reussis, total, erreur, message = executer(appliquer(source, set()), dossier)
    ok = not erreur and sorted(set(reussis)) == list(range(1, total + 1))
    bon &= ok
    print(f"{code} corrigé : {'OK' if ok else 'ÉCHEC'} — contrôles {sorted(set(reussis))} sur {total}"
          f" ({time.time() - debut:.0f} s){'  ' + message if erreur else ''}")
    if corrige_seul:
        return bon
    for n in numeros_de_pannes(source):
        debut = time.time()
        reussis, total, erreur, message = executer(appliquer(source, {n}), dossier)
        ok = erreur and set(reussis) == set(range(1, n))
        bon &= ok
        print(f"{code} panne {n} : {'OK' if ok else 'ÉCHEC'} — contrôles réussis {sorted(set(reussis))}"
              f" ({time.time() - debut:.0f} s) — {message}")
    return bon


if __name__ == "__main__":
    codes = [a for a in sys.argv[1:] if not a.startswith("--")]
    resultat = all([verifier(c, "--corrige" in sys.argv) for c in codes])
    sys.exit(0 if resultat else 1)
