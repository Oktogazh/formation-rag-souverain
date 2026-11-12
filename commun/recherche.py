"""La documentation de la police aux frontières, indexée et prête à interroger.

C'est la version de référence de ce que le TP 2 construit à la main. Les TP 3, 5
et 6 s'en servent, pour rester indépendants du TP 2.

La recherche est exacte, en numpy : une multiplication matrice × vecteur, puis un tri.
C'est ce que fait l'index exact de FAISS (IndexFlatIP), sans FAISS. Raison : sur
macOS, FAISS et PyTorch, installés par pip, embarquent chacun leur propre OpenMP ;
dans un même processus (le TP 6 entraîne un modèle puis cherche), le second fait
planter le noyau (« OMP: Error #15 »). Sur quelques dizaines de morceaux, FAISS
n'apporterait rien ; sur des millions, il faudrait l'installer par conda, avec
PyTorch, pour qu'ils partagent le même OpenMP.
"""
import hashlib

import numpy as np

from commun import RESULTATS, corpus
from commun.modele import ENCODEUR, vectoriser


class Documentation:
    def __init__(self, avec_changements: bool = False):
        self.morceaux = corpus.decouper(corpus.charger_documents(avec_changements))
        textes = [m["texte"] for m in self.morceaux]
        cle = hashlib.sha1((ENCODEUR + "\x00".join(textes)).encode()).hexdigest()[:12]
        cache = RESULTATS / f"vecteurs-{cle}.npy"
        if cache.exists():
            vecteurs = np.load(cache)
        else:
            vecteurs = vectoriser(textes)
            np.save(cache, vecteurs)
        self.vecteurs = vecteurs

    def chercher(self, question: str, k: int = 3, date_min: str | None = None,
                 pays: str | None = None) -> list[dict]:
        """Les k morceaux les plus proches de la question, filtrés par date et par pays si demandé."""
        scores = self.vecteurs @ vectoriser([question])[0]
        retenus = []
        for rang in np.argsort(-scores):
            score = scores[rang]
            m = self.morceaux[rang]
            if date_min and m["date"] < date_min:
                continue
            if pays and m["pays"] not in (pays, "tous"):
                continue
            retenus.append({**m, "score": float(score)})
            if len(retenus) == k:
                break
        return retenus

    def contexte(self, question: str, k: int = 3, **filtres) -> str:
        """Les morceaux trouvés, mis en forme pour un prompt, chacun avec sa source."""
        return "\n\n".join(f"[{m['id']}, {m['date']}]\n{m['texte']}"
                           for m in self.chercher(question, k, **filtres))
