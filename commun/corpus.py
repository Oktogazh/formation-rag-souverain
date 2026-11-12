"""Lire le corpus fictif de Roissy : documents, mentions de contrôle, passages, questions."""
import json
from pathlib import Path

from commun import DONNEES


def _lire_document(chemin: Path) -> dict:
    texte = chemin.read_text(encoding="utf-8")
    _, entete, corps = texte.split("---\n", 2)
    meta = dict(ligne.split(": ", 1) for ligne in entete.strip().splitlines())
    titre = next(l[2:].strip() for l in corps.splitlines() if l.startswith("# "))
    return {**meta, "titre": titre, "texte": corps.strip()}


def charger_documents(avec_changements: bool = False) -> list[dict]:
    """Les documents de la police aux frontières : règles, procédures, situation des pays."""
    chemins = sorted((DONNEES / "documents").glob("*/*.md"))
    if avec_changements:
        chemins += sorted((DONNEES / "changements").glob("*.md"))
    return [_lire_document(c) for c in chemins]


def decouper(documents: list[dict]) -> list[dict]:
    """Découpe chaque document en morceaux, un par section « ## ».

    Le titre du document est recopié en tête de chaque morceau : sans lui, la
    section « Province de Hadrel » ne dirait ni de quel pays ni de quelle date elle parle.
    """
    morceaux = []
    for doc in documents:
        sections = doc["texte"].split("\n## ")[1:]
        for n, section in enumerate(sections, 1):
            intitule, _, corps = section.partition("\n")
            morceaux.append({
                "id": f"{doc['id']}#{n}", "doc": doc["id"], "famille": doc["famille"],
                "pays": doc["pays"], "date": doc["date"], "langue": doc["langue"],
                "section": intitule.strip(),
                "texte": f"{doc['titre']} — {intitule.strip()}\n\n{corps.strip()}",
            })
    return morceaux


def lire_jsonl(chemin: Path) -> list[dict]:
    return [json.loads(l) for l in chemin.read_text(encoding="utf-8").splitlines() if l.strip()]


def questions(nom: str) -> list[dict]:
    """nom : 'rag', 'agent' ou 'sans-recherche'."""
    return lire_jsonl(DONNEES / f"questions-{nom}.jsonl")


def mentions(nom: str) -> list[dict]:
    """nom : 'train-fr', 'eval-fr', 'train-en', 'eval-en' ou 'service-fr'."""
    return lire_jsonl(DONNEES / "mentions" / f"{nom}.jsonl")


def passages() -> dict[str, list[list[str]]]:
    """Historique fictif des passages aux frontières, par identifiant de passager."""
    return {p["passager"]: p["passages"] for p in lire_jsonl(DONNEES / "passages.jsonl")}
