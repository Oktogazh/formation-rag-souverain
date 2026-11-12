"""Fabrique les mentions de contrôle fictives et les historiques de passages.

Usage formateur : python scripts/fabriquer_mentions.py
Déterministe (graine fixe) : relancer le script redonne exactement les mêmes fichiers.

Les mentions sont produites par gabarits, pas par un modèle : chaque mention a une
extraction attendue certaine, ce qui rend la mesure du TP 4 et du TP 6 incontestable.
"""
import json
import random
from datetime import date, timedelta
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
SORTIE = RACINE / "data" / "roissy"
GRAINE = 20261109

PAYS = {
    "Valdorie": {"adj_m": "valdorien", "adj_f": "valdorienne", "en": "Valdorian"},
    "Morvanie": {"adj_m": "morvanien", "adj_f": "morvanienne", "en": "Morvanian"},
    "Ostrane": {"adj_m": "ostranien", "adj_f": "ostranienne", "en": "Ostranian"},
}
DOCUMENTS = {
    "passeport ordinaire": ["passeport ordinaire", "passeport ordinaire en cours de validité",
                            "passeport ordinaire valide"],
    "passeport de service": ["passeport de service", "passeport de service valide"],
    "titre de voyage": ["titre de voyage pour réfugié", "titre de voyage"],
}
DOCUMENTS_EN = {"passeport ordinaire": "ordinary passport", "passeport de service": "service passport",
                "titre de voyage": "refugee travel document"}
MOTIFS = {
    "tourisme": ["déclare venir en vacances", "indique un séjour touristique de deux semaines",
                 "vient visiter Paris", "déclare un voyage d'agrément"],
    "visite familiale": ["déclare rendre visite à sa sœur", "vient voir son frère installé à Lyon",
                         "indique rejoindre sa famille pour un mariage", "déclare rendre visite à ses parents"],
    "affaires": ["déclare un rendez-vous professionnel à La Défense", "vient pour une réunion de travail",
                 "indique participer à un salon professionnel", "déclare un déplacement pour son employeur"],
    "transit": ["est en correspondance pour Montréal", "indique un transit vers Dakar",
                "déclare poursuivre son voyage vers Lisbonne", "attend une correspondance pour Tunis"],
    "études": ["déclare venir s'inscrire à l'université", "indique commencer un stage d'études",
               "vient passer un concours dans une école", "déclare suivre une formation linguistique"],
}
MOTIFS_EN = {
    "tourisme": ["states a two-week holiday", "says he is visiting Paris as a tourist"],
    "visite familiale": ["states visiting his sister", "says she is joining her family for a wedding"],
    "affaires": ["states a business meeting in La Défense", "says he attends a trade fair"],
    "transit": ["is connecting to Montreal", "states onward travel to Lisbon"],
    "études": ["says she is enrolling at university", "states attending a language course"],
}
ASILE_OUI = ["Déclare vouloir demander l'asile.", "Sollicite une protection et dit craindre un retour dans son pays.",
             "Indique craindre pour sa sécurité en cas de retour et demande l'asile.",
             "Formule une demande d'asile à la frontière."]
ASILE_NON = ["", "Ne formule aucune demande de protection.", "Aucune demande d'asile.",
             "Billet retour présenté.", "Réservation d'hôtel présentée."]
ASILE_OUI_EN = ["Wishes to apply for asylum.", "Requests protection, fears returning home."]
ASILE_NON_EN = ["", "No protection request.", "Return ticket shown."]
PRENOMS = ["Arvel", "Mira", "Teodor", "Lenka", "Iskar", "Daria", "Bojan", "Velia", "Radek", "Sena",
           "Oskel", "Yana", "Doran", "Ilka", "Maren", "Tovan"]
NOMS = ["DOSKA", "VARENIK", "OSTROV", "BRENNAR", "SALIK", "MORAVEC", "KESTEL", "LUNAR", "HADRIC",
        "VELKOV", "IVRANI", "SORBEK", "TAVEN", "ZERIK"]
VOLS = ["MV218", "VD402", "OS117", "MV224", "VD410", "OS131"]


def mention_fr(r, pays, doc, motif, asile):
    genre = r.choice("mf")
    adj, nationalite = PAYS[pays]["adj_" + genre], PAYS[pays]["adj_f"]
    civ = "M." if genre == "m" else "Mme"
    nom = f"{civ} {r.choice(PRENOMS)} {r.choice(NOMS)}"
    heure = f"{r.randint(5, 23):02d}h{r.randint(0, 59):02d}"
    vol = r.choice(VOLS)
    porteur = r.choice([f"{'Passager' if genre == 'm' else 'Passagère'} de nationalité {nationalite}",
                        f"Ressortissant{'e' if genre == 'f' else ''} {adj}",
                        f"{nom}, de nationalité {nationalite}"])
    morceaux = [f"{r.randint(1, 28):02d}/11 {heure}, vol {vol}.",
                f"{porteur}, {r.choice(DOCUMENTS[doc])}.",
                f"{'Il' if genre == 'm' else 'Elle'} {r.choice(MOTIFS[motif])}.",
                (r.choice(ASILE_OUI) if asile else r.choice(ASILE_NON))]
    if r.random() < 0.3:
        morceaux.insert(1, r.choice(["Contrôle approfondi au poste 12.", "Signalé par la compagnie.",
                                     "Arrivée terminal 2E.", "Interprète non nécessaire."]))
    return " ".join(m for m in morceaux if m), nom


def mention_en(r, pays, doc, motif, asile):
    genre = r.choice("mf")
    heure = f"{r.randint(5, 23):02d}:{r.randint(0, 59):02d}"
    morceaux = [f"{r.randint(1, 28):02d}/11 {heure}, flight {r.choice(VOLS)}.",
                f"{PAYS[pays]['en']} national, {DOCUMENTS_EN[doc]}.",
                f"{'He' if genre == 'm' else 'She'} {r.choice(MOTIFS_EN[motif])}.",
                (r.choice(ASILE_OUI_EN) if asile else r.choice(ASILE_NON_EN))]
    return " ".join(m for m in morceaux if m)


def tirer(r):
    pays = r.choice(list(PAYS))
    doc = r.choices(list(DOCUMENTS), weights=[6, 2, 1])[0]
    motif = r.choice(list(MOTIFS))
    # Plus de demandes d'asile parmi les Morvaniens : le conflit de Hadrel.
    asile = r.random() < {"Morvanie": 0.45, "Valdorie": 0.2, "Ostrane": 0.1}[pays]
    return pays, doc, motif, asile


def fiche(pays, doc, motif, asile):
    return {"nationalite": pays, "document": doc, "motif": motif, "demande_asile": asile}


def ecrire(chemin, lignes):
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lignes), encoding="utf-8")
    print(f"{chemin.relative_to(RACINE)} : {len(lignes)} lignes")


def main():
    r = random.Random(GRAINE)
    vues, mentions = set(), []
    while len(mentions) < 360:
        tirage = tirer(r)
        texte, nom = mention_fr(r, *tirage)
        if texte in vues:
            continue
        vues.add(texte)
        mentions.append({"id": f"M-{len(mentions) + 1:04d}", "langue": "fr", "nom": nom,
                         "entree": texte, "sortie": fiche(*tirage)})
    anglais = []
    while len(anglais) < 60:
        tirage = tirer(r)
        texte = mention_en(r, *tirage)
        if texte in vues:
            continue
        vues.add(texte)
        anglais.append({"id": f"E-{len(anglais) + 1:04d}", "langue": "en", "nom": None,
                        "entree": texte, "sortie": fiche(*tirage)})
    ecrire(SORTIE / "mentions" / "train-fr.jsonl", mentions[:300])
    ecrire(SORTIE / "mentions" / "eval-fr.jsonl", mentions[300:340])
    ecrire(SORTIE / "mentions" / "service-fr.jsonl", mentions[340:360])
    ecrire(SORTIE / "mentions" / "train-en.jsonl", anglais[:40])
    ecrire(SORTIE / "mentions" / "eval-en.jsonl", anglais[40:60])

    # Historiques de passages : trente passagers fictifs, plus P-0412, le cas du TP 3.
    passagers = [{"passager": "P-0412", "nationalite": "Valdorie",
                  "passages": [["2026-06-03", "2026-06-28"], ["2026-08-01", "2026-08-29"],
                               ["2026-10-05", "2026-10-20"]]},
                 {"passager": "P-0377", "nationalite": "Morvanie",
                  "passages": [["2026-01-01", "2026-03-31"]]}]
    for i in range(30):
        jour, sejours = date(2026, 1, 1) + timedelta(days=r.randint(0, 60)), []
        for _ in range(r.randint(1, 4)):
            duree = r.randint(3, 30)
            sejours.append([jour.isoformat(), (jour + timedelta(days=duree - 1)).isoformat()])
            jour += timedelta(days=duree + r.randint(10, 60))
            if jour > date(2026, 11, 1):
                break
        passagers.append({"passager": f"P-{500 + i:04d}", "nationalite": r.choice(list(PAYS)),
                          "passages": sejours})
    ecrire(SORTIE / "passages.jsonl", passagers)


if __name__ == "__main__":
    main()
