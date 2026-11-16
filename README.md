# RAG et fine-tuning d'un LLM — les travaux pratiques

Formation **« RAG et fine-tuning d'un LLM »** (PLB, 3 jours). Six TP, un par
demi-journée, **chacun réalisable seul** : rater un TP ne vous empêche jamais de
faire le suivant.

## Synopsis

Aéroport de **Roissy-Charles-de-Gaulle**, novembre 2026. Les vols arrivent des
**Alpes méridionales**, une région partagée entre trois pays : la Valdorie, la
Morvanie et l'Ostrane. La Valdorie et la Morvanie se battent pour la plaine de
Sarvik, depuis qu'une crue a déplacé en 1957 le fleuve qui leur servait de
frontière. Les deux peuples parlent la même langue, sous deux noms, et chacun
abrite une minorité de l'autre.

La police aux frontières veut un **assistant** : retrouver la bonne règle d'entrée,
la bonne procédure, ce que l'on sait de la province d'où vient un passager ; tirer
une fiche des mentions de contrôle rédigées au guichet ; calculer la durée de séjour
autorisée. **L'assistant prépare, l'agent décide.**

**Tout est fictif** — pays, passagers, règles, procédures — sauf l'aéroport. Et
**tout tourne sur votre machine** : modèles à poids ouverts servis par Ollama,
fine-tuning sur processeur, MLflow en local. Aucun service en ligne, aucun GPU.

## Installation

```bash
git clone <adresse du dépôt>
cd <dossier du dépôt>
conda env create -f environment.yml
conda activate formation-rag-souverain
python -m ipykernel install --user --name formation-rag-souverain --display-name "Formation RAG souverain"
ollama pull ministral-3:3b
ollama pull bge-m3
python tp.py check
```

**Envoyez la sortie de `python tp.py check` au formateur** avant la session. Prévoyez
**8 Go** d'espace libre.

## Les six TP

Ouvrez le notebook du jour avec `jupyter lab`, noyau **Formation RAG souverain**.

| Dossier | Demi-journée | Ce que vous faites | Pannes |
|---|---|---|---|
| `tp01-modele/` | J1 matin | faire tourner un modèle local ; voir ce qu'il ne sait pas | 4 |
| `tp02-rag/` | J1 après-midi | construire un RAG à la main : vecteurs, recherche, filtre de date, prompt augmenté | 4 |
| `tp03-agent/` | J2 matin | un agent ReAct avec deux outils : la documentation et le calcul des 90/180 jours | 3 |
| `tp04-fine-tuning/` | J2 après-midi | apprendre à un petit modèle à extraire une fiche d'une mention, avec LoRA | 4 |
| `tp05-service/` | J3 matin | mettre l'assistant en service sur votre machine : API, contrat, pic de charge | 4 |
| `tp06-mlflow/` | J3 après-midi | comparer deux versions du modèle dans MLflow ; RAG ou fine-tuning ; la chaîne complète | 4 |

## Comment se fait un TP : réparer un notebook en panne

Chaque notebook est **complet** : tout le code est écrit. Mais il est **en panne** à
trois ou quatre endroits, annoncés en tête. Une panne, c'est une mauvaise valeur,
une étape oubliée, une variable inversée : les erreurs que tout le monde fait en
construisant ce genre de système.

- Exécutez les cellules **dans l'ordre**.
- Après chaque section en panne, une **cellule de contrôle** affiche ✅ (passez à la
  suite) ou ❌ (la panne est au-dessus).
- Certaines pannes font planter le code : lisez la **dernière ligne** de l'erreur.
  D'autres ne plantent pas : le code tourne et le résultat est faux. Seul le contrôle
  les voit.
- Chaque panne a **trois indices** repliés, de plus en plus précis.
- Le **corrigé** est dans le dossier `corrige/` de chaque TP. À la fin de la
  formation, `python tp.py corriger` remplace les six notebooks par leur version
  réparée (les vôtres sont d'abord mises de côté dans `resultats/mes-notebooks/`).

## Le dépôt

```
data/roissy/        le corpus fictif : documents, mentions de contrôle, passages, questions
commun/             le socle partagé par les TP — à ne pas modifier
tpNN-*/             un notebook par TP, et son corrigé dans corrige/
resultats/          ce que les TP produisent (index, adaptateurs, base MLflow) — non versionné
sources/, outils/   pour le formateur : la source unique de chaque TP, la fabrique, le vérificateur
```

## Scripts

Chaque `sources/tpNN.py` est la **source unique** d'un TP : le code juste, et pour
chaque panne sa version cassée, entre des marqueurs `# <<<PANNE n` / `# ===` /
`# >>>PANNE n`.

```bash
python outils/fabriquer.py             # refabrique les 6 notebooks et leurs corrigés
python outils/verifier.py tp02         # exécute le corrigé, puis chaque panne seule
python scripts/fabriquer_mentions.py   # régénère les mentions et les passages (déterministe)
```

`verifier.py` exécute les notebooks pour de vrai, modèles compris, et vérifie que le
corrigé passe tous ses contrôles et que **la panne n, seule, arrête le notebook au
contrôle n** — ni avant, ni après.
