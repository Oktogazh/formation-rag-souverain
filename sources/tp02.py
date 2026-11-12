# %% [markdown]
# # TP 2 — Le RAG : donner au modèle les bons documents
#
# **Durée** : ≈ 75 min · **Niveau** ★★★☆☆ · **4 pannes** · indépendant des autres TP
#
# ## L'histoire
#
# Au TP 1, le modèle a répondu avec aplomb à des questions sur la Valdorie, la
# Morvanie et l'Ostrane… des pays qu'il ne pouvait pas connaître. Ce qu'il lui
# manquait, c'était la **documentation** dont disposent les agents de Roissy :
#
# - les **règles d'entrée**, une fiche par pays, plus les règles générales (la règle
#   des 90 jours sur 180, les documents exigés) ;
# - les **procédures** de la police aux frontières (contrôle approfondi, zone
#   d'attente, demande d'asile à la frontière, interprétariat…) ;
# - la **documentation sur les pays d'origine** : des rapports datés sur la situation
#   de chaque province, de 2023 à 2026, dont deux en anglais.
#
# Ces rapports racontent un conflit qui s'aggrave. En 2023, la province morvanienne
# de Hadrel est calme et le pont de Sarvik est ouvert. Fin 2025, le pont ferme et
# les arrestations commencent. En mars 2026, l'armée morvanienne affronte la Garde
# de la plaine, une milice de la minorité valdorienne, et 25 000 personnes ont fui.
# **Un même lieu, trois réalités, selon la date du rapport.** Retenez-le : c'est le
# piège de la section 4.
#
# ## Ce que vous allez faire
#
# Construire, à la main et en entier, la chaîne d'un **RAG** (*retrieval-augmented
# generation*, génération augmentée par la recherche) :
#
# 1. transformer chaque morceau de document en **vecteur** ;
# 2. **chercher** les morceaux les plus proches d'une question ;
# 3. **filtrer** ceux qui sont périmés ;
# 4. les **donner au modèle** dans le prompt, et lui demander de citer ses sources.
#
# Puis mesurer : sur 20 questions, combien de fois la bonne source est-elle dans les
# trois premiers résultats ?
#
# ## Comment travailler
#
# Le code est **complet, mais en panne à 4 endroits**. Exécutez les cellules une par
# une ; chaque section qui contient une panne se termine par un **contrôle** ✅ / ❌.
# Après une correction, réexécutez la cellule corrigée, puis le contrôle.

# %% [markdown]
# ## Préparation

# %%
import pathlib
import sys

RACINE = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "commun").is_dir())
sys.path.insert(0, str(RACINE))

import numpy as np
from openai import OpenAI

from commun import corpus
from commun.controle import controle

client = OpenAI(base_url="http://localhost:11434/v1", api_key="local")
MODELE, ENCODEUR = "ministral-3:3b", "bge-m3"
questions = corpus.questions("rag")
print(f"{len(questions)} questions d'évaluation")

# %% [markdown]
# ## 1. Les morceaux — un découpage fait en amont
#
# On ne donne pas un document entier au modèle : on le découpe en **morceaux**
# (*chunks*), et l'on ne donne que les morceaux utiles. Le cours a présenté quatre
# façons de découper :
#
# | Stratégie | Principe | Quand |
# |---|---|---|
# | **Taille fixe** | tous les *n* tokens | texte sans structure ; simple, mais coupe au milieu des idées |
# | **Taille fixe avec chevauchement** | chaque morceau reprend la fin du précédent | limite les coupures malheureuses, au prix de doublons |
# | **Par balises** (titres Markdown, balises HTML) | un morceau par section | documents structurés par leur auteur |
# | **Sémantique** | on coupe là où le sens change, mesuré par des vecteurs | longs textes continus, sans structure |
#
# **Ce TP ne vous fait pas découper : c'est déjà fait**, dans `commun/corpus.py`, et
# voici pourquoi on a choisi le découpage **par balises**. Nos documents sont courts,
# et leurs auteurs les ont déjà organisés : un titre `##` par province dans les
# rapports, un titre `##` par règle dans les fiches. Une section porte une seule idée.
# Couper à taille fixe mélangerait deux provinces dans un même morceau.
#
# Un détail compte : le **titre du document est recopié en tête de chaque morceau**.
# Sans lui, la section « Province de Hadrel » ne dirait ni de quel pays, ni de quelle
# date elle parle. Chaque morceau garde aussi ses **métadonnées** : pays, date,
# langue. Elles serviront à filtrer.
#
# Sur un vrai fonds documentaire — des PDF de cent pages, des procès-verbaux sans
# titres —, le choix du découpage serait un travail à part entière, à mesurer comme
# on mesure ici la recherche.

# %%
morceaux = corpus.decouper(corpus.charger_documents())
print(f"{len(morceaux)} morceaux, tirés de {len({m['doc'] for m in morceaux})} documents\n")
print(morceaux[20]["id"], "·", morceaux[20]["date"], "·", morceaux[20]["langue"])
print(morceaux[20]["texte"])

# %% [markdown]
# ## 2. Vectoriser
#
# Chaque morceau devient un **vecteur** : une liste de 1 024 nombres calculée par un
# *encodeur*, ici `bge-m3`, qui tourne dans Ollama. Deux textes qui parlent de la
# même chose ont des vecteurs qui pointent dans la même direction.
#
# Pourquoi `bge-m3` ? Parce qu'il est **multilingue** : une question en français et
# un rapport en anglais sur le même sujet tombent au même endroit. Un encodeur
# entraîné sur une seule langue ne retrouverait jamais les deux rapports anglais de
# notre corpus, et rien ne le signalerait.
#
# On ramène chaque vecteur à une longueur de 1. Ainsi, le produit scalaire de deux
# vecteurs est le **cosinus** de leur angle : 1 pour deux textes identiques, autour
# de 0 pour deux textes sans rapport. Pas de panne dans cette section.

# %%
def vectoriser(textes: list[str]) -> np.ndarray:
    reponse = client.embeddings.create(model=ENCODEUR, input=textes)
    vecteurs = np.array([d.embedding for d in reponse.data], dtype="float32")
    return vecteurs / np.linalg.norm(vecteurs, axis=1, keepdims=True)


matrice = vectoriser([m["texte"] for m in morceaux])
print("Forme de la matrice :", matrice.shape, "— une ligne par morceau, une colonne par dimension")

# %% [markdown]
# ## 3. Chercher — *contient les pannes 1 et 2*
#
# Chercher, c'est trois gestes :
#
# 1. vectoriser la question, avec **le même encodeur** que les morceaux ;
# 2. calculer son cosinus avec chaque morceau : un seul produit matrice × vecteur ;
# 3. trier les morceaux du plus proche au plus lointain, et garder les *k* premiers.
#
# C'est exactement ce que fait une base vectorielle comme FAISS (citée au programme),
# avec des astuces pour aller vite sur des millions de morceaux. Sur 46 morceaux,
# `numpy` suffit, et tout reste visible. Les TP suivants font de même.
#
# Le paramètre `filtre` servira à la section 4 : il ne fait rien pour l'instant.

# %%
def chercher(question: str, k: int = 3, filtre=None) -> list[dict]:
    # <<<PANNE 1 | vectoriser() rend une matrice d'une ligne ; il faut le vecteur de cette ligne
    q = vectoriser([question])[0]
    # ===
    q = vectoriser([question])
    # >>>PANNE 1
    scores = matrice @ q
    # <<<PANNE 2 | trier du plus grand cosinus au plus petit
    ordre = np.argsort(-scores)
    # ===
    ordre = np.argsort(scores)
    # >>>PANNE 2
    retenus = [{**morceaux[i], "score": float(scores[i])} for i in ordre
               if filtre is None or filtre(morceaux[i])]
    return retenus[:k]


resultats = chercher("Quel poste appeler pour obtenir un interprète ?")
for m in resultats:
    print(f"{m['score']:.3f}  {m['id']}")

# %% [markdown]
# <details><summary>Indice 1 — où regarder</summary>
#
# L'erreur parle de dimensions qui ne correspondent pas (`matmul`, `mismatch`).
# Affichez `vectoriser(["test"]).shape` : combien de dimensions a ce tableau ?
# </details>
#
# <details><summary>Indice 2 — quelle ligne</summary>
#
# `vectoriser` rend une **matrice** (une ligne par texte). Pour une seule question,
# on veut la **ligne**, c'est-à-dire un vecteur.
# </details>
#
# <details><summary>Indice 3 — la correction</summary>
#
# `q = vectoriser([question])[0]`
# </details>

# %%
controle(1, len(resultats) == 3, "la recherche ne rend pas trois morceaux.")

# %% [markdown]
# La recherche tourne. Mais rend-elle les **bons** morceaux ? On mesure : pour chaque
# question sans contrainte de date, la bonne source
# est-elle parmi les trois premiers résultats ? C'est le **rappel à k=3**.

# %%
def rappel(questions: list[dict], filtre_de=lambda q: None) -> int:
    return sum(any(m["doc"] in q["documents_attendus"] for m in chercher(q["question"], 3, filtre_de(q)))
               for q in questions)


sans_date = [q for q in questions if not q["date_min"]]
trouvees = rappel(sans_date)
print(f"Bonne source dans les 3 premiers : {trouvees} / {len(sans_date)}")

# %% [markdown]
# <details><summary>Indice 1 — où regarder</summary>
#
# Le code tourne, mais le rappel est catastrophique. Remontez aux scores affichés
# pour la question sur l'interprète : les premiers sont-ils les plus grands ?
# </details>
#
# <details><summary>Indice 2 — quelle ligne</summary>
#
# `np.argsort` trie du plus **petit** au plus grand. On ramène donc les morceaux les
# **moins** proches de la question.
# </details>
#
# <details><summary>Indice 3 — la correction</summary>
#
# `ordre = np.argsort(-scores)` : trier les scores négatifs revient à trier du plus
# grand au plus petit.
# </details>

# %%
controle(2, trouvees >= 13, f"la bonne source n'est dans les 3 premiers que {trouvees} fois sur {len(sans_date)}.",
         f"Rappel à 3 : {trouvees}/{len(sans_date)}")

# %% [markdown]
# ## 4. Filtrer ce qui est périmé — *contient la panne 3*
#
# Voici le piège annoncé. Posons une question simple sur la province de Hadrel :

# %%
for m in chercher("La province de Hadrel est-elle calme ?", k=3):
    print(f"{m['score']:.3f}  {m['id']:<24} {m['date']}  {m['langue']}")

# %% [markdown]
# Le morceau le plus proche est le rapport de **juin 2023**, en anglais, qui dit que
# la province est calme. Sémantiquement, c'est la meilleure réponse : il parle
# exactement de calme à Hadrel. Mais il a trois ans, et la province est en guerre.
#
# La similarité ne sait rien du **temps**. Il faut le lui dire, avec les
# **métadonnées** : on ne garde que les morceaux datés après une date donnée. La
# fonction `apres(date_min)` fabrique ce filtre ; les dates sont au format
# AAAA-MM-JJ, qui se compare comme du texte.

# %%
def apres(date_min: str):
    # <<<PANNE 3 | garder ce qui est postérieur à date_min, pas ce qui est antérieur
    return lambda m: m["date"] >= date_min
    # ===
    return lambda m: m["date"] <= date_min
    # >>>PANNE 3


for m in chercher("La province de Hadrel est-elle calme ?", k=3, filtre=apres("2026-01-01")):
    print(f"{m['score']:.3f}  {m['id']:<24} {m['date']}  {m['langue']}")

avec_date = [q for q in questions if q["date_min"]]
a_jour = all(m["date"] >= q["date_min"] for q in avec_date
             for m in chercher(q["question"], 3, apres(q["date_min"])))
total = rappel(questions, lambda q: apres(q["date_min"]) if q["date_min"] else None)
print(f"\nRappel à 3 sur les 20 questions, filtre compris : {total} / 20")

# %% [markdown]
# <details><summary>Indice 1 — où regarder</summary>
#
# Après filtrage, quelles dates voyez-vous ? Sont-elles après le 1er janvier 2026 ?
# </details>
#
# <details><summary>Indice 2 — quelle ligne</summary>
#
# La comparaison dans `apres` : elle garde les morceaux **antérieurs** à la date.
# </details>
#
# <details><summary>Indice 3 — la correction</summary>
#
# `return lambda m: m["date"] >= date_min`
# </details>

# %%
controle(3, a_jour and total >= 18, "le filtre laisse passer des documents périmés, ou écarte les bons.",
         f"Rappel à 3 : {total}/20")

# %% [markdown]
# ## 5. Le prompt augmenté — *contient la panne 4*
#
# Dernière étape : donner au modèle les morceaux trouvés, et lui demander de
# répondre **à partir d'eux seuls**, en **citant** l'identifiant de chaque source.
# La citation est ce qui rend la réponse vérifiable par l'agent : il peut ouvrir la
# fiche et lire lui-même.
#
# Ici, consigne, documents et question partent ensemble dans **un seul message
# `user`**, pour que vous voyiez le prompt entier d'un coup. En production, on les
# sépare d'habitude : la consigne, qui ne change jamais, en message `system` ; les
# documents et la question, qui changent à chaque fois, en message `user`. Les
# documents sont des **données**, pas des consignes : si l'un d'eux contenait « ignore
# tes instructions », mieux vaut qu'il ne pèse pas comme une consigne système.

# %%
def prompt_augmente(question: str, trouves: list[dict]) -> str:
    contexte = "\n\n".join(f"[{m['id']}, {m['date']}]\n{m['texte']}" for m in trouves)
    # <<<PANNE 4 | sans le préfixe f, Python n'insère pas les variables dans le texte
    return f"""Tu assistes un agent de la police aux frontières. Réponds à la question
en t'appuyant uniquement sur les documents ci-dessous, et cite entre crochets
l'identifiant de chaque document utilisé. Si les documents ne permettent pas de
répondre, dis-le.

Documents :
{contexte}

Question : {question}"""
    # ===
    return """Tu assistes un agent de la police aux frontières. Réponds à la question
en t'appuyant uniquement sur les documents ci-dessous, et cite entre crochets
l'identifiant de chaque document utilisé. Si les documents ne permettent pas de
répondre, dis-le.

Documents :
{contexte}

Question : {question}"""
    # >>>PANNE 4


def rag(question: str, filtre=None) -> str:
    p = prompt_augmente(question, chercher(question, 3, filtre))
    r = client.chat.completions.create(model=MODELE, temperature=0, max_tokens=300,
                                       messages=[{"role": "user", "content": p}])
    return r.choices[0].message.content


exemple = prompt_augmente("Qui décide d'une non-admission ?", chercher("Qui décide d'une non-admission ?"))
print(exemple)

# %% [markdown]
# <details><summary>Indice 1 — où regarder</summary>
#
# Lisez le prompt affiché : voyez-vous le texte des documents, ou autre chose à la
# place ?
# </details>
#
# <details><summary>Indice 2 — quelle ligne</summary>
#
# Le `return` de `prompt_augmente` : `{contexte}` et `{question}` apparaissent tels
# quels. Le modèle reçoit des accolades, pas des documents.
# </details>
#
# <details><summary>Indice 3 — la correction</summary>
#
# Ajouter le préfixe `f` devant la chaîne : `return f"""Tu assistes…`. C'est une
# **f-string** : Python y remplace `{contexte}` par la valeur de la variable.
# </details>

# %%
controle(4, "Non-admission" in exemple and "{contexte}" not in exemple,
         "le prompt ne contient pas les documents trouvés.")

# %% [markdown]
# ## 6. Avec et sans RAG
#
# Plus de panne. Comparons, sur trois questions, la réponse du modèle seul et la
# réponse avec RAG. *(Compter une minute.)*

# %%
def seul(question: str) -> str:
    r = client.chat.completions.create(model=MODELE, temperature=0, max_tokens=300,
                                       messages=[{"role": "user", "content": question}])
    return r.choices[0].message.content


for question, filtre in [("Un Ostranien titulaire d'un passeport de service doit-il présenter un visa ?", None),
                         ("Un passager en zone d'attente demande l'asile : qui faut-il prévenir ?", None),
                         ("La province de Hadrel est-elle calme ?", apres("2026-01-01"))]:
    print(f"\n### {question}\n— Modèle seul :\n{seul(question)}\n\n— Avec RAG :\n{rag(question, filtre)}")

# %% [markdown]
# ## À retenir
#
# - Un RAG, c'est **chercher puis donner** : vectoriser, comparer, trier, filtrer,
#   insérer dans le prompt. Chaque geste peut casser **sans message d'erreur**.
# - Le **découpage** est un choix, fait en amont : ici par sections, parce que nos
#   documents sont structurés. Sur des textes continus, il faudrait le mesurer.
# - Le **même encodeur** pour les documents et pour la question ; un encodeur
#   **multilingue** si le fonds mélange les langues.
# - La similarité ignore le temps : les **métadonnées** (date, pays) font le reste.
# - **Citer les sources** rend la réponse vérifiable. L'assistant prépare, l'agent
#   vérifie et décide.
