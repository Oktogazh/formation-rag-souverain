# %% [markdown]
# # TP 1 — Ce que le modèle ne sait pas
#
# **Durée** : ≈ 50 min · **Niveau** ★★☆☆☆ · **4 pannes** · indépendant des autres TP
#
# ## L'histoire
#
# Aéroport de Roissy-Charles-de-Gaulle, terminal 2E, un lundi de novembre 2026.
# Les vols arrivent des **Alpes méridionales**, une région de montagnes partagée
# entre trois pays : la **Valdorie**, la **Morvanie** et l'**Ostrane**.
#
# Depuis janvier 2026, la Valdorie et la Morvanie se battent pour une plaine, celle
# de Sarvik. La raison du conflit tient en une crue : en 1957, le fleuve Ivra, qui
# servait de frontière, a changé de lit de douze kilomètres. Chacun des deux pays
# revendique le tracé qui l'arrange. Les deux peuples parlent la même langue — ils
# lui donnent seulement deux noms — et chacun abrite une minorité de l'autre. Depuis
# le début des combats, de plus en plus de passagers venus de la région se présentent
# au guichet, et certains demandent l'asile.
#
# Les agents de la police aux frontières doivent, vite, retrouver la bonne règle
# d'entrée, la bonne procédure, et savoir ce qui se passe dans la province d'où vient
# le passager. Le service numérique de la direction veut leur construire un
# **assistant**. Il tournera **entièrement en local** : pas un mot de ces dossiers ne
# doit sortir du réseau.
#
# *Tout est fictif : les pays, les passagers, les règles et les procédures. Seul
# l'aéroport existe.*
#
# ## Ce que vous allez faire
#
# Avant de construire quoi que ce soit, il faut savoir **ce que le modèle sait seul**.
# Vous allez faire tourner deux modèles sur votre machine, de deux façons :
#
# 1. un tout petit modèle (135 millions de paramètres), **dans** le notebook, avec la
#    bibliothèque Transformers de Hugging Face ;
# 2. le modèle de la formation, `ministral-3:3b`, servi par **Ollama**, et interrogé
#    par l'interface compatible OpenAI.
#
# Puis vous lui poserez les questions d'un agent de Roissy, et vous noterez ce qui
# échoue. C'est cette liste qui justifie toute la suite de la formation.
#
# ## Comment travailler
#
# Le code de ce notebook est **complet, mais il est en panne à 4 endroits**. Une panne,
# c'est une mauvaise valeur, une étape oubliée, une variable inversée : le genre
# d'erreur que tout le monde fait en construisant ce type de système.
#
# - Exécutez les cellules **une par une**, dans l'ordre.
# - Chaque section qui contient une panne le dit, et se termine par une **cellule de
#   contrôle** : ✅ vous pouvez passer à la suite, ❌ la panne est au-dessus.
# - Certaines pannes font planter le code (un message d'erreur rouge) : lisez la
#   **dernière ligne** du message, c'est presque toujours elle qui dit tout. D'autres ne
#   plantent pas : le code tourne, mais le résultat est faux. Celles-là, seul le
#   contrôle les voit.
# - Bloqué ? Le corrigé complet est dans le dossier `corrige/`, à côté de ce notebook.
#
# Une fois une panne corrigée, **réexécutez la cellule corrigée, puis le contrôle**.

# %% [markdown]
# ## Préparation
#
# Cette cellule retrouve la racine du dépôt et importe le socle commun. Elle ne
# contient pas de panne.

# %%
import pathlib
import sys

RACINE = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "commun").is_dir())
sys.path.insert(0, str(RACINE))

from commun import corpus
from commun.controle import controle
from commun.regle_90_180 import dernier_jour_autorise

print("Racine du dépôt :", RACINE)

# %% [markdown]
# ## 1. Un modèle dans le notebook — *contient la panne 1*
#
# On charge `SmolLM2-135M-Instruct` avec Transformers. Il est minuscule : il tient
# dans la mémoire de n'importe quel ordinateur portable et répond en quelques secondes
# sur processeur. Il est aussi très limité, et c'est voulu : on voit tout ce qui se
# passe.
#
# Rappel du cours : un modèle de langue ne lit pas du texte. Il lit des **tokens**,
# c'est-à-dire des numéros, rangés dans un **tenseur** (un tableau de nombres que
# PyTorch sait multiplier). Le *tokenizer* fait la traduction dans les deux sens :
# texte → numéros à l'entrée, numéros → texte à la sortie.
#
# La cellule ci-dessous :
#
# 1. met la question au format de conversation que le modèle a appris (le *gabarit de
#    chat*) ;
# 2. transforme cette question en tokens et affiche les vingt premiers ;
# 3. demande au modèle de générer la suite, puis la retraduit en texte.

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

NOM_PETIT = "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = AutoTokenizer.from_pretrained(NOM_PETIT)
petit = AutoModelForCausalLM.from_pretrained(NOM_PETIT)

messages = [{"role": "user", "content": "Qu'est-ce qu'un passeport ?"}]
texte = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# <<<PANNE 1 | le modèle attend un tenseur PyTorch, pas une liste Python
entree = tok(texte, return_tensors="pt")
# ===
entree = tok(texte)
# >>>PANNE 1

print("Les 20 premiers tokens :", tok.convert_ids_to_tokens(entree["input_ids"][0])[:20])
sortie = petit.generate(**entree, max_new_tokens=60, do_sample=False)
reponse = tok.decode(sortie[0][entree["input_ids"].shape[1]:], skip_special_tokens=True)
print("\nRéponse :", reponse)

# %% [markdown]
# <details><summary>Indice 1 — où regarder</summary>
#
# Lisez la dernière ligne du message d'erreur : elle parle d'un objet `list` auquel il
# manque quelque chose. Quel objet est une liste, alors que le modèle attend autre chose ?
# </details>
#
# <details><summary>Indice 2 — quelle ligne</summary>
#
# La ligne `entree = tok(texte)` : par défaut, le tokenizer rend des **listes Python**.
# `generate` attend des **tenseurs PyTorch**. Un argument du tokenizer règle cela.
# </details>
#
# <details><summary>Indice 3 — la correction</summary>
#
# `entree = tok(texte, return_tensors="pt")` — `"pt"` pour PyTorch.
# </details>

# %%
controle(1, len(reponse.strip()) > 0, "le petit modèle n'a rien répondu.")

# %% [markdown]
# ## 2. Le budget de génération — *contient la panne 2*
#
# Un modèle génère **un token à la fois**, chacun choisi d'après tous les précédents.
# On lui fixe un budget : `max_new_tokens`, le nombre maximal de tokens à produire.
# Trop petit, la réponse s'arrête au milieu d'une phrase. Trop grand, on paie en temps
# de calcul pour rien (le modèle s'arrête de lui-même quand il a fini).
#
# La fonction ci-dessous pose une question au petit modèle et compte les tokens
# produits.

# %%
def demander_au_petit(question: str) -> tuple[str, int]:
    texte = tok.apply_chat_template([{"role": "user", "content": question}],
                                    tokenize=False, add_generation_prompt=True)
    entree = tok(texte, return_tensors="pt")
    # <<<PANNE 2 | un budget de 5 tokens coupe toute réponse au milieu d'un mot
    sortie = petit.generate(**entree, max_new_tokens=120, do_sample=False)
    # ===
    sortie = petit.generate(**entree, max_new_tokens=5, do_sample=False)
    # >>>PANNE 2
    nouveaux = sortie[0][entree["input_ids"].shape[1]:]
    return tok.decode(nouveaux, skip_special_tokens=True), len(nouveaux)


reponse, n_tokens = demander_au_petit("Qu'est-ce qu'un visa ?")
print(f"{n_tokens} tokens générés :\n{reponse}")

# %% [markdown]
# <details><summary>Indice 1 — où regarder</summary>
#
# Combien de tokens ont été générés ? Lisez la réponse : se termine-t-elle comme une
# phrase ?
# </details>
#
# <details><summary>Indice 2 — quelle ligne</summary>
#
# L'appel à `petit.generate` dans `demander_au_petit` : son budget est trop petit.
# </details>
#
# <details><summary>Indice 3 — la correction</summary>
#
# `max_new_tokens=120`. Réexécutez la cellule de la fonction, puis le contrôle.
# </details>

# %%
controle(2, n_tokens > 20, f"la réponse ne fait que {n_tokens} tokens : elle est coupée.")

# %% [markdown]
# ## 3. Le vrai modèle, servi par Ollama — *contient la panne 3*
#
# Le petit modèle tourne **dans** le notebook. En production, c'est l'inverse : le
# modèle tourne dans un **serveur** à part (ici Ollama, plus tard peut-être un serveur
# d'inférence externe mais souverain), et les programmes l'interrogent par le réseau.
#
# Presque tous ces serveurs parlent la même langue : l'**interface compatible
# OpenAI**. On s'en sert avec la bibliothèque `openai`, mais **rien ne part chez
# OpenAI** : on lui donne simplement l'adresse de notre serveur local. Changer de
# serveur, c'est changer cette adresse, et rien d'autre.
#
# Ollama sert cette interface sous un chemin précis de son adresse. Il faut lui donner
# le bon.

# %%
from openai import OpenAI

# <<<PANNE 3 | l'interface compatible OpenAI d'Ollama est servie sous le chemin /v1
client = OpenAI(base_url="http://localhost:11434/v1", api_key="local")
# ===
client = OpenAI(base_url="http://localhost:11434", api_key="local")
# >>>PANNE 3
MODELE = "ministral-3:3b"


def demander(question: str, temperature: float = 0.0, max_tokens: int = 250) -> str:
    r = client.chat.completions.create(model=MODELE, temperature=temperature, max_tokens=max_tokens,
                                       messages=[{"role": "user", "content": question}])
    return r.choices[0].message.content


reponse = demander("En une phrase : qu'est-ce que l'espace Schengen ?")
print(reponse)

# %% [markdown]
# <details><summary>Indice 1 — où regarder</summary>
#
# L'erreur dit `404 Not Found` : le serveur répond, mais pas à cette adresse.
# </details>
#
# <details><summary>Indice 2 — quelle ligne</summary>
#
# La création du `client` : l'adresse `base_url` est incomplète. L'interface
# compatible OpenAI d'Ollama n'est pas à la racine du serveur.
# </details>
#
# <details><summary>Indice 3 — la correction</summary>
#
# `base_url="http://localhost:11434/v1"`.
# </details>

# %%
controle(3, bool(reponse and reponse.strip()), "le modèle servi par Ollama n'a rien répondu.")

# %% [markdown]
# ### Où sont passés les tokens ?
#
# Ici, plus de tokenizer ni de tenseur : on envoie du texte, on reçoit du texte. La
# tokenisation n'a pas disparu, elle se fait **dans le serveur**. Le seul endroit où
# elle reparaît côté client, c'est le champ `usage` de la réponse : le nombre de tokens
# lus (`prompt_tokens`) et produits (`completion_tokens`). C'est aussi ce que comptent
# les fournisseurs qui facturent à l'usage.

# %%
r = client.chat.completions.create(model=MODELE, temperature=0, max_tokens=250,
                                   messages=[{"role": "user", "content": "En une phrase : qu'est-ce que l'espace Schengen ?"}])
print(r.usage)

# %% [markdown]
# ## 4. La température — *contient la panne 4*
#
# À chaque token, le modèle calcule une probabilité pour chaque token possible. La
# **température** règle la façon de choisir : à 0, on prend toujours le plus
# probable, et la même question donne toujours la même réponse ; plus elle monte,
# plus le choix devient aléatoire.
#
# Pour **mesurer**, il faut pouvoir refaire une expérience et retrouver le même
# résultat. Dans toute la formation, on travaille donc à température 0. La cellule
# ci-dessous pose trois fois la même question.

# %%
# <<<PANNE 4 | pour mesurer, la même question doit toujours donner la même réponse
TEMPERATURE = 0.0
# ===
TEMPERATURE = 1.5
# >>>PANNE 4

# Appel d'échauffement : Ollama garde en cache le début du prompt, et la toute première
# réponse peut différer des suivantes. On la jette pour que les essais soient comparables.
demander("Cite trois documents de voyage.", temperature=0.0, max_tokens=60)

essais = [demander("Cite trois documents de voyage.", temperature=TEMPERATURE, max_tokens=60)
          for _ in range(3)]
for n, e in enumerate(essais, 1):
    print(f"--- essai {n}\n{e}")

# %% [markdown]
# <details><summary>Indice 1 — où regarder</summary>
#
# Les trois réponses sont-elles identiques ?
# </details>
#
# <details><summary>Indice 2 — quelle ligne</summary>
#
# La valeur de `TEMPERATURE`.
# </details>
#
# <details><summary>Indice 3 — la correction</summary>
#
# `TEMPERATURE = 0.0`. À température 0, le modèle choisit toujours le token le plus
# probable : la réponse ne varie plus.
# </details>

# %%
controle(4, len(set(essais)) == 1, "les trois réponses diffèrent : impossible de mesurer quoi que ce soit.")

# %% [markdown]
# ## 5. Les questions d'un agent de Roissy
#
# Plus de panne : le notebook fonctionne. Place à l'observation, l'objet même de ce TP.
#
# Voici huit questions qu'un agent pourrait poser à l'assistant. Pour chacune, on a
# noté ce qui manquerait au modèle pour répondre juste :
#
# - **document** : la réponse est écrite quelque part (une règle, une procédure, un
#   rapport), mais le modèle ne l'a jamais lue. Les pays sont fictifs : il ne peut
#   pas les connaître.
# - **calcul** : la réponse demande un calcul exact, ici la règle des 90 jours sur 180.
#
# La cellule pose les huit questions au modèle. Lisez les réponses : le modèle
# avoue-t-il qu'il ne sait pas, ou invente-t-il ? *(Compter 1 à 2 minutes.)*

# %%
questions = corpus.questions("sans-recherche")
reponses = {}
for q in questions:
    reponses[q["id"]] = demander(q["question"])
    print(f"\n### {q['id']} · il manque : {q['manque']}\n{q['question']}\n→ {reponses[q['id']]}")

# %% [markdown]
# ### Les deux calculs, faits juste
#
# Les questions S07 et S08 portent sur la règle des 90 jours sur 180. La règle, en
# deux phrases : un voyageur sans visa long peut rester au plus 90 jours sur toute
# période de 180 jours dans l'espace Schengen ; et **chaque jour** de présence, on
# regarde les 180 jours qui se terminent ce jour-là, on compte les jours passés dans
# l'espace Schengen, et le total ne doit jamais dépasser 90.
#
# La fenêtre **glisse** : pendant le séjour, les jours anciens en sortent et libèrent
# de la place. C'est pourquoi le calcul « 90 moins les jours déjà passés » se trompe.
# Voici le résultat d'un calcul exact, jour par jour (le code est dans
# `commun/regle_90_180.py`, une quinzaine de lignes) :

# %%
from datetime import date

s07 = [(date(2026, 6, 3), date(2026, 6, 28)), (date(2026, 8, 1), date(2026, 8, 29)),
       (date(2026, 10, 5), date(2026, 10, 20))]
print("S07 — dernier jour autorisé :", dernier_jour_autorise(s07, date(2026, 11, 14)))
s08 = [(date(2026, 1, 1), date(2026, 3, 31))]
print("S08 — entrée le 29 juin :", dernier_jour_autorise(s08, date(2026, 6, 29)) or "refusée (91 jours)")
print("S08 — entrée le 30 juin :", dernier_jour_autorise(s08, date(2026, 6, 30)))

# %% [markdown]
# ## À retenir
#
# - Un modèle de langue lit des **tokens**, génère **un token à la fois**, dans un
#   **budget** que l'on fixe.
# - Le modèle tourne dans un **serveur** ; on lui parle par l'**interface compatible
#   OpenAI**, en local. Changer de serveur, c'est changer d'adresse.
# - **Température 0** pour mesurer.
# - Seul, le modèle échoue de **deux façons** différentes, qui appellent deux
#   remèdes différents :
#   - il lui manque un **document** → il faut aller le chercher et le lui donner :
#     c'est le **RAG** (TP 2) ;
#   - il lui manque un **calcul** → il faut lui donner un outil et le laisser
#     l'appeler : c'est l'**agent** (TP 3).
# - Le plus inquiétant n'est pas qu'il se trompe : c'est qu'il se trompe **avec
#   assurance**, sans dire qu'il ne sait pas.
