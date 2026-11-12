# %% [markdown]
# # TP 3 — L'agent : laisser le modèle choisir ses outils
#
# **Durée** : ≈ 60 min · **Niveau** ★★★☆☆ · **3 pannes** · indépendant des autres TP
#
# ## L'histoire
#
# Lundi 14 novembre 2026, 8 h 42, terminal 2E. Le passager **P-0412**, de nationalité
# valdorienne, présente son passeport au guichet. La Valdorie est exemptée de visa,
# mais l'agent voit dans l'historique trois séjours en 2026 : 26 jours en juin,
# 29 en août, 16 en octobre. **Jusqu'à quand peut-il rester ?**
#
# Au TP 1, le modèle seul s'est trompé sur ce calcul. Un RAG (TP 2) ne l'aiderait
# pas davantage : la fiche « règle des 90 jours sur 180 » explique la règle, mais
# l'appliquer à trois séjours demande un calcul jour par jour, que le modèle fait
# mal de tête. La bonne réponse est le 28 décembre ; les modèles locaux, sondés en
# préparant la formation, répondaient le 13 mars 2027 ou le 20 octobre 2026.
#
# Ce qu'il faut, c'est un assistant qui **sache quand chercher dans la documentation
# et quand appeler une calculatrice**. C'est un **agent**.
#
# ## Ce que vous allez faire
#
# Un agent, c'est un modèle de langue dans une **boucle** (la logique **ReAct**,
# pour *reason + act*) :
#
# 1. le modèle lit la question et **raisonne** : de quoi ai-je besoin ?
# 2. il **agit** : il demande l'appel d'un outil, avec des arguments ;
# 3. le programme exécute l'outil et lui rend le résultat (l'**observation**) ;
# 4. le modèle recommence, jusqu'à pouvoir répondre.
#
# Le modèle n'exécute rien lui-même : il **choisit**. Et il choisit en lisant une
# seule chose, la **description** de chaque outil. Vous allez donner deux outils à
# l'agent — la documentation de la police aux frontières (le RAG, déjà construit) et
# le calcul de la règle des 90 jours sur 180 — puis lire, dans la trace, ce qu'il en
# fait.
#
# Un mot sur cette documentation. Dans la réalité, ses sources seraient variées. Une
# partie pourrait venir des **télégrammes diplomatiques** des ambassades de la région :
# ils sont toujours rédigés en français, et ils sortent toujours à peu près dans le même
# format (en-tête, objet, résumé, développement). Une source aussi régulière se découpe
# facilement : on coupe sur la structure, à chaque fois de la même façon. Mais
# l'assistant recevra aussi d'autres sources (rapports, notes, messages, documents
# présentés par les passagers), moins régulières, parfois dans une autre langue. Le
# découpage et la recherche doivent tenir compte de cette variété, et pas seulement du
# cas le plus simple.
#
# ## Comment travailler
#
# Le code est **complet, mais en panne à 3 endroits**. Exécutez les cellules une par
# une ; chaque contrôle dit ✅ ou ❌. **Attention** : dans ce TP, une panne peut se trouver plus haut que
# l'endroit où elle se voit. C'est le cas le plus fréquent avec les agents.

# %% [markdown]
# ## Préparation
#
# On charge la documentation de la police aux frontières, déjà indexée. La
# **recherche** est la même qu'au TP 2 : la version de référence de `commun/recherche.py`,
# en numpy, sans bibliothèque spécialisée. Vous n'avez pas besoin d'avoir fait le TP 2.
# Le découpage des documents a été fait en amont, par section (un titre `##` = un
# morceau).
#
# Ce qui change, c'est **ce qui est autour** : la recherche n'est plus appelée par
# notre code, elle devient un **outil** que l'agent décide d'appeler. Et cet agent, on
# ne l'écrit pas à la main : on le construit avec **LangChain**.
#
# ### LangChain, LangGraph : ce qu'on importe ci-dessous
#
# - **LangChain** est une bibliothèque d'**assemblage**. Elle ne contient aucun modèle :
#   elle donne une interface commune aux pièces d'une application de LLM (modèles de
#   chat, outils, prompts, messages), pour qu'on puisse en changer sans réécrire le
#   reste. Elle n'est pas indispensable : tout ce qu'elle fait ici s'écrit en Python
#   pur, mais il faudrait écrire et maintenir la boucle soi-même.
# - **LangGraph**, de la même équipe, exécute des **graphes** : des étapes (des
#   *nœuds*) reliées par des transitions, avec un **état** qui passe de l'une à
#   l'autre. Un agent ReAct est un petit graphe à deux nœuds, « appeler le modèle » et
#   « exécuter les outils », qui boucle tant que le modèle demande un outil.
#   `create_agent` (LangChain) fabrique ce graphe pour nous.
# - **`ChatOllama`** est le connecteur LangChain vers Ollama. Il parle l'API propre
#   d'Ollama, pas l'interface compatible OpenAI du TP 1. On aurait pu prendre
#   `ChatOpenAI(base_url="http://localhost:11434/v1", …)` : le reste du TP serait
#   identique. C'est ce que permet LangChain : changer de connecteur sans toucher à
#   l'agent.
# - **`@tool`** transforme une fonction Python en outil présentable au modèle (voir
#   la section 1).

# %%
import pathlib
import sys

RACINE = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "commun").is_dir())
sys.path.insert(0, str(RACINE))

from datetime import date

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from commun import corpus
from commun.controle import controle
from commun.recherche import Documentation
from commun.regle_90_180 import dernier_jour_autorise

documentation = Documentation()
historiques = corpus.passages()
print(f"{len(documentation.morceaux)} morceaux indexés, {len(historiques)} historiques de passagers")

# %% [markdown]
# ## 1. Les deux outils — *contient les pannes 1 et 3*
#
# Un outil, c'est une fonction Python ordinaire, décorée par `@tool`. Le décorateur
# en tire ce que le modèle verra : le **nom** de la fonction, ses **arguments**, et
# sa **description**, c'est-à-dire sa *docstring* (le texte entre triples guillemets
# sous la ligne `def`).
#
# Le modèle ne voit jamais le code de l'outil. **Il ne voit que sa description.** Une
# description absente, vague ou trompeuse, et le modèle ne saura pas quand
# s'en servir.
#
# Le second outil va chercher lui-même l'historique du passager : l'agent n'a qu'un
# identifiant et une date à lui passer. Moins l'agent a d'étapes à enchaîner, moins
# il se trompe. L'historique est une liste de séjours, chacun noté
# `[date d'entrée, date de sortie]` ; le calcul jour par jour est dans
# `commun/regle_90_180.py`.

# %%
# <<<PANNE 1 | sans description, @tool ne sait pas présenter l'outil au modèle
@tool
def chercher_documentation(question: str) -> str:
    """Cherche dans la documentation de la police aux frontières : règles d'entrée par pays
    (visa, documents exigés), procédures (zone d'attente, asile, interprétariat, non-admission),
    et rapports sur la situation des pays d'origine."""
    return documentation.contexte(question, k=3)
# ===
@tool
def chercher_documentation(question: str) -> str:
    return documentation.contexte(question, k=3)
# >>>PANNE 1


@tool
def jours_de_sejour(passager: str, date_entree: str) -> str:
    """Dit si un passager peut entrer dans l'espace Schengen et jusqu'à quand il peut rester
    (règle des 90 jours sur 180). Utilise cet outil pour TOUTE question d'entrée ou de durée de
    séjour d'un passager identifié : il lit lui-même l'historique du passager.
    passager : identifiant, par exemple P-0412. date_entree : date d'arrivée au format AAAA-MM-JJ."""
    # <<<PANNE 3 | chaque séjour est un couple (entrée, sortie), dans cet ordre
    sejours = [(date.fromisoformat(entree), date.fromisoformat(sortie))
               for entree, sortie in historiques.get(passager, [])]
    # ===
    sejours = [(date.fromisoformat(sortie), date.fromisoformat(entree))
               for entree, sortie in historiques.get(passager, [])]
    # >>>PANNE 3
    fin = dernier_jour_autorise(sejours, date.fromisoformat(date_entree))
    return f"Dernier jour autorisé : {fin.isoformat()}" if fin else "Entrée refusée : la limite de 90 jours serait dépassée dès l'arrivée."


outils = [chercher_documentation, jours_de_sejour]
for o in outils:
    print(f"— {o.name}({', '.join(o.args)})\n  {o.description}\n")

# %% [markdown]
# <details><summary>Indice 1 — panne 1</summary>
#
# Lisez la dernière ligne de l'erreur : elle dit ce qui manque à la fonction.
# </details>
#
# <details><summary>Indice 2 — panne 1</summary>
#
# `chercher_documentation` n'a pas de *docstring*. Or c'est la docstring qui devient
# la description que le modèle lit.
# </details>
#
# <details><summary>Indice 3 — panne 1</summary>
#
# Ajouter sous la ligne `def` une description qui dit **ce que contient** la
# documentation : règles d'entrée, procédures, situation des pays d'origine.
# </details>

# %%
controle(1, all(o.description.strip() for o in outils), "un outil n'a pas de description.")

# %% [markdown]
# ## 2. L'agent — *contient la panne 2*
#
# `create_agent` (LangChain) assemble la boucle ReAct : un modèle, des outils, une
# **consigne système**. Le modèle est `ministral-3:3b`, servi par Ollama en local, à
# température 0.
#
# La consigne compte autant que les descriptions des outils. Un petit modèle croit
# souvent savoir : sans la phrase « tu ne connais ni les règles, ni les procédures…
# de Roissy », il répond de mémoire à une question de procédure, et invente des
# services et des numéros de poste qui n'existent pas.
#
# Comment l'agent « se souvient »-il d'un tour à l'autre ? Le modèle, lui, n'a aucune
# mémoire. C'est l'**état** du graphe qui en tient lieu : une liste de messages, qui
# s'allonge à chaque étape (la question, puis chaque appel d'outil, puis chaque
# observation). À chaque tour, **toute** la liste est renvoyée au modèle. S'il juge
# une observation insuffisante, il peut rappeler un outil avec d'autres arguments,
# par exemple une question reformulée. Mais personne ne lui dit si sa réponse est
# juste : il n'a que ce qu'il lit dans cette liste.
#
# Chaque tour de boucle compte : appeler le modèle, exécuter un outil, rappeler le
# modèle… La **limite de récursion** fixe le nombre maximal de ces étapes, pour qu'un
# agent qui tourne en rond finisse par s'arrêter. Trop haute, un agent perdu coûte
# cher ; trop basse, il est coupé avant d'avoir pu répondre.

# %%
agent = create_agent(
    ChatOllama(model="ministral-3:3b", temperature=0),
    outils,
    system_prompt="Tu assistes un agent de la police aux frontières à Roissy. Tu ne connais ni les règles, "
                  "ni les procédures, ni les numéros de poste de Roissy : cherche-les toujours avec "
                  "chercher_documentation avant de répondre. Pour savoir si un passager peut entrer ou "
                  "jusqu'à quand il peut rester, appelle jours_de_sejour : ne calcule jamais toi-même. "
                  "Réponds en français, brièvement.",
)
# <<<PANNE 2 | une question avec un outil demande au moins trois étapes : modèle, outil, modèle
LIMITE = 12
# ===
LIMITE = 2
# >>>PANNE 2


def demander(question: str) -> dict:
    return agent.invoke({"messages": [{"role": "user", "content": question}]},
                        {"recursion_limit": LIMITE})


def outils_appeles(resultat: dict) -> list[str]:
    return [appel["name"] for m in resultat["messages"] for appel in getattr(m, "tool_calls", [])]


def entrees_par_tour(resultat: dict) -> None:
    """Ce que le modèle a reçu à chaque tour de la boucle (reconstitué depuis l'état)."""
    messages = resultat["messages"]
    vus = 0
    tour = 0
    for i, m in enumerate(messages):
        if m.type != "ai":
            continue
        tour += 1
        print(f"\n=== Tour {tour} : le modèle reçoit {i} message(s) + la consigne système ===")
        for j, e in enumerate(messages[:i]):
            texte = e.content or ", ".join(f"{a['name']}({a['args']})" for a in getattr(e, "tool_calls", []))
            nouveau = "NOUVEAU" if j >= vus else "       "
            print(f"  {nouveau} [{e.type:5}] {texte[:90].replace(chr(10), ' ')}")
        vus = i
        sortie = ", ".join(f"{a['name']}({a['args']})" for a in m.tool_calls) or "une réponse"
        print(f"  ⇒ le modèle produit : {sortie[:100]}")


r = demander("Un passager morvanien demande l'asile au guichet : quel poste faut-il appeler ?")
print("Outils appelés :", outils_appeles(r))
print("Réponse :", r["messages"][-1].content)

# %% [markdown]
# ### Ce que le modèle reçoit, tour après tour
#
# Pour comprendre la boucle, regardons **l'entrée du modèle à chaque tour**. Elle ne
# change que d'une façon : **elle s'allonge**. Au tour 1, il n'a que la question. Au
# tour 2, il a en plus son propre appel d'outil et l'observation rendue. C'est tout ce
# qui lui permet de « réfléchir » : une liste de messages qui grossit. Quand un outil
# répond mal, c'est ici que la mauvaise observation entre, et que le tour suivant s'y
# fie.

# %%
entrees_par_tour(r)

# %% [markdown]
# <details><summary>Indice 1 — où regarder</summary>
#
# L'erreur s'appelle `GraphRecursionError`. Que dit-elle avoir atteint ?
# </details>
#
# <details><summary>Indice 2 — quelle ligne</summary>
#
# La valeur de `LIMITE`. Comptez les étapes : le modèle demande l'outil (1), l'outil
# s'exécute (2), le modèle lit le résultat et répond (3).
# </details>
#
# <details><summary>Indice 3 — la correction</summary>
#
# `LIMITE = 12` : assez pour deux ou trois appels d'outils, pas assez pour tourner
# en rond longtemps. Réexécutez la cellule, puis le contrôle.
# </details>

# %%
controle(2, "chercher_documentation" in outils_appeles(r) and "4410" in r["messages"][-1].content,
         "l'agent n'a pas trouvé le poste de l'officier de permanence asile.")

# %% [markdown]
# ## 3. Lire la trace — *la panne 3 se voit ici*
#
# Voici les cinq questions d'évaluation. Pour chacune, on affiche la **trace** de la
# boucle : chaque appel d'outil (→), chaque observation (←), puis la réponse. On
# vérifie deux choses : l'agent a-t-il appelé les **bons outils**, et sa réponse
# contient-elle le **bon élément** ? *(Compter une à deux minutes.)*
#
# Rappel de l'histoire : P-0412, entré le 14 novembre, peut rester jusqu'au
# **28 décembre 2026**. Lisez les traces des questions A1 à A3 : l'agent appelle-t-il
# le bon outil ? Et surtout, **que lui répond l'outil** (les lignes ←) ? Un agent ne
# vérifie pas ce que ses outils lui disent : si l'outil se trompe, l'agent se trompe
# avec lui, en toute confiance.

# %%
def tracer(resultat: dict) -> None:
    for m in resultat["messages"][1:]:
        for appel in getattr(m, "tool_calls", []):
            print(f"   → {appel['name']}({appel['args']})")
        if m.type == "tool":
            print(f"   ← {m.content[:120].replace(chr(10), ' ')}…")
    print(f"   RÉPONSE : {resultat['messages'][-1].content[:400]}")


bonnes = 0
for q in corpus.questions("agent"):
    r = demander(q["question"])
    appels = outils_appeles(r)
    juste = all(o in appels for o in q["outils_attendus"]) and any(
        a.lower() in r["messages"][-1].content.lower() for a in q["reponse_contient_un_de"])
    bonnes += juste
    print(f"\n{'✔' if juste else '✘'} {q['id']} · {q['question']}")
    tracer(r)
print(f"\nBonnes réponses, avec les bons outils : {bonnes} / 5")

# %% [markdown]
# Pour une question qui a demandé plusieurs tours, vous pouvez rejouer la lecture de
# l'entrée du modèle : à chaque tour, qu'est-ce qui s'est ajouté, et le modèle s'en
# est-il servi ? Changez l'indice pour choisir la question (0 = A1, 1 = A2…).

# %%
INDICE_QUESTION = 1
entrees_par_tour(demander(corpus.questions("agent")[INDICE_QUESTION]["question"]))

# %% [markdown]
# <details><summary>Indice 1 — où regarder</summary>
#
# Dans la trace de A2, quelle date renvoie `jours_de_sejour` (ligne ←) ? Est-ce le
# 28 décembre 2026 ? Et pour P-0377, qui a déjà passé 90 jours en Schengen ?
# </details>
#
# <details><summary>Indice 2 — quelle ligne</summary>
#
# L'outil répond comme si le passager n'avait **jamais** séjourné en Schengen. Dans
# `jours_de_sejour` (section 1), regardez comment la liste `sejours` est construite :
# dans quel ordre sont les deux dates de chaque couple ?
# </details>
#
# <details><summary>Indice 3 — la correction</summary>
#
# Les dates sont inversées : chaque séjour devient (sortie, entrée), une durée
# négative, que le calcul compte pour zéro jour. La bonne ligne :
# `sejours = [(date.fromisoformat(entree), date.fromisoformat(sortie)) for entree, sortie in ...]`.
# Réexécutez la section 1, la section 2 (pour recréer l'agent avec l'outil corrigé),
# puis cette section.
# </details>

# %%
controle(3, bonnes >= 4, f"seulement {bonnes} bonnes réponses sur 5 : lisez les traces, les outils appelés et ce qu'ils répondent.",
         f"{bonnes}/5")

# %% [markdown]
# ## À retenir
#
# - Un agent est un modèle dans une **boucle** : raisonner, appeler un outil, lire
#   l'observation, recommencer. La **limite de récursion** borne cette boucle.
# - Le modèle **choisit** les outils, il ne les exécute pas. Il choisit en lisant
#   leur **description** : c'est elle qu'on soigne, comme une documentation d'API
#   écrite pour un lecteur pressé.
# - Les faits sont dans les **documents** (le RAG devient un outil) ; les calculs
#   exacts sont dans le **code** (un outil aussi). Le modèle orchestre.
# - Un agent qui ne trouve pas le bon outil ne plante pas : il **improvise**. Un agent
#   dont l'outil se trompe ne le remarque pas : il **répète**. Seule la **trace** le
#   montre ; en production, on la conserve. Et chaque outil se teste seul, avant
#   d'être confié à un agent.
