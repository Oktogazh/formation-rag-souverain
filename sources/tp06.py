# %% [markdown]
# # TP 6 — Deux versions d'un modèle, une règle qui change : MLflow, RAG et fine-tuning
#
# **Durée** : ≈ 90 min · **Niveau** ★★★☆☆ · **4 pannes** · indépendant des autres TP
#
# ## L'histoire
#
# 5 novembre 2026. Deux nouvelles arrivent le même jour à Roissy.
#
# - **La guerre reprend.** La trêve de Valberg, conclue en juin, est rompue depuis le
#   8 octobre. Les combats ont repris dans le nord de la province de Hadrel ; Sarvik
#   est encerclée. Un nouveau rapport sur la Morvanie vient d'être publié.
# - **La règle change.** À compter du 1er décembre 2026, l'exemption de visa des
#   ressortissants morvaniens est suspendue : il leur faudra un visa, même pour un
#   court séjour.
#
# Pendant ce temps, l'extracteur de mentions de contrôle (le petit modèle fine-tuné
# du TP 4) a un défaut : les officiers de liaison étrangers rédigent leurs mentions
# en **anglais**, et l'extracteur, entraîné sur du français seul, s'y trompe en
# silence.
#
# L'équipe doit donc : entraîner une **seconde version** de l'extracteur, la comparer
# proprement à la première, choisir laquelle mettre **en production** ; puis mettre
# l'assistant à jour avec la nouvelle règle ; et enfin assembler toute la chaîne.
#
# ## Ce que vous allez faire
#
# 1. Entraîner **deux variantes** du même modèle — l'une sur des mentions françaises,
#    l'autre sur des mentions françaises et anglaises — et ranger chaque
#    entraînement dans **MLflow** : ses réglages, sa courbe de perte, ses scores, et
#    le modèle lui-même.
# 2. Les **comparer** dans l'interface de MLflow.
# 3. **Enregistrer** les deux versions et désigner celle qui part en production.
# 4. Faire entrer la **nouvelle règle** dans l'assistant, et voir pourquoi c'est le
#    travail du RAG, pas du fine-tuning.
# 5. Assembler la **chaîne complète** : mention → fiche → documentation → synthèse.
#
# ### MLflow en trois mots
#
# MLflow est un outil libre de **suivi d'expériences**. Chaque entraînement y devient
# un **run** : ses **paramètres** (les réglages choisis avant), ses **métriques** (les
# nombres mesurés pendant et après), ses **artefacts** (les fichiers produits, dont le
# modèle). Le **registre de modèles** garde ensuite les versions retenues, et un
# **alias** (`production`) désigne celle qu'on utilise. Il tourne ici sur votre
# machine, et rien n'en sort.
#
# ## Comment travailler
#
# Le code est **complet, mais en panne à 4 endroits**. Chaque contrôle dit ✅ ou ❌.
# Les deux entraînements prennent ensemble cinq à six minutes : lisez pendant qu'ils
# tournent.

# %% [markdown]
# ## Préparation
#
# On démarre un **serveur MLflow** local. Il range tout dans le dossier `resultats/`
# du dépôt : une petite base de données (les runs, les métriques) et un dossier
# d'artefacts (les modèles). Ouvrez l'adresse affichée dans votre navigateur.

# %%
import pathlib
import sys

RACINE = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "commun").is_dir())
sys.path.insert(0, str(RACINE))

import mlflow
from mlflow import MlflowClient

from commun import RESULTATS, corpus, extraction, modele, processus
from commun.controle import controle
from commun.recherche import Documentation

ADRESSE_MLFLOW = processus.lancer(
    "mlflow", ["mlflow", "server", "--host", "127.0.0.1", "--port", "5055",
               "--backend-store-uri", f"sqlite:///{RESULTATS / 'mlflow.db'}",
               "--artifacts-destination", str(RESULTATS / "mlflow-artefacts")],
    "http://127.0.0.1:5055", sante="/health")
mlflow.set_tracking_uri(ADRESSE_MLFLOW)
mlflow.set_experiment("extracteur-mentions")
print("MLflow :", ADRESSE_MLFLOW)

# %% [markdown]
# ## 1. Deux variantes, deux runs — *contient les pannes 1 et 2*
#
# Les deux variantes ne diffèrent que par leurs **données d'entraînement** :
#
# | Variante | Données | Exemples |
# |---|---|---|
# | `fr` | mentions françaises | 300 |
# | `fr+en` | mentions françaises, plus 40 mentions anglaises | 340 |
#
# Tout le reste est identique : même petit modèle (`SmolLM2-135M-Instruct`), même
# greffe LoRA (rang 8), même nombre de pas, même taux d'apprentissage. Les fonctions
# d'entraînement et d'extraction sont celles du TP 4, rangées dans
# `commun/extraction.py`.
#
# Chaque variante est mesurée sur **les deux mêmes jeux d'évaluation** : 20 mentions
# françaises et 20 mentions anglaises, jamais vues à l'entraînement.
#
# Ce qu'on range dans MLflow, pour chaque run :
#
# - `mlflow.log_params(...)` : les **paramètres**, des valeurs choisies *avant*
#   l'entraînement ; on y met **tout ce qui distingue** les variantes ;
# - `mlflow.log_metric(...)` : les **métriques**, des **nombres** mesurés : la perte
#   à chaque pas (d'où une courbe), puis les scores ;
# - `mlflow.pyfunc.log_model(...)` : le **modèle**, avec la façon de s'en servir
#   (`commun/extracteur_mlflow.py`), pour pouvoir le recharger plus tard d'une ligne.
#
# *(Cinq à six minutes pour les deux variantes.)*

# %%
PAS, TAUX, RANG = 60, 1e-3, 8
VARIANTES = {"fr": corpus.mentions("train-fr"),
             "fr+en": corpus.mentions("train-fr") + corpus.mentions("train-en")}
EVALUATIONS = {"fr": corpus.mentions("eval-fr")[:20], "en": corpus.mentions("eval-en")}


def entrainer_variante(nom: str, donnees: list[dict]) -> str:
    tok, base = extraction.charger()
    greffe = extraction.greffer(base, rang=RANG)
    with mlflow.start_run(run_name=nom):
        # <<<PANNE 2 | les paramètres doivent dire ce qui distingue les deux variantes
        mlflow.log_params({"donnees": nom, "exemples": len(donnees), "rang": RANG, "pas": PAS, "taux": TAUX})
        # ===
        mlflow.log_params({"rang": RANG, "pas": PAS, "taux": TAUX})
        # >>>PANNE 2
        extraction.entrainer(greffe, tok, donnees, pas=PAS, taux=TAUX,
                             a_chaque_pas=lambda n, perte: mlflow.log_metric("perte", perte, step=n))
        for langue, jeu in EVALUATIONS.items():
            fiches = [extraction.lire_fiche(extraction.extraire(greffe, tok, e["entree"])) for e in jeu]
            score = extraction.evaluer(fiches, jeu)
            # <<<PANNE 1 | une métrique est un nombre, pas un texte
            mlflow.log_metric(f"champs_justes_{langue}", score["champs_justes"])
            # ===
            mlflow.log_metric(f"champs_justes_{langue}", f"{score['champs_justes']:.0%}")
            # >>>PANNE 1
            print(f"{nom} · {langue} : {score['json_valides']}/20 JSON valides, {score['champs_justes']:.0%} de champs justes")
        dossier = RESULTATS / f"tp06-adaptateur-{nom}"
        greffe.save_pretrained(dossier)
        info = mlflow.pyfunc.log_model(name="extracteur", python_model=str(RACINE / "commun" / "extracteur_mlflow.py"),
                                       artifacts={"adaptateur": str(dossier)}, pip_requirements=[])
    return info.model_uri


modeles = {nom: entrainer_variante(nom, donnees) for nom, donnees in VARIANTES.items()}

# %% [markdown]
# <details><summary>Indice 1 — où regarder</summary>
#
# L'erreur parle d'une valeur invalide pour une métrique. Laquelle ?
# </details>
#
# <details><summary>Indice 2 — quelle ligne</summary>
#
# `mlflow.log_metric(..., f"{score['champs_justes']:.0%}")` envoie un **texte**
# (« 95% »). MLflow ne range que des nombres dans les métriques : c'est ce qui lui
# permet de tracer des courbes et de trier.
# </details>
#
# <details><summary>Indice 3 — la correction</summary>
#
# `mlflow.log_metric(f"champs_justes_{langue}", score["champs_justes"])`. La mise en
# forme en pourcentage, c'est pour l'affichage, pas pour le stockage.
# </details>

# %%
derniers = mlflow.search_runs(experiment_names=["extracteur-mentions"], order_by=["start_time DESC"], max_results=2)
controle(1, "metrics.champs_justes_en" in derniers and derniers["metrics.champs_justes_en"].notna().all(),
         "les scores des deux variantes ne sont pas dans MLflow.")

# %% [markdown]
# ## 2. Comparer dans MLflow — *révèle la panne 2*
#
# Ouvrez MLflow dans votre navigateur (l'adresse est affichée dans la préparation),
# puis l'expérience **extracteur-mentions**.
#
# 1. Cochez les deux derniers runs et cliquez sur **Compare**.
# 2. Regardez les **courbes de perte** : les deux descendent-elles de la même façon ?
# 3. Regardez les métriques `champs_justes_fr` et `champs_justes_en`.
# 4. Regardez les **paramètres** : peut-on dire, en lisant MLflow seul, ce qui
#    distingue les deux runs ?
#
# La même comparaison, dans le notebook :

# %%
colonnes = [c for c in derniers.columns if c.startswith(("params.", "metrics.champs"))]
print(derniers[["tags.mlflow.runName", *colonnes]].to_string(index=False))

# %% [markdown]
# <details><summary>Indice 1 — où regarder</summary>
#
# Dans le tableau, les colonnes `params.*` ont-elles des valeurs différentes pour les
# deux runs ? Dans six mois, quelqu'un qui ouvre MLflow saura-t-il ce qu'on a changé ?
# </details>
#
# <details><summary>Indice 2 — quelle ligne</summary>
#
# Dans `entrainer_variante`, `mlflow.log_params` ne range que les réglages
# **communs** aux deux variantes. Ce qui les distingue — les données — n'y est pas.
# Seul le nom du run en garde une trace, et un nom se change d'un clic.
# </details>
#
# <details><summary>Indice 3 — la correction</summary>
#
# `mlflow.log_params({"donnees": nom, "exemples": len(donnees), "rang": RANG, "pas": PAS, "taux": TAUX})`.
# Réexécutez la section 1 (les deux entraînements), puis cette section.
# </details>

# %%
controle(2, "params.donnees" in derniers and derniers["params.donnees"].nunique() == 2,
         "rien, dans les paramètres, ne distingue les deux variantes.")

# %% [markdown]
# ## 3. Enregistrer et choisir — *contient la panne 3*
#
# Le **registre de modèles** garde les versions qu'on a décidé de conserver, sous un
# nom : `extracteur-mentions`, version 1, version 2… Un **alias** désigne celle qu'on
# utilise : `production`. Changer de modèle en production, c'est déplacer l'alias ;
# revenir en arrière aussi. Le code qui se sert du modèle ne change pas : il demande
# toujours `extracteur-mentions@production`.
#
# On enregistre les deux versions, puis on place l'alias sur la meilleure **pour les
# mentions anglaises** — c'est le défaut qu'on veut corriger — à condition qu'elle ne
# recule pas sur le français.

# %%
client = MlflowClient()
derniers = mlflow.search_runs(experiment_names=["extracteur-mentions"], order_by=["start_time DESC"], max_results=2)
versions = {nom: mlflow.register_model(uri, "extracteur-mentions").version for nom, uri in modeles.items()}
print("Versions enregistrées :", versions)
print(derniers[["tags.mlflow.runName", "metrics.champs_justes_fr", "metrics.champs_justes_en"]].to_string(index=False))

# <<<PANNE 3 | la meilleure variante est celle qui a le PLUS de champs justes
meilleur = derniers.sort_values("metrics.champs_justes_en", ascending=False).iloc[0]
# ===
meilleur = derniers.sort_values("metrics.champs_justes_en").iloc[0]
# >>>PANNE 3
choix = meilleur["tags.mlflow.runName"]
client.set_registered_model_alias("extracteur-mentions", "production", versions[choix])
print(f"\nEn production : la variante « {choix} », version {versions[choix]}")

# %% [markdown]
# <details><summary>Indice 1 — où regarder</summary>
#
# Quelle variante a été mise en production ? Est-ce celle qui a le meilleur score
# sur les mentions anglaises ?
# </details>
#
# <details><summary>Indice 2 — quelle ligne</summary>
#
# `sort_values` trie par ordre **croissant** par défaut : la première ligne est la
# **moins** bonne.
# </details>
#
# <details><summary>Indice 3 — la correction</summary>
#
# `derniers.sort_values("metrics.champs_justes_en", ascending=False).iloc[0]`. Puis
# réexécutez cette cellule : l'alias se déplace, sans rien réentraîner. C'est tout
# l'intérêt du registre.
# </details>

# %%
plus_haut = derniers.loc[derniers["metrics.champs_justes_en"].idxmax(), "tags.mlflow.runName"]
en_production = client.get_model_version_by_alias("extracteur-mentions", "production").version
controle(3, en_production == versions[plus_haut], "l'alias « production » désigne la moins bonne des deux variantes.",
         f"production = « {plus_haut} »")

# %% [markdown]
# Dans MLflow, onglet **Models** : le modèle `extracteur-mentions`, ses deux versions,
# et l'alias `production`. Chaque version renvoie au run qui l'a produite, avec ses
# paramètres et sa courbe : on sait toujours **d'où vient** le modèle en service.
#
# ## 4. La règle change — *contient la panne 4*
#
# La nouvelle fiche « visa exigé pour les Morvaniens à partir du 1er décembre » et le
# nouveau rapport sur Hadrel sont arrivés. Comment les faire entrer dans l'assistant ?
#
# - **Par le fine-tuning**, il faudrait fabriquer des exemples où la bonne réponse
#   est « visa exigé », réentraîner, réévaluer, redéployer — et recommencer à chaque
#   changement de règle. Pire : le modèle garderait des traces de l'ancienne règle,
#   sans qu'on sache lesquelles.
# - **Par le RAG**, il suffit d'**ajouter les documents à l'index**. Quelques
#   secondes, et la réponse cite sa source.
#
# C'est la ligne de partage de la formation : **le fine-tuning pour une forme stable**
# (l'extraction des mentions), **le RAG pour des faits qui changent** (les règles, la
# situation des pays). La consigne du prompt ajoute une précaution, vue au TP 2 : si
# deux documents se contredisent, le plus récent l'emporte.

# %%
# <<<PANNE 4 | l'index doit être reconstruit avec les nouveaux documents
documentation = Documentation(avec_changements=True)
# ===
documentation = Documentation()
# >>>PANNE 4


def repondre(question: str, trouves: list[dict]) -> str:
    contexte = "\n\n".join(f"[{m['id']}, {m['date']}]\n{m['texte']}" for m in trouves)
    return modele.generer([{"role": "user", "content":
        "Tu assistes un agent de la police aux frontières. Réponds brièvement, en t'appuyant uniquement sur ces "
        "documents et en citant leurs identifiants entre crochets. Si deux documents se contredisent, le plus "
        f"récent l'emporte.\n\n{contexte}\n\nQuestion : {question}"}], max_tokens=250)


QUESTION = "Un ressortissant morvanien arrive le 3 décembre 2026 pour un court séjour : a-t-il besoin d'un visa ?"
trouves = documentation.chercher(QUESTION, k=3)
for m in trouves:
    print(f"{m['id']:<26} {m['date']}")
print("\n" + repondre(QUESTION, trouves))

# %% [markdown]
# <details><summary>Indice 1 — où regarder</summary>
#
# Parmi les sources trouvées, voyez-vous une fiche datée de novembre 2026 ?
# </details>
#
# <details><summary>Indice 2 — quelle ligne</summary>
#
# `Documentation()` charge l'index d'**avant** les changements. Les nouveaux
# documents sont dans `data/roissy/changements/`, mais personne ne les a indexés.
# </details>
#
# <details><summary>Indice 3 — la correction</summary>
#
# `documentation = Documentation(avec_changements=True)`. Le RAG ne suit un
# changement que si l'on **réindexe**.
# </details>

# %%
controle(4, any(m["doc"] == "regle-morvanie-2026-12" for m in trouves),
         "l'assistant ne connaît pas la nouvelle règle : il répond avec l'ancienne fiche.")

# %% [markdown]
# ## 5. La chaîne complète
#
# Plus de panne. Voici l'assistant entier, tel qu'il servirait au guichet le
# 3 décembre 2026 :
#
# 1. l'**extracteur fine-tuné**, chargé depuis le registre par son alias
#    `production`, lit la mention de contrôle et en tire une fiche ;
# 2. la fiche devient une question, que le **RAG** instruit avec la documentation à
#    jour ;
# 3. le modèle de la formation rédige une **synthèse** qui cite ses sources ;
# 4. **l'agent décide.**
#
# Trois mentions, dont une en anglais. *(Une minute environ.)*

# %%
extracteur = mlflow.pyfunc.load_model("models:/extracteur-mentions@production")
MENTIONS = [
    "03/12 07h55, vol MV218. Ressortissant morvanien, passeport ordinaire valide. Il déclare rendre visite "
    "à son frère installé à Lyon. Aucune demande d'asile.",
    "03/12 08h10, flight MV224. Morvanian national, ordinary passport. She states visiting her sister. "
    "Wishes to apply for asylum.",
    "03/12 08h31, vol OS117. Ressortissant ostranien, passeport de service valide. Il vient pour une "
    "réunion de travail.",
]
for mention, texte in zip(MENTIONS, extracteur.predict(MENTIONS)):
    fiche = extraction.lire_fiche(texte) or {}
    question = (f"Passager arrivé le 3 décembre 2026 : nationalité {fiche.get('nationalite')}, "
                f"{fiche.get('document')}, motif : {fiche.get('motif')}. A-t-il besoin d'un visa ?")
    if fiche.get("demande_asile"):
        question += " Il demande l'asile : qui prévenir, et que disent les rapports les plus récents sur son pays ?"
    print(f"\n### {mention}\nFICHE : {fiche}\nSYNTHÈSE : {repondre(question, documentation.chercher(question, k=4))}")

# %% [markdown]
# ## Pour finir
#
# Quand vous avez fini d'explorer MLflow, arrêtez le serveur.

# %%
processus.arreter("mlflow")
print("Serveur MLflow arrêté. Tout est conservé dans resultats/ : relancez la préparation pour le rouvrir.")

# %% [markdown]
# ## Les six TP, réparés
#
# La formation est finie : vous pouvez récupérer les six notebooks **sans leurs
# pannes**, pour les relire ou les rejouer de bout en bout. Passez la variable
# ci-dessous à `True` et exécutez la cellule. Vos propres versions sont d'abord mises
# de côté dans `resultats/mes-notebooks/`.
#
# (Ce notebook-ci est remplacé lui aussi : rouvrez-le pour voir sa version corrigée.
# La même chose se fait dans un terminal : `python tp.py corriger`.)

# %%
REMPLACER_PAR_LES_CORRIGES = False

if REMPLACER_PAR_LES_CORRIGES:
    import subprocess
    print(subprocess.run([sys.executable, str(RACINE / "tp.py"), "corriger"], capture_output=True, text=True).stdout)
else:
    print("Rien n'a été remplacé. Passez la variable à True pour récupérer les six TP réparés.")

# %% [markdown]
# ## À retenir
#
# - **MLflow** range chaque entraînement : paramètres (ce qu'on a choisi), métriques
#   (des **nombres**), artefacts (le modèle). On y met **tout ce qui distingue** deux
#   variantes, sinon la comparaison est muette.
# - Le **registre** garde les versions ; l'**alias** `production` désigne celle qu'on
#   sert. Changer de modèle, c'est déplacer l'alias — dans le bon sens.
# - Deux variantes ne se comparent que sur **les mêmes jeux d'évaluation**.
# - **Fine-tuning** pour une **forme** stable ; **RAG** pour des **faits** qui
#   changent. Et le RAG ne suit un changement que si l'on **réindexe**.
# - L'assistant **prépare** ; l'agent **décide**. Ce qui touche une personne reste
#   une décision humaine.
