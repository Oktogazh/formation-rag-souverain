# %% [markdown]
# # TP 4 — Le fine-tuning : apprendre une forme à un petit modèle
#
# **Durée** : ≈ 75 min · **Niveau** ★★★★☆ · **4 pannes** · indépendant des autres TP
#
# ## L'histoire
#
# Depuis que les combats ont repris dans la province de Hadrel, le terminal 2E
# rédige chaque jour quelque trois cents **mentions de contrôle**. Ce sont des notes
# en texte libre, écrites vite, chacune à sa façon :
#
# > *14/11 08h42, vol MV218. Ressortissante morvanienne, passeport ordinaire valide.
# > Elle déclare rendre visite à sa sœur. Formule une demande d'asile à la frontière.*
#
# Une **mention**, c'est donc ce petit texte libre : quelques phrases tapées par un
# agent au guichet, juste après le contrôle d'un passager. Pas de formulaire, pas de
# champs : chacun écrit à sa façon. Trois autres exemples, tirés du jeu de données :
#
# > *04/11 10h28, vol OS117. Passagère de nationalité morvanienne, passeport
# > ordinaire. Elle déclare venir en vacances.*
#
# > *22/11 11h18, vol VD410. Contrôle approfondi au poste 12. Ressortissante
# > ostranienne, passeport de service. Elle indique participer à un salon
# > professionnel. Billet retour présenté.*
#
# > *12/11 19h46, vol OS131. Signalé par la compagnie. Passager de nationalité
# > morvanienne, passeport ordinaire. Il déclare suivre une formation linguistique.*
#
# La même information (« morvanienne », « Ressortissante ostranienne ») s'écrit de
# plusieurs façons, et certaines phrases ne servent pas à la fiche (« Signalé par la
# compagnie »).
#
# La direction voudrait en tirer une **fiche** structurée — nationalité, document,
# motif, demande d'asile oui ou non — pour deux raisons : repérer sans délai les
# demandes d'asile, que la procédure impose de signaler au poste 4410, et compter,
# jour après jour, qui arrive et pourquoi.
#
# Le modèle de la formation (`ministral-3:3b`) saurait le faire avec une bonne
# consigne, mais lentement, et pas toujours dans le même format. Or ici, il n'y a
# **rien à savoir** : tout est dans la mention. Il y a seulement **une forme à
# respecter**, toujours la même. C'est exactement ce que le fine-tuning apprend bien.
#
# ## Ce que vous allez faire
#
# Prendre un tout petit modèle, `SmolLM2-135M-Instruct` (cent fois plus petit que
# celui du TP 1), et lui apprendre la tâche avec **LoRA**, sur processeur, en deux
# minutes :
#
# 1. mesurer ce qu'il sait faire **avant** ;
# 2. lui **greffer** des matrices LoRA ;
# 3. préparer les **données** ;
# 4. l'**entraîner**, en lisant la courbe de perte ;
# 5. mesurer **après**, sur des mentions qu'il n'a jamais vues.
#
# ### LoRA en trois phrases
#
# Le modèle d'origine est **gelé** : aucun de ses 135 millions de poids ne bouge. À
# côté de certaines de ses matrices `W`, on greffe deux petites matrices `A` et `B`
# dont le produit `B·A` corrige la sortie de `W` ; seules `A` et `B` s'entraînent.
# `B` part de zéro : au premier pas, le modèle greffé est exactement le modèle
# d'origine.
#
# ### Le vocabulaire, en français et en anglais
#
# La documentation, les bibliothèques et les messages d'erreur sont en anglais. Ce
# notebook emploie les mots français ; voici leur équivalent :
#
# | Dans ce notebook | En anglais | Ce que c'est |
# |---|---|---|
# | perte | *loss* | l'écart entre ce que le modèle prédit et ce qu'on attend |
# | pas | *step* | une mise à jour des poids |
# | lot | *batch* | les exemples traités ensemble à chaque pas |
# | taille de lot | *batch size* | leur nombre (ici 8) |
# | époque | *epoch* | un passage complet sur tout le jeu d'entraînement |
# | taux d'apprentissage | *learning rate* (`lr`) | la longueur de chaque pas |
# | gradient | *gradient* | dans quel sens bouger chaque poids |
# | rétropropagation | *backpropagation* (`backward()`) | le calcul des gradients |
# | optimiseur | *optimizer* | ce qui applique le pas (ici AdamW) |
# | étiquettes | *labels* | les tokens que le modèle doit prédire |
# | jeu d'entraînement / d'évaluation | *training set* / *evaluation set* | |
# | rang | *rank* (`r`) | la taille intérieure des matrices LoRA |
# | adaptateur | *adapter* | les matrices LoRA entraînées, enregistrées à part |
#
# ## Comment travailler
#
# Le code est **complet, mais en panne à 4 endroits**. Exécutez les cellules une par
# une ; chaque contrôle dit ✅ ou ❌. L'entraînement prend deux minutes : lisez le texte pendant qu'il
# tourne.
#
# Ce TP est le plus difficile de la formation. Le texte est plus long que dans les
# autres : prenez le temps de le lire, c'est lui qui donne les clés des pannes.

# %% [markdown]
# ## Préparation

# %%
import pathlib
import sys

RACINE = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "commun").is_dir())
sys.path.insert(0, str(RACINE))

import json
import random

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from commun import RESULTATS, corpus
from commun.controle import controle

torch.manual_seed(0)
NOM = "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = AutoTokenizer.from_pretrained(NOM)
modele = AutoModelForCausalLM.from_pretrained(NOM, dtype=torch.float32)
CONSIGNE = "Extrais la fiche de la mention de contrôle, en JSON."
CHAMPS = ["nationalite", "document", "motif", "demande_asile"]

train = corpus.mentions("train-fr")
print(f"{len(train)} mentions d'entraînement. Un exemple :\n")
print(train[0]["entree"], "\n→", json.dumps(train[0]["sortie"], ensure_ascii=False))

# %% [markdown]
# ## 1. Avant l'entraînement
#
# Deux fonctions servent jusqu'à la fin du TP :
#
# - `extraire` met la mention au **format de conversation** du modèle (consigne,
#   puis mention), le fait générer, et rend le texte produit ;
# - `mesurer` compte, sur un jeu de mentions, les **JSON valides** et la part des
#   **champs justes**.
#
# **Ce que le modèle lit vraiment.** Comme au TP 1, le modèle ne reçoit pas
# « une consigne et une mention » : il reçoit **un seul texte**, que le *gabarit de
# chat* fabrique en entourant chaque message de balises spéciales (`<|im_start|>system`,
# `<|im_end|>`…), et qui se termine par l'ouverture du tour de l'assistant. C'est la
# fonction `prompt` qui le fabrique. Retenez cette idée, elle servira à la fin du TP :
# le fine-tuning va apprendre au modèle que **ce texte-là, dans ce format-là**, appelle
# une fiche JSON. Si, à l'usage, on lui donne autre chose (la mention seule, sans
# balises ni consigne), il ne reconnaît plus la situation qu'il a apprise.
#
# Pas de panne dans cette section. On mesure le modèle tel qu'il sort de chez
# HuggingFace, sur dix mentions. *(Une vingtaine de secondes.)*

# %%
def prompt(mention: str) -> str:
    return tok.apply_chat_template([{"role": "system", "content": CONSIGNE},
                                    {"role": "user", "content": mention}],
                                   tokenize=False, add_generation_prompt=True)


def extraire(mention: str) -> str:
    # <<<PANNE 4 | à l'usage, le modèle doit recevoir le même format qu'à l'entraînement
    ids = tok(prompt(mention), return_tensors="pt", add_special_tokens=False)
    # ===
    ids = tok(mention, return_tensors="pt", add_special_tokens=False)
    # >>>PANNE 4
    with torch.no_grad():
        sortie = modele.generate(**ids, max_new_tokens=60, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(sortie[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def mesurer(jeu: list[dict]) -> dict:
    valides, justes = 0, 0
    for e in jeu:
        try:
            fiche = json.loads(extraire(e["entree"]))
            valides += 1
            justes += sum(fiche.get(c) == e["sortie"][c] for c in CHAMPS)
        except (json.JSONDecodeError, AttributeError):
            pass
    return {"json_valides": f"{valides}/{len(jeu)}", "champs_justes": round(justes / (4 * len(jeu)), 2)}


print("Ce que le modèle lit :\n" + prompt(train[0]["entree"]))
print("Mention :", train[0]["entree"])
print("Extraction :", extraire(train[0]["entree"]))
print("\nAvant :", mesurer(corpus.mentions("eval-fr")[:10]))

# %% [markdown]
# Le modèle d'origine répond quelque chose, mais pas une fiche JSON. Il ne sait pas ce
# qu'on attend de lui.
#
# *(La fonction `extraire` contient la panne 4, qui ne se verra qu'à la section 5 :
# avant l'entraînement, le modèle échoue de toute façon.)*
#
# ## 2. Greffer LoRA — *contient la panne 1*
#
# `LoraConfig` dit **où** greffer les matrices `A` et `B` : dans quelles couches
# linéaires du modèle, désignées par leur **nom**. Ces noms dépendent de
# l'architecture. SmolLM2 suit l'architecture de Llama : dans chacune de ses 30
# couches, l'attention a quatre matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`) et
# le réseau qui suit en a trois (`gate_proj`, `up_proj`, `down_proj`). Affichez
# `print(modele)` pour les voir.
#
# Trois réglages :
#
# - le **rang** `r = 8` : `A` a 8 lignes, `B` a 8 colonnes. Plus le rang est grand,
#   plus la correction `B·A` peut être riche, et plus il y a de poids à entraîner.
#   Des rangs de 4 à 64 sont courants ; 8 ou 16 suffisent pour une tâche de forme
#   comme celle-ci.
# - **`lora_alpha`** règle l'**intensité** de la correction. La sortie d'une couche
#   greffée vaut `W·x + (alpha / r) · B·A·x` : le facteur `alpha / r` (ici 16 / 8 = 2)
#   multiplie ce qu'apprend LoRA avant de l'ajouter. Son intérêt : si l'on change le
#   rang, on garde le même `alpha / r`, et l'on n'a pas à réajuster le taux
#   d'apprentissage. L'usage courant est `alpha = r` ou `alpha = 2r`.
# - **`target_modules`** dit **où** greffer.
#
# **Où greffer ?** On peut greffer LoRA sur n'importe quelle couche linéaire du
# modèle. Ici, on greffe **partout** : les sept matrices de chacune des 30 couches.
# On peut se le permettre parce que le modèle est minuscule : même partout, les
# matrices LoRA restent sous la barre des **5 % de paramètres entraînables**, ce qui
# suffit largement pour un entraînement LoRA (le contrôle 1 vérifie ce seuil).
#
# Ce n'est pas toujours le choix fait. L'article d'origine de LoRA (2021) ne greffait
# que deux matrices de l'attention, **`q_proj` et `v_proj`**, et c'est encore le
# réglage par défaut de la bibliothèque PEFT pour les modèles de type Llama : le
# moins cher en mémoire. Depuis QLoRA (2023), l'usage s'est déplacé vers **toutes les
# couches linéaires** (`target_modules="all-linear"`), qui donnent de meilleurs
# résultats pour un coût encore modeste. Sur un grand modèle, quand la mémoire
# manque, on revient à l'attention seule, ou à une partie des couches.

# %%
config = LoraConfig(
    r=8, lora_alpha=16, task_type="CAUSAL_LM",
    # <<<PANNE 1 | les noms des couches de GPT-2 n'existent pas dans un modèle de type Llama
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # ===
    target_modules=["c_attn", "c_proj", "c_fc"],
    # >>>PANNE 1
)
modele = get_peft_model(modele, config)
entrainables = sum(p.numel() for p in modele.parameters() if p.requires_grad)
total = sum(p.numel() for p in modele.parameters())
print(f"Paramètres entraînables : {entrainables:,} sur {total:,} ({100 * entrainables / total:.2f} %)")

# %% [markdown]
# <details><summary>Indice 1 — où regarder</summary>
#
# L'erreur dit que des « target modules » sont introuvables. Lesquels ?
# </details>
#
# <details><summary>Indice 2 — quelle ligne</summary>
#
# `target_modules` : `c_attn`, `c_proj`, `c_fc` sont les noms des couches de GPT-2.
# Exécutez `print(modele)` dans une nouvelle cellule pour lire ceux de SmolLM2.
# </details>
#
# <details><summary>Indice 3 — la correction</summary>
#
# `target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`.
# Réexécutez **la préparation** (pour recharger un modèle sans greffe), puis cette
# cellule.
# </details>

# %%
controle(1, 0 < entrainables < 0.05 * total, "aucune matrice LoRA n'a été greffée.",
         f"{entrainables:,} paramètres entraînables")

# %% [markdown]
# Moins de 2 % des paramètres s'entraînent. C'est ce qui rend l'entraînement possible
# sur un processeur — et l'adaptateur obtenu pèsera une dizaine de mégaoctets.
#
# ## 3. Les données — *contient la panne 2*
#
# Un exemple d'entraînement, c'est la conversation complète : consigne, mention,
# puis la fiche attendue, mise bout à bout en une seule suite de tokens.
#
# **Les étiquettes.** Un modèle de langue apprend à **prédire le token suivant**. À
# chaque position de la suite, on lui donne donc une **étiquette** (*label*) : le
# token qu'il aurait dû prédire. La perte est la moyenne, sur toutes les positions,
# de sa « surprise » devant la bonne étiquette.
#
# Or on ne veut pas lui apprendre à écrire des consignes ni des mentions : elles lui
# seront toujours **données**. On veut seulement lui apprendre **la réponse**. On
# remplace donc l'étiquette de chaque token du prompt par la valeur spéciale `-100`.
# Ce n'est pas un token : c'est une convention de PyTorch, qui veut dire « position
# ignorée, ne compte pas dans la perte » (`ignore_index=-100` dans sa fonction de
# perte). Résultat : le modèle lit le prompt, mais n'est noté que sur la fiche.
#
# En pratique, on donne comme étiquettes la suite elle-même, et Transformers la décale
# d'un cran : l'étiquette posée sur un token est ce que le modèle devait prédire en
# lisant tout ce qui le précède. Sur le premier exemple, cela donne (les derniers
# tokens du prompt, puis les premiers de la réponse) :
#
# ```
# token          étiquette
# ass            -100        ← prompt : lu, pas noté
# istant         -100
# \n             -100
# {"             {"          ← réponse : noté
# national       national
# ite            ite
# ":             ":
# ```
#
# Reste le jeu d'**évaluation** : des mentions sur lesquelles on mesurera le modèle
# après l'entraînement. Règle d'or : **jamais** des mentions qu'il a vues pendant
# l'entraînement. Il pourrait les avoir apprises par cœur, et la mesure ne dirait
# plus rien de sa capacité à traiter une mention nouvelle.
#
# Ici, le jeu de données est livré **déjà découpé** en fichiers séparés
# (`train-fr`, `eval-fr`…). **Dans la vraie vie, ce découpage, c'est à vous de le
# faire**, avant tout entraînement, et une fois pour toutes. Quelques règles :
# mélanger au hasard avant de couper (avec une graine fixe, pour pouvoir refaire) ;
# garder de côté 10 à 20 % des exemples ; vérifier que chaque catégorie rare (ici, les
# demandes d'asile) est présente des deux côtés ; et surveiller les **quasi-doublons**,
# deux mentions presque identiques réparties de part et d'autre, qui font passer
# pour de la compréhension ce qui n'est que du par-cœur.

# %%
def exemple(e: dict) -> tuple[list[int], list[int]]:
    p = tok(prompt(e["entree"]), add_special_tokens=False)["input_ids"]
    r = tok(json.dumps(e["sortie"], ensure_ascii=False) + tok.eos_token, add_special_tokens=False)["input_ids"]
    return p + r, [-100] * len(p) + r


exemples = [exemple(e) for e in train]
# <<<PANNE 2 | on évalue sur des mentions jamais vues à l'entraînement
jeu_eval = corpus.mentions("eval-fr")[:20]
# ===
jeu_eval = train[:20]
# >>>PANNE 2
ids, etiquettes = exemples[0]
print(f"Premier exemple : {len(ids)} tokens, dont {sum(e != -100 for e in etiquettes)} comptent dans la perte")

# %% [markdown]
# <details><summary>Indice 1 — où regarder</summary>
#
# D'où viennent les mentions de `jeu_eval` ?
# </details>
#
# <details><summary>Indice 2 — quelle ligne</summary>
#
# `jeu_eval = train[:20]` : ce sont les vingt premières mentions… d'entraînement.
# </details>
#
# <details><summary>Indice 3 — la correction</summary>
#
# `jeu_eval = corpus.mentions("eval-fr")[:20]` : le fichier d'évaluation, séparé.
# </details>

# %%
vues = {e["id"] for e in train}
controle(2, not any(e["id"] in vues for e in jeu_eval), "le jeu d'évaluation contient des mentions d'entraînement.")

# %% [markdown]
# ## 4. Entraîner — *contient la panne 3*
#
# La boucle d'entraînement tient en quelques lignes. À chaque **pas** (*step*) :
#
# 1. on tire un **lot** (*batch*) de 8 exemples au hasard, qu'on aligne sur la même
#    longueur : les exemples courts sont complétés par des tokens de remplissage
#    (*padding*), que le **masque d'attention** cache au modèle et que l'étiquette
#    `-100` retire de la perte ;
# 2. le modèle calcule la **perte** (*loss*) : à quel point il était loin de la fiche
#    attendue, en moyenne sur le lot ;
# 3. `backward()` (la **rétropropagation**, *backpropagation*) calcule, pour chaque
#    poids de `A` et `B`, dans quel sens le bouger pour réduire la perte (le
#    **gradient**) ;
# 4. l'optimiseur (*optimizer*, ici AdamW) fait ce pas, d'une longueur fixée par le
#    **taux d'apprentissage** (*learning rate*, l'argument `lr`).
#
# Pourquoi un lot, et pas un exemple à la fois ? Le gradient d'un seul exemple est
# bruité ; la moyenne sur 8 donne une direction plus sûre, pour le même nombre de pas.
#
# Un taux trop petit, et l'on n'avance pas ; trop grand, et chaque pas dépasse le
# but : la perte remonte, oscille, ou devient `nan` (*not a number*). Pour LoRA, les
# valeurs courantes vont de `1e-4` à `1e-3`. On fait 60 pas de 8 exemples, soit 480
# exemples vus : un peu plus d'une **époque** (*epoch*) sur les 300 mentions.
# *(Deux minutes environ : la perte s'affiche tous les 10 pas.)*
#
# **Pourquoi une boucle PyTorch écrite à la main ?** Pour que vous voyiez chaque
# étape. Le modèle vient de Transformers, mais la boucle est du PyTorch pur. En
# pratique, on la confie souvent à une bibliothèque : le `Trainer` de Transformers, ou
# le `SFTTrainer` de TRL (Hugging Face), qui applique aussi le gabarit de chat, pose
# les `-100` et gère le remplissage tout seul. Ils font exactement ce que fait cette
# cellule, plus l'enregistrement régulier, l'évaluation en cours de route et le calcul
# sur plusieurs GPU. Mais ils le cachent : quand quelque chose ne va pas, c'est cette
# boucle qu'il faut avoir en tête.

# %%
# <<<PANNE 3 | un taux d'apprentissage trop grand fait diverger l'entraînement
TAUX = 1e-3
# ===
TAUX = 0.5
# >>>PANNE 3
PAS, LOT = 60, 8

optimiseur = torch.optim.AdamW([p for p in modele.parameters() if p.requires_grad], lr=TAUX)
hasard, pertes = random.Random(0), []
modele.train()
for n in range(PAS):
    lot = hasard.sample(exemples, LOT)
    longueur = max(len(i) for i, _ in lot)
    entree = torch.tensor([i + [tok.eos_token_id] * (longueur - len(i)) for i, _ in lot])
    cibles = torch.tensor([e + [-100] * (longueur - len(e)) for _, e in lot])
    masque = torch.tensor([[1] * len(i) + [0] * (longueur - len(i)) for i, _ in lot])
    perte = modele(input_ids=entree, attention_mask=masque, labels=cibles).loss
    perte.backward()
    optimiseur.step()
    optimiseur.zero_grad()
    pertes.append(perte.item())
    if n % 10 == 0 or n == PAS - 1:
        print(f"pas {n:>2} · perte {perte.item():.3f}")
modele.eval()

# %% [markdown]
# <details><summary>Indice 1 — où regarder</summary>
#
# Lisez la colonne des pertes : descend-elle régulièrement vers 0 ?
# </details>
#
# <details><summary>Indice 2 — quelle ligne</summary>
#
# `TAUX = 0.5` : chaque pas est des centaines de fois trop grand.
# </details>
#
# <details><summary>Indice 3 — la correction</summary>
#
# `TAUX = 1e-3`. Puis réexécutez **la préparation et toutes les cellules depuis le
# début** : les poids ont été abîmés par l'entraînement raté, il faut repartir du
# modèle d'origine.
# </details>

# %%
fin = sum(pertes[-5:]) / 5
controle(3, fin == fin and fin < 0.3 * pertes[0], "la perte ne descend pas : l'entraînement a divergé.",
         f"perte {pertes[0]:.2f} → {fin:.3f}")

# %% [markdown]
# La perte part d'environ 3 et tombe presque à 0. Une perte nulle sur l'entraînement
# ne prouve rien à elle seule : elle peut vouloir dire « appris par cœur ». Seule la
# mesure sur des mentions **jamais vues** dira si le modèle a appris la tâche.
#
# ## 5. Après l'entraînement — *la panne 4 se voit ici*
#
# On mesure sur les mentions d'évaluation. *(Une vingtaine de secondes.)*
#
# Si la mesure est mauvaise alors que la perte était presque nulle, ce n'est pas
# l'entraînement qui a raté : c'est **l'usage**. Revenez à l'idée de la section 1 :
# à l'entraînement, chaque exemple commençait par le texte fabriqué par `prompt`
# (balises, consigne, mention, ouverture du tour de l'assistant). Le modèle a appris
# « après ce texte, j'écris une fiche ». Il faut donc que `extraire` lui présente
# **exactement le même début**. Comparez ce qu'affiche la cellule ci-dessous pour les
# deux textes.

# %%
apres = mesurer(jeu_eval)
print("Après :", apres)
print("\nÀ l'entraînement, le modèle lisait :\n" + tok.decode(exemples[0][0][:40]) + "…")
for e in jeu_eval[:3]:
    print("\n", e["entree"], "\n→", extraire(e["entree"]))

# %% [markdown]
# <details><summary>Indice 1 — où regarder</summary>
#
# La perte était presque nulle, mais les fiches sont fausses ou absentes. Le modèle
# reçoit-il, à l'usage, la même chose qu'à l'entraînement ? Ce qu'il lisait à
# l'entraînement est affiché ci-dessus ; ce qu'il lit à l'usage, c'est ce que
# `extraire` passe au tokenizer.
# </details>
#
# <details><summary>Indice 2 — quelle ligne</summary>
#
# Dans `extraire` (section 1), la ligne `ids = tok(mention, …)` passe la mention
# **brute** au tokenizer : sans les balises, sans la consigne, sans l'ouverture du tour
# de l'assistant. Or la fonction `prompt`, juste au-dessus, fabrique exactement le
# texte qu'utilisait l'entraînement (c'est elle qu'appelle `exemple`, section 3).
# </details>
#
# <details><summary>Indice 3 — la correction</summary>
#
# Dans `extraire`, remplacer `tok(mention, …)` par `tok(prompt(mention), …)` :
#
# `ids = tok(prompt(mention), return_tensors="pt", add_special_tokens=False)`
#
# Puis, **sans réexécuter la préparation** (elle rechargerait le modèle d'origine et
# effacerait l'entraînement) : réexécutez la cellule de la section 1 qui définit
# `extraire` (la mesure « avant » qu'elle contient passera sur le modèle entraîné et
# affichera de bons chiffres : c'est normal), puis cette section-ci et son contrôle.
# </details>

# %%
n_eval = len(jeu_eval)
controle(4, apres["json_valides"] == f"{n_eval}/{n_eval}" and apres["champs_justes"] >= 0.9,
         f"le modèle entraîné rend {apres['json_valides']} JSON valides, {apres['champs_justes']:.0%} de champs justes.",
         f"{apres['json_valides']} JSON valides, {apres['champs_justes']:.0%} de champs justes")

# %% [markdown]
# ## 6. Ce que le modèle n'a pas appris
#
# Plus de panne. Deux observations, pour préparer le TP 6.
#
# **Une mention en anglais.** Les officiers de liaison étrangers rédigent parfois
# leurs mentions en anglais. Le modèle n'en a vu aucune à l'entraînement.

# %%
anglais = corpus.mentions("eval-en")
print("Mentions en anglais :", mesurer(anglais))
for e in anglais[:3]:
    print("\n", e["entree"], "\n→", extraire(e["entree"]), "\n  attendu :", json.dumps(e["sortie"], ensure_ascii=False))

# %% [markdown]
# Le JSON reste valide — la **forme** est apprise —, mais des champs sont faux, et
# rien ne le signale. Un modèle fine-tuné ne fait bien que ce qu'on lui a montré. Le
# TP 6 entraînera une seconde version, avec des mentions anglaises, et comparera les
# deux dans MLflow.
#
# **Une question qui n'est pas une mention.**

# %%
print(extraire("Quelle est la capitale de la Morvanie ?"))

# %% [markdown]
# Il rend une fiche, même absurde. Le fine-tuning lui a appris **une forme**, pas des
# connaissances : pour les faits, c'est le RAG.
#
# **Fine-tuning et RAG ne s'opposent pas : l'un peut servir l'autre.** Avant d'entrer
# dans la base vectorielle d'un RAG, les documents sont découpés en morceaux (les
# *chunks*). Un petit modèle fine-tuné peut faire ce travail en amont : découper,
# ramener chaque morceau à une **taille et une forme standard**, écarter ce qui ne
# sert pas. C'est l'un des usages les plus pertinents du fine-tuning : une tâche de
# forme, répétée des milliers de fois, et un modèle assez petit pour tourner sur vos
# propres machines. Les documents ne quittent jamais l'environnement sécurisé pour
# être préparés.
#
# On garde l'adaptateur : quelques mégaoctets, à greffer sur le modèle d'origine.

# %%
modele.save_pretrained(RESULTATS / "tp04-adaptateur")
print(sorted(p.name for p in (RESULTATS / "tp04-adaptateur").iterdir()))

# %% [markdown]
# ## À retenir
#
# - Le fine-tuning apprend une **forme** (un format, une tâche répétitive), pas des
#   faits. Les faits qui changent restent dans les documents.
# - Un petit modèle fine-tuné peut **préparer les morceaux** d'un RAG en amont :
#   taille et forme standard, sans que les documents sortent de chez vous.
# - **LoRA** gèle le modèle et entraîne deux petites matrices par couche greffée ;
#   les couches se désignent par leur **nom**, propre à chaque architecture.
# - On n'apprend que la **réponse** (étiquettes `-100` sur le prompt), et l'on
#   utilise le modèle avec **le même format** qu'à l'entraînement.
# - La **courbe de perte** dit si l'entraînement converge ; seul un jeu
#   d'**évaluation séparé** dit si le modèle a appris.
# - Un modèle fine-tuné échoue **en silence** hors de ce qu'il a vu : l'anglais,
#   une question.
