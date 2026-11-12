"""L'extracteur de mentions de contrôle : un petit modèle, un adaptateur LoRA.

C'est la version de référence de ce que le TP 4 écrit dans le notebook. Le TP 6
s'en sert pour entraîner ses deux variantes sans dépendre du TP 4.
"""
import json
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELE_PETIT = os.environ.get("TP_PETIT_MODELE", "HuggingFaceTB/SmolLM2-135M-Instruct")
CONSIGNE = "Extrais la fiche de la mention de contrôle, en JSON."
CIBLES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
CHAMPS = ["nationalite", "document", "motif", "demande_asile"]


def charger():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODELE_PETIT)
    return tok, AutoModelForCausalLM.from_pretrained(MODELE_PETIT, dtype=torch.float32)


def prompt(tok, mention: str) -> str:
    return tok.apply_chat_template([{"role": "system", "content": CONSIGNE},
                                    {"role": "user", "content": mention}],
                                   tokenize=False, add_generation_prompt=True)


def exemple(tok, e: dict) -> tuple[list[int], list[int]]:
    """Les tokens d'un exemple, et les étiquettes : -100 sur le prompt, pour n'apprendre que la réponse."""
    p = tok(prompt(tok, e["entree"]), add_special_tokens=False)["input_ids"]
    r = tok(json.dumps(e["sortie"], ensure_ascii=False) + tok.eos_token, add_special_tokens=False)["input_ids"]
    return p + r, [-100] * len(p) + r


def greffer(modele, rang: int = 8):
    from peft import LoraConfig, get_peft_model
    return get_peft_model(modele, LoraConfig(r=rang, lora_alpha=2 * rang, target_modules=CIBLES,
                                             task_type="CAUSAL_LM"))


def entrainer(modele, tok, jeu: list[dict], pas: int = 60, taux: float = 1e-3, lot: int = 8,
              a_chaque_pas=None) -> list[float]:
    exemples = [exemple(tok, e) for e in jeu]
    optim = torch.optim.AdamW([p for p in modele.parameters() if p.requires_grad], lr=taux)
    hasard, pertes = random.Random(0), []
    modele.train()
    for n in range(pas):
        choix = hasard.sample(exemples, lot)
        longueur = max(len(ids) for ids, _ in choix)
        ids = torch.tensor([i + [tok.eos_token_id] * (longueur - len(i)) for i, _ in choix])
        etiquettes = torch.tensor([e + [-100] * (longueur - len(e)) for _, e in choix])
        masque = torch.tensor([[1] * len(i) + [0] * (longueur - len(i)) for i, _ in choix])
        perte = modele(input_ids=ids, attention_mask=masque, labels=etiquettes).loss
        perte.backward()
        optim.step()
        optim.zero_grad()
        pertes.append(perte.item())
        if a_chaque_pas:
            a_chaque_pas(n, perte.item())
    modele.eval()
    return pertes


def extraire(modele, tok, mention: str) -> str:
    ids = tok(prompt(tok, mention), return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        sortie = modele.generate(**ids, max_new_tokens=60, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(sortie[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def lire_fiche(texte: str) -> dict | None:
    try:
        fiche = json.loads(texte)
        return fiche if isinstance(fiche, dict) else None
    except json.JSONDecodeError:
        return None


def evaluer(fiches: list[dict | None], jeu: list[dict]) -> dict:
    """JSON valides, et part des champs justes, comparés à l'extraction attendue."""
    valides = sum(f is not None for f in fiches)
    justes = sum((f or {}).get(c) == e["sortie"][c] for f, e in zip(fiches, jeu) for c in CHAMPS)
    return {"json_valides": valides, "champs_justes": round(justes / (len(CHAMPS) * len(jeu)), 3)}
