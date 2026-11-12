"""Parler aux modèles locaux, par l'interface compatible OpenAI que sert Ollama.

Tout passe par la même interface : le jour où un serveur d'inférence interne
remplace Ollama, seule ADRESSE change.
"""
import os

import numpy as np
from openai import AsyncOpenAI, OpenAI

ADRESSE = os.environ.get("TP_ADRESSE", "http://localhost:11434/v1")
MODELE = os.environ.get("TP_MODELE", "ministral-3:3b")
ENCODEUR = os.environ.get("TP_ENCODEUR", "bge-m3")

# Ollama ignore la clé, mais le client OpenAI en exige une.
client = OpenAI(base_url=ADRESSE, api_key="local")
client_async = AsyncOpenAI(base_url=ADRESSE, api_key="local")


def generer(messages: list[dict], temperature: float = 0.0, max_tokens: int = 400) -> str:
    reponse = client.chat.completions.create(model=MODELE, messages=messages,
                                             temperature=temperature, max_tokens=max_tokens, seed=0)
    return reponse.choices[0].message.content


async def agenerer(messages: list[dict], temperature: float = 0.0, max_tokens: int = 400) -> str:
    """Comme generer(), mais sans bloquer : pendant l'attente, le programme fait autre chose."""
    reponse = await client_async.chat.completions.create(model=MODELE, messages=messages,
                                                         temperature=temperature, max_tokens=max_tokens, seed=0)
    return reponse.choices[0].message.content


def vectoriser(textes: list[str]) -> np.ndarray:
    """Un vecteur par texte, de longueur 1 : le produit scalaire de deux vecteurs est leur cosinus."""
    reponse = client.embeddings.create(model=ENCODEUR, input=textes)
    vecteurs = np.array([d.embedding for d in reponse.data], dtype="float32")
    return vecteurs / np.linalg.norm(vecteurs, axis=1, keepdims=True)
