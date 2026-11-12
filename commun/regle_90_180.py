"""La règle des 90 jours sur 180 — futur outil de l'agent du TP 3.

Convention retenue (celle du calculateur de la Commission européenne) : pour
chaque jour de présence, on compte les jours passés dans l'espace Schengen
pendant les 180 jours qui se terminent ce jour-là, ce jour compris ; le total ne
doit jamais dépasser 90. Les jours d'entrée et de sortie comptent tous les deux.
"""
from datetime import date, timedelta


def jours_presents(sejours: list[tuple[date, date]], jour: date) -> int:
    """Jours de présence dans la fenêtre de 180 jours qui se termine à `jour`."""
    debut = jour - timedelta(days=179)
    return sum(max(0, (min(sortie, jour) - max(entree, debut)).days + 1)
               for entree, sortie in sejours)


def dernier_jour_autorise(sejours: list[tuple[date, date]], entree: date) -> date | None:
    """Dernier jour de présence possible pour une entrée à `entree` ; None si l'entrée est refusée."""
    jour, dernier = entree, None
    while jours_presents(sejours + [(entree, jour)], jour) <= 90:
        dernier, jour = jour, jour + timedelta(days=1)
    return dernier


if __name__ == "__main__":
    passes = [(date(2026, 6, 3), date(2026, 6, 28)),
              (date(2026, 8, 1), date(2026, 8, 29)),
              (date(2026, 10, 5), date(2026, 10, 20))]
    print(dernier_jour_autorise(passes, date(2026, 11, 14)))   # 2026-12-28
