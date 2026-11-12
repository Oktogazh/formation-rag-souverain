"""Les cellules de contrôle des TP : elles disent si l'on peut passer à la suite."""


class ControleEchoue(Exception):
    """Arrête l'exécution du notebook sur un message court, sans pile d'appels."""

    def __init__(self, numero: int, symptome: str):
        super().__init__(f"Contrôle {numero} : {symptome}")
        self.numero, self.symptome = numero, symptome

    def _render_traceback_(self):  # affichage propre dans Jupyter
        return [f"❌ Contrôle {self.numero} — {self.symptome}",
                "   La panne est dans la section au-dessus. Les indices sont dans les cellules repliées."]


def controle(numero: int, condition: bool, symptome: str, detail: str = "") -> None:
    if not condition:
        raise ControleEchoue(numero, symptome)
    print(f"✅ Contrôle {numero} : OK, passez à la suite.{('  ' + detail) if detail else ''}")
