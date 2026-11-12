"""L'extracteur, emballé pour MLflow (modèle « depuis le code », TP 6).

MLflow range un modèle avec la façon de s'en servir. Ce fichier est cette façon :
charger le petit modèle, lui greffer l'adaptateur enregistré, et extraire.
"""
import mlflow
from mlflow.pyfunc import PythonModel


class Extracteur(PythonModel):
    def load_context(self, context):
        from peft import PeftModel

        from commun import extraction
        self.tok, base = extraction.charger()
        self.modele = PeftModel.from_pretrained(base, context.artifacts["adaptateur"])
        self.modele.eval()

    def predict(self, context, model_input, params=None):
        from commun import extraction
        return [extraction.extraire(self.modele, self.tok, m) for m in model_input]


mlflow.models.set_model(Extracteur())
