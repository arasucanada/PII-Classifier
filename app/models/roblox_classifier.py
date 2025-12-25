import torch
from app.models.base import BasePIIModel
from app.utils.model_loader import load_or_download
from transformers import pipeline

class RobloxPIIClassifier(BasePIIModel):

    def __init__(self):
        self.tokenizer, self.model = load_or_download(
            model_id="Roblox/roblox-pii-classifier",
            local_dir="models_store/roblox",
            model_type="sequence"
        )

    def analyze(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]
        labels = self.model.config.id2label

        results = {labels[i]: float(probs[i]) for i in range(len(probs))}

        # pipe = pipeline(
        #     "text-classification",
        #     model=self.model,
        #     tokenizer=self.tokenizer
        # )
        # results = pipe(text)

        return {
            "asking_pii": results.get("p_privacy_asking_for_pii", 0) > 0.2,
            "giving_pii": results.get("p_privacy_giving_pii", 0) > 0.3,
            "raw_scores": results
        }
        # return {
        #     "asking_pii": results[0]['label'] == 'p_privacy_asking_pii',
        #     "giving_pii": results[0]['label'] == 'p_privacy_giving_pii',
        #     "raw_scores": results
        # }