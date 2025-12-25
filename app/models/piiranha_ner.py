import torch
from app.models.base import BasePIIModel
from app.utils.model_loader import load_or_download
from transformers import pipeline

class PiiranhaPIIRecognizer(BasePIIModel):

    def __init__(self):
        self.tokenizer, self.model = load_or_download(
            model_id="iiiorg/piiranha-v1-detect-personal-information",
            local_dir="models_store/piiranha",
            model_type="token"
        )

    def analyze(self, text: str) -> dict:
        pipe = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple"
        )
        results = pipe(text)

        # Sort results by start position in descending order to avoid index shifting
        results_sorted = sorted(results, key=lambda x: x['start'], reverse=True)

        masked_text = text
        for result in results_sorted:
            start = result['start']
            end = result['end']
            masked_text = masked_text[:start] + "******" + masked_text[end:]

        return {
            "masked_text": masked_text,
            "entities": results
        }
