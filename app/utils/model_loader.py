import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from dotenv import load_dotenv  
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def load_or_download(
    model_id: str,
    local_dir: str,
    model_type: str
):
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        tokenizer = AutoTokenizer.from_pretrained(local_dir, fix_mistral_regex=True)
        model_cls = (
            AutoModelForSequenceClassification
            if model_type == "sequence"
            else AutoModelForTokenClassification
        )
        model = model_cls.from_pretrained(local_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=True)
        model_cls = (
            AutoModelForSequenceClassification
            if model_type == "sequence"
            else AutoModelForTokenClassification
        )
        model = model_cls.from_pretrained(model_id, token=HF_TOKEN)

        os.makedirs(local_dir, exist_ok=True)
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir)

    model.eval()
    return tokenizer, model
