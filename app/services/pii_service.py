from app.models.registry import MODEL_REGISTRY

def analyze_text(text: str) -> dict:
    results = {}

    for name, model in MODEL_REGISTRY.items():
        results[name] = model.analyze(text)

    return results
