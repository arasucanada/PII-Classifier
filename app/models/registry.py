from app.models.roblox_classifier import RobloxPIIClassifier
from app.models.piiranha_ner import PiiranhaPIIRecognizer

MODEL_REGISTRY = {
    "roblox": RobloxPIIClassifier(),
    "piiranha": PiiranhaPIIRecognizer()
}
