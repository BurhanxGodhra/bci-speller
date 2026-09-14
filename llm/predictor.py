import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class WordPredictor:
    def __init__(self, model_name="gpt2", device="cpu", top_k=3):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.top_k = top_k

    def predict_next(self, text: str) -> list[str]:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]

        candidates = torch.topk(logits, self.top_k * 5).indices.tolist()
        words = []
        for tid in candidates:
            tok = self.tokenizer.decode([tid]).strip()
            if tok.isalpha() and tok not in words:
                words.append(tok)
            if len(words) >= self.top_k:
                break
        return words
