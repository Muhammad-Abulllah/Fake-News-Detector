from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = FastAPI(title="Fake News Classifier API")

# Load model and tokenizer
MODEL_PATH = "./bert_final"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

class Article(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Fake News Classifier API is running"}

@app.post("/predict")
def predict(article: Article):
    inputs = tokenizer(
        article.text,
        return_tensors='pt',
        max_length=256,
        truncation=True,
        padding='max_length'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label = "FAKE" if pred == 1 else "REAL"

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "message": f"This article is {label} with {round(confidence * 100, 2)}% confidence"
    }