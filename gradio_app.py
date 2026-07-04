import gradio as gr
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model
MODEL_PATH = "./bert_final"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def predict(text):
    if not text.strip():
        return "Please enter an article or headline.", {" ": 0}

    inputs = tokenizer(
        text,
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

    label = "🔴 FAKE NEWS" if pred == 1 else "🟢 REAL NEWS"
    scores = {
        "REAL": round(probs[0][0].item(), 4),
        "FAKE": round(probs[0][1].item(), 4)
    }
    return label, scores

# Build UI
with gr.Blocks(title="Fake News Detector") as demo:
    gr.Markdown("""
    # 🔍 Fake News Detector
    **BERT-based classifier fine-tuned on WELFake dataset (63K articles)**
    - Accuracy: 99.05% | F1: 98.94% | Precision: 99.32% | Recall: 98.55%
    - Outperforms TF-IDF baseline by 4% across all metrics
    """)

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Paste article or headline here",
                placeholder="Enter news article text...",
                lines=8
            )
            submit_btn = gr.Button("Analyze", variant="primary")
            gr.Examples(
                examples=[
                    ["The Federal Reserve raised interest rates by 25 basis points on Wednesday, citing continued concerns about inflation."],
                    ["Scientists confirm that drinking bleach cures cancer and the government is hiding this from the public."],
                    ["NASA announces successful launch of Artemis mission carrying four astronauts to the Moon."],
                    ["SHOCKING: Bill Gates microchips found in COVID vaccines, leaked documents reveal secret population control plan."]
                ],
                inputs=text_input
            )

        with gr.Column():
            label_output = gr.Textbox(label="Prediction", lines=1)
            confidence_output = gr.Label(label="Confidence Scores")

    submit_btn.click(
        fn=predict,
        inputs=text_input,
        outputs=[label_output, confidence_output]
    )

if __name__ == "__main__":
    demo.launch()