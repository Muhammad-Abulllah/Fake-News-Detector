# Fake News Classifier

BERT-based fake news classifier fine-tuned on the WELFake dataset. The project includes exploratory notebooks, a FastAPI endpoint, a Gradio demo, trained model artifacts, and model comparison outputs.

## Project Structure

```text
fake-news-classifier/
├── notebooks/
│   ├── EDA.ipynb
│   ├── Tensors.ipynb
│   └── Train Test Split.ipynb
├── bert_final/
├── app.py
├── gradio_app.py
├── requirements.txt
├── model_comparison.csv
├── model_comparison.png
└── README.md
```

Additional notebooks may also be kept in `notebooks/` for tokenizer experiments and baseline models.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the FastAPI App

```bash
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs` to test the `/predict` endpoint.

## Run the Gradio Demo

```bash
python gradio_app.py
```

## Model Files

The app expects the fine-tuned BERT model in `bert_final/`:

```text
bert_final/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

`model.safetensors` is large, so use Git LFS if you plan to push the model to GitHub.
