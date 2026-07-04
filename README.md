---

## 📁 Dataset

**WELFake** — 72,134 news articles (real + fake) scraped from four news datasets.
- After cleaning and deduplication: **63,071 articles**
- Class distribution: 55% Real / 45% Fake
- Split: 80% train / 10% val / 10% test

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10 |
| Deep Learning | PyTorch |
| NLP Model | Hugging Face Transformers (BERT) |
| Training | Google Colab (T4 GPU) |
| Backend API | FastAPI + Uvicorn |
| Frontend UI | Gradio |
| Deployment | Hugging Face Spaces |
| Data Processing | Pandas, Scikit-learn |
| Visualization | Matplotlib, Seaborn |

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/fake-news-classifier
cd fake-news-classifier
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run Gradio app**
```bash
python gradio_app.py
```

**5. Or run FastAPI**
```bash
uvicorn app:app --reload
# Visit http://127.0.0.1:8000/docs
```

---

## 📈 Training Details

- **Model:** bert-base-uncased (110M parameters)
- **Optimizer:** AdamW (lr=2e-5)
- **Scheduler:** Linear warmup
- **Batch size:** 16
- **Epochs:** 3
- **Hardware:** Google Colab T4 GPU (~42 min/epoch)

| Epoch | Train Loss | Val Loss |
|-------|------------|----------|
| 1 | 0.0867 | 0.0392 |
| 2 | 0.0263 | 0.0418 |
| 3 | 0.0072 | 0.0533 |

---

## 👤 Author

**Muhammad Abdullah**
- 📧 nisarabdullah6741@gmail.com
- 💼 [LinkedIn](https://www.linkedin.com/in/abdullah-nisar-aa7335247/)
