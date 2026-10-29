# 🌾 Edge Vision Transformer — Paddy Disease Detection

> **Phase A** — Early detection of **Leaf Blast** and **Brown Spot** diseases in paddy crops using a Hybrid ResNet-18 + Vision Transformer (ViT) architecture, with edge deployment via ONNX Runtime.

---

## 📋 Project Info

| Field | Detail |
|---|---|
| **Project Code** | 26P_3CSE373 |
| **Institute** | IILM University |
| **Team** | Neelakshi Bougal · Ayush Kumar · Ansh Patel · Ansh Kumar |
| **Mentor** | Rahul Maurya |

---

## 🎯 Phase A Target Classes

| Index | Class | Description |
|---|---|---|
| 0 | Leaf Blast | Fungal disease caused by *Magnaporthe oryzae* |
| 1 | Brown Spot | Fungal disease caused by *Bipolaris oryzae* |
| 2 | Healthy | No disease detected |

> **Phase B** (future): Extended classes (Sheath Blight, Bacterial Blight, Tungro, etc.) added only after Phase A achieves **F1 ≥ 0.90** for all disease classes.

---

## 🏗️ Architecture

```
Input (224×224×3)
    │
[ResNet-18 Backbone]  →  Feature Maps (7×7×512)
    │
[Patch Embedding]     →  49 tokens × 256-dim
    │
[Transformer Encoder] ×4 layers
    ├── Multi-Head Self-Attention (8 heads)
    └── Feed Forward MLP
    │
[CLS Token / Mean Pool]
    │
[Classification Head: MLP → Softmax (3 classes)]
```

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
git clone https://github.com/your-repo/edge-vit-paddy.git
cd edge-vit-paddy

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your PostgreSQL credentials
```

Edit `config/config.yaml` to adjust hyperparameters and paths.

### 3. Prepare Dataset

Download the **Rice Leaf Diseases** dataset from Kaggle:
```bash
kaggle datasets download -d vbookshelf/rice-leaf-diseases
```

Place images in:
```
data/raw/
├── Leaf Blast/
├── Brown Spot/
└── Healthy/
```

### 4. Train

```bash
python train.py --config config/config.yaml
```

### 5. Export to ONNX

```bash
python src/optimization/export_onnx.py --config config/config.yaml
```

### 6. Run Dashboard

```bash
streamlit run app/main.py
```

---

## 📁 Project Structure

```
edge-vit-paddy/
├── config/config.yaml          # Central config (classes, hyperparams, paths)
├── data/                       # Dataset splits (gitignored)
├── src/
│   ├── data/                   # Dataset, preprocessing, dataloaders
│   ├── models/                 # ResNet extractor, ViT encoder, hybrid model
│   ├── training/               # Trainer, validator
│   ├── evaluation/             # Metrics, confusion matrix, visualizer
│   ├── optimization/           # ONNX export, INT8 quantization
│   ├── inference/              # ONNX inference engine
│   └── services/               # Prediction service, remediation, reports
├── db/                         # SQLAlchemy models, CRUD, Alembic migrations
├── app/                        # Streamlit dashboard
├── models/                     # Exported ONNX models
├── checkpoints/                # PyTorch checkpoints
├── outputs/plots/              # Training curves, confusion matrices
├── tests/                      # Pytest test suite
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── train.py
```

---

## 📊 Phase A Performance Targets

| Metric | Target |
|---|---|
| Leaf Blast F1 | ≥ 0.90 |
| Brown Spot F1 | ≥ 0.90 |
| Overall Accuracy | ≥ 92% |

---

## 🧪 Testing

```bash
pytest tests/ -v
python train.py --config config/config.yaml --epochs 1 --dry-run
python src/optimization/export_onnx.py --verify
```

---

## 🐳 Docker Deployment

```bash
docker-compose up --build
```

Access dashboard at `http://localhost:8501`

---

## 📝 License

Academic project — IILM University, 2026.

