# AI-Driven Personalized Career Guidance System

An intelligent career recommendation system powered by machine learning and NLP, designed as an M.Tech Major Project.

## 🚀 Quick Start

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app/app.py
```

The app will be available at: **http://localhost:8501**

---

## 📁 Project Structure

```
career-recommender-major-project/
│
├── app/                          # Core application modules
│   ├── app.py                   # Streamlit web interface
│   ├── recommender.py           # Hybrid recommendation engine
│   ├── domain_predictor.py      # Domain classification
│   ├── personality.py           # Big Five personality analysis
│   ├── sbert_semantic.py        # SBERT semantic embeddings
│   ├── clustering.py            # K-Means clustering & PCA
│   ├── nlp_pipeline.py          # NLP preprocessing utilities
│   └── data_loader.py           # Dataset loading utilities
│
├── data/                         # Datasets
│   ├── domain_dataset.csv       # 187,405 training samples
│   └── careers_dataset.csv      # 767 career profiles
│
├── models/                       # Trained ML models
│   ├── domain_classifier.pkl    # Logistic Regression classifier
│   └── domain_tfidf_vectorizer.pkl
│
├── train/                        # Training scripts
│   ├── train_domain_classifier.py
│   ├── generate_massive_dataset.py
│   └── generate_comprehensive_careers.py
│
├── docs/                         # Documentation
│   ├── README.md                # Detailed documentation
│   ├── ARCHITECTURE.md          # System design & API reference
│   ├── EXPERIMENTS.md           # Experimental results
│   ├── SAMPLE_TEST_INPUTS.md    # Test cases for demo
│   ├── BEGINNERS_GUIDE.md       # Getting started guide
│   └── MTech_Major_Dissertation.docx
│
├── outputs/                      # Generated outputs & figures
│   └── appendix_figures/        # Dissertation figures
│
├── experiments/                  # Experimental scripts
│   ├── compare_similarity_models.py
│   └── interest_clustering.py
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Domain Classification** | 99.30% accuracy with TF-IDF + Logistic Regression |
| **Semantic Similarity** | SBERT (all-MiniLM-L6-v2) for deep semantic matching |
| **Hybrid Scoring** | Multi-signal scoring (similarity + keywords + personality + skills) |
| **Personality Analysis** | Big Five (OCEAN) trait extraction and matching |
| **Interest Clustering** | K-Means clustering with PCA visualization |
| **Interactive Dashboard** | Real-time visualizations with Plotly charts |

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| **ML/NLP** | scikit-learn, sentence-transformers, NLTK |
| **Web UI** | Streamlit, Plotly |
| **Data Processing** | pandas, numpy |
| **Language** | Python 3.11+ |

---

## 📊 System Performance

| Metric | Value |
|--------|-------|
| Domain Classification Accuracy | **99.30%** |
| Training Samples | 187,405 |
| Career Profiles | 767 |
| Career Domains | 48 |
| SBERT Model | all-MiniLM-L6-v2 (384-dim) |

---

## 🔧 Hybrid Scoring Formula

```
Hybrid Score = 0.40 × Semantic Similarity
             + 0.25 × Keyword Overlap  
             + 0.20 × Personality Match
             + 0.15 × Skills Match
```

---

## 📚 Documentation

- **[Detailed README](docs/README.md)** - Comprehensive usage guide
- **[Architecture](docs/ARCHITECTURE.md)** - System design & components
- **[Experiments](docs/EXPERIMENTS.md)** - Validation & performance metrics
- **[Sample Inputs](docs/SAMPLE_TEST_INPUTS.md)** - 12 test cases for demonstration
- **[Beginner's Guide](docs/BEGINNERS_GUIDE.md)** - Getting started guide

---

## 🎓 M.Tech Project

**Status**: ✅ Complete - Ready for Defense

**Last Updated**: November 2025
