# System Architecture

## Overview

The AI-Driven Personalized Career Guidance System is built using a modular, layered architecture that separates concerns across data processing, machine learning, natural language processing, and user interface components.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                      │
│                     (Streamlit Web App)                          │
│  ┌────────────┬──────────────┬─────────────┬──────────────┐    │
│  │  Input UI  │ Model Select │ Viz Display │ Results View │    │
│  └────────────┴──────────────┴─────────────┴──────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Application Logic Layer                       │
│  ┌─────────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Domain Predictor│  │ Recommender  │  │ Personality      │  │
│  │                 │  │ Engine       │  │ Analyzer         │  │
│  └─────────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NLP & ML Processing Layer                     │
│  ┌──────────┬──────────────┬────────────┬──────────────────┐   │
│  │ TF-IDF   │ SBERT        │ K-Means    │ Text Processing  │   │
│  │ Vectorizer│ Embedder     │ Clustering │ Pipeline         │   │
│  └──────────┴──────────────┴────────────┴──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Data Layer                               │
│  ┌────────────────┬──────────────────┬────────────────────┐    │
│  │ Domain Dataset │ Careers Database │ Trained Models     │    │
│  │ (187K samples) │ (767 careers)    │ (Classifier, TF-IDF)│   │
│  └────────────────┴──────────────────┴────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Layer Descriptions

### 1. User Interface Layer (`app/app.py`)

**Purpose**: Provides interactive web interface for user input and results visualization

**Components**:
- **Input Section**: Multi-line text area for user descriptions
- **Model Selection**: Radio button for choosing similarity engine (TF-IDF/SBERT/Hybrid)
- **Display Options**: Customizable output settings (top-k, scores, visualizations)
- **Results Display**: Structured presentation of recommendations with interactive charts

**Technologies**:
- Streamlit 1.28.0 for web framework
- Plotly 5.19.0 for interactive visualizations
- Session state management for user preferences

**Key Features**:
- Real-time model switching
- Interactive charts (bar, pie, radar plots)
- Responsive layout with columns and expanders
- Color-coded match quality indicators

---

### 2. Application Logic Layer

#### 2.1 Domain Predictor (`app/domain_predictor.py`)

**Purpose**: Predicts career domain from user text using trained ML classifier

**Input**: User text description (string)

**Output**: 
- Predicted domain label
- Confidence score (0-1)
- Probability distribution across all 48 domains

**Algorithm**:
```python
1. Load trained Logistic Regression model + TF-IDF vectorizer
2. Preprocess user text (tokenization, stopword removal)
3. Transform text to TF-IDF vector (3000 features)
4. Predict domain using classifier.predict_proba()
5. Return top domain with confidence
```

**Key Functions**:
- `is_model_loaded()`: Checks if models are available
- `predict_domain(text)`: Returns domain string
- `predict_domain_with_probabilities(text)`: Returns full prediction dict

**Dependencies**:
- scikit-learn (Logistic Regression)
- joblib (model loading)
- nlp_pipeline (preprocessing)

---

#### 2.2 Recommender Engine (`app/recommender.py`)

**Purpose**: Generates ranked career recommendations using hybrid scoring

**Input**:
- User text
- Predicted domain
- Similarity model choice (tfidf/sbert/hybrid)
- Top-k parameter

**Output**: List of career dictionaries with scores

**Scoring Formula**:
```
Hybrid Score = 0.40 × Similarity + 0.25 × Keywords + 0.20 × Personality + 0.15 × Skills
```

**Key Functions**:

1. **`recommend_careers_for_domain(user_text, domain, top_k, similarity_model)`**
   - Filters careers by domain
   - Computes similarity scores using selected model
   - Calculates keyword overlap, personality match, skills match
   - Combines scores using weighted formula
   - Returns top-k ranked recommendations

2. **`recommend_careers_cross_domain(user_text, top_k, domains_to_search)`**
   - Similar to above but searches across multiple/all domains
   - Useful for exploratory recommendations

3. **`explain_recommendation(recommendation)`**
   - Generates human-readable explanation
   - Highlights strongest matching factors

**Similarity Models**:

- **TF-IDF**: `compute_similarity_scores(user_text, career_keywords)`
  - Uses nlp_pipeline.py TF-IDF vectorizer
  - Cosine similarity on sparse 3000-dim vectors
  
- **SBERT**: `compute_sbert_similarity_scores(user_text, career_keywords)`
  - Uses sbert_semantic.py embedder
  - Cosine similarity on dense 384-dim embeddings
  
- **Hybrid**: Averages TF-IDF and SBERT scores

---

#### 2.3 Personality Analyzer (`app/personality.py`)

**Purpose**: Extracts Big Five personality traits from user text

**Input**: Tokenized user text (list of words)

**Output**: Dictionary of personality trait scores

**Big Five Traits**:
1. **Openness**: Creativity, curiosity, open-mindedness
2. **Conscientiousness**: Organization, responsibility, planning
3. **Extraversion**: Sociability, energy, assertiveness
4. **Agreeableness**: Cooperation, empathy, kindness
5. **Neuroticism**: Emotional stability, stress handling

**Algorithm**:
```python
1. Load trait keyword dictionaries
2. For each trait:
   - Count keyword matches in user text
   - Normalize by trait dictionary size
3. Return trait: score mapping
```

**Key Functions**:
- `personality_scores_from_text(tokens)`: Returns scores dict
- `personality_summary(scores)`: Returns human-readable summary
- `compute_personality_match(user_scores, career_requirements)`: Calculates match score (0-1)

---

### 3. NLP & ML Processing Layer

#### 3.1 TF-IDF Vectorizer (`app/nlp_pipeline.py`)

**Purpose**: Classical text vectorization using term frequency-inverse document frequency

**Configuration**:
- **Features**: 3000 max features
- **N-grams**: Unigrams + Bigrams (1-2)
- **Min DF**: 2 (ignore very rare terms)
- **Sublinear TF**: True (logarithmic term frequency scaling)
- **Preprocessing**: Lowercase, stopword removal, lemmatization

**Key Functions**:

1. **`preprocess_text(text)`**
   - Tokenization using NLTK
   - Lowercase conversion
   - Stopword removal (English)
   - Lemmatization (WordNet)

2. **`tokenize_text(text)`**
   - Returns list of preprocessed tokens
   - Used for keyword matching and personality analysis

3. **`compute_similarity_scores(user_text, career_texts)`**
   - Vectorizes user text and career texts
   - Computes cosine similarity matrix
   - Returns list of similarity scores (0-1)

**Strengths**:
- Fast computation (sparse matrices)
- Effective for keyword-based matching
- Interpretable features

**Limitations**:
- No semantic understanding
- Struggles with synonyms and paraphrasing
- High dimensionality (3000 features)

---

#### 3.2 SBERT Embedder (`app/sbert_semantic.py`)

**Purpose**: Semantic text embeddings using Sentence-BERT

**Model**: `all-MiniLM-L6-v2`
- **Parameters**: 22.7 million
- **Embedding Size**: 384 dimensions
- **Training**: Fine-tuned on 1 billion sentence pairs
- **Performance**: Fast inference (~10-20ms per sentence)

**Architecture**:
```python
class SBERTEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = 384
    
    def encode_sentences(self, texts):
        """Returns normalized 384-dim embeddings"""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings
    
    def compute_similarity(self, text1, text2):
        """Cosine similarity between two texts"""
        emb1, emb2 = self.encode_sentences([text1, text2])
        return cosine_similarity(emb1, emb2)
    
    def batch_similarity_scores(self, query_text, corpus_texts):
        """Query vs corpus similarities"""
        query_emb = self.encode_sentences([query_text])
        corpus_embs = self.encode_sentences(corpus_texts)
        scores = cosine_similarity(query_emb, corpus_embs)
        return scores[0]
```

**Key Features**:
- L2 normalization for cosine similarity
- Batch processing for efficiency
- GPU acceleration support (if available)
- Cached embeddings for repeated queries

**Strengths**:
- Deep semantic understanding
- Handles synonyms and paraphrasing
- Context-aware embeddings
- Pre-trained on diverse text

**Limitations**:
- Slower than TF-IDF
- Requires PyTorch (~500MB)
- Less interpretable

---

#### 3.3 K-Means Clustering (`app/clustering.py`)

**Purpose**: Unsupervised grouping of career interests for exploratory analysis

**Supported Feature Spaces**:
1. **TF-IDF**: 3000-dimensional sparse vectors
2. **SBERT**: 384-dimensional dense embeddings

**Key Functions**:

1. **`tfidf_kmeans_cluster(texts, k, random_state)`**
   - Vectorizes texts using TF-IDF
   - Applies K-Means with k clusters
   - Returns cluster labels, vectorizer, kmeans model

2. **`sbert_kmeans_cluster(texts, k, random_state)`**
   - Embeds texts using SBERT
   - Applies K-Means with k clusters
   - Returns cluster labels, embedder, kmeans model

3. **`find_optimal_k(texts, method, k_range, metric)`**
   - Tests different k values
   - Computes silhouette score, inertia, or calinski-harabasz
   - Returns optimal k using elbow method

4. **`analyze_clusters(cluster_labels, texts, domain_labels)`**
   - Computes cluster sizes
   - Calculates cluster purity
   - Identifies representative samples

5. **`get_cluster_representatives(cluster_labels, embeddings, n_per_cluster)`**
   - Finds samples closest to cluster centroids
   - Returns representative texts for each cluster

**Evaluation Metrics**:
- **Silhouette Score**: Measures cluster cohesion and separation (-1 to 1)
- **Inertia**: Sum of squared distances to centroids (lower is better)
- **Purity**: Percentage of dominant domain in each cluster
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance

---

#### 3.4 Text Processing Pipeline (`app/nlp_pipeline.py`)

**Purpose**: Core NLP utilities for text preprocessing

**NLTK Components**:
- **punkt**: Sentence and word tokenization
- **stopwords**: English stopword list
- **wordnet**: Lemmatization dictionary

**Preprocessing Steps**:
```python
1. Lowercase conversion
2. Punctuation removal
3. Tokenization (word_tokenize)
4. Stopword filtering (English)
5. Lemmatization (WordNetLemmatizer)
6. Token cleaning (length > 2, alphanumeric)
```

**Key Functions**:
- `download_nltk_data()`: Auto-downloads NLTK resources
- `preprocess_text(text)`: Full preprocessing pipeline
- `tokenize_text(text)`: Returns clean token list
- `compute_similarity_scores(query, corpus)`: TF-IDF similarity computation

---

### 4. Data Layer

#### 4.1 Domain Dataset (`data/domain_dataset.csv`)

**Purpose**: Training data for domain classification

**Schema**:
```csv
text,label
"I love programming and building software...",software_engineering
"I enjoy working with patients and...",healthcare
```

**Statistics**:
- **Total Rows**: 187,405
- **Columns**: 2 (text, label)
- **Domains**: 48 unique labels
- **Avg Text Length**: ~150 words
- **Distribution**: Stratified across domains

**Data Sources**:
- Career descriptions from professional websites
- Job postings from employment platforms
- User-generated career interest statements
- Academic career guidance resources

**Data Quality**:
- Cleaned and deduplicated
- Balanced sampling (min 2000 samples per domain)
- Manual validation of 10% subset

---

#### 4.2 Careers Database (`data/careers_dataset.csv`)

**Purpose**: Comprehensive career profile database

**Schema**:
```csv
career,domain,description,skills,keywords,personality
Data Scientist,data_science,"Analyzes complex data...",
"Python,ML,Statistics,...","data,analytics,...","Openness,Conscientiousness"
```

**Statistics**:
- **Total Careers**: 767
- **Columns**: 6 attributes
- **Avg Skills per Career**: 8-12
- **Avg Keywords**: 10-15
- **Description Length**: 100-200 words

**Career Attributes**:

1. **career** (string): Official career title
   - Example: "Data Scientist", "UX Designer"
   
2. **domain** (string): Associated career domain
   - Maps to domain_dataset labels
   - Example: "data_science", "design"

3. **description** (string): Comprehensive career overview
   - Responsibilities and typical tasks
   - Work environment and context
   - Career prospects and growth

4. **skills** (comma-separated): Required competencies
   - Technical skills (Python, SQL, etc.)
   - Soft skills (communication, leadership)
   - Tools and technologies

5. **keywords** (comma-separated): Representative terms
   - Used for TF-IDF matching
   - Industry jargon and concepts
   - Alternative titles and synonyms

6. **personality** (comma-separated): Big Five traits
   - Ideal personality profile
   - Example: "Openness, Conscientiousness"

**Data Curation**:
- Sourced from O*NET, Bureau of Labor Statistics
- Expert review by career counselors
- Regular updates for emerging careers

---

#### 4.3 Trained Models (`models/`)

**4.3.1 Domain Classifier** (`domain_classifier.pkl`)

**Model Type**: Logistic Regression (scikit-learn)

**Hyperparameters**:
```python
{
    'C': 10.0,                    # Regularization strength
    'penalty': 'l2',              # L2 regularization
    'solver': 'lbfgs',            # Optimization algorithm
    'max_iter': 1000,             # Max iterations
    'multi_class': 'multinomial', # One-vs-Rest strategy
    'class_weight': 'balanced',   # Handle imbalanced classes
    'random_state': 42
}
```

**Training Details**:
- **Training Samples**: 149,924 (80%)
- **Test Samples**: 37,481 (20%)
- **Training Time**: ~45 seconds
- **Model Size**: 2.1 MB
- **Test Accuracy**: 99.30%

**Performance Metrics**:
- **Precision**: 0.993 (macro avg)
- **Recall**: 0.993 (macro avg)
- **F1-Score**: 0.993 (macro avg)
- **Confusion Matrix**: Near-perfect diagonal

---

**4.3.2 TF-IDF Vectorizer** (`domain_tfidf_vectorizer.pkl`)

**Type**: TfidfVectorizer (scikit-learn)

**Configuration**:
```python
{
    'max_features': 3000,
    'ngram_range': (1, 2),        # Unigrams + Bigrams
    'min_df': 2,                  # Min document frequency
    'sublinear_tf': True,         # Log scaling
    'lowercase': True,
    'stop_words': 'english'
}
```

**Vocabulary**:
- **Total Terms**: 3000
- **Unigrams**: ~2000
- **Bigrams**: ~1000
- **IDF Range**: 1.0 - 9.5

**Model Size**: 450 KB

---

## Data Flow

### End-to-End Recommendation Flow

```
User Input (Text)
      │
      ├─────────────────────────────────────────┐
      │                                         │
      ▼                                         ▼
[Domain Prediction]                      [Text Processing]
      │                                         │
      │ 1. Preprocess text                      │ 1. Tokenize
      │ 2. TF-IDF vectorize                     │ 2. Remove stopwords
      │ 3. Logistic Regression                  │ 3. Lemmatize
      │ 4. Get probabilities                    │
      │                                         │
      ├─────────────────────────────────────────┤
      │                                         │
      ▼                                         ▼
[Filter Careers by Domain]            [Personality Analysis]
      │                                         │
      │ - Load careers_dataset                  │ 1. Match keywords
      │ - Filter by predicted domain            │ 2. Score traits
      │                                         │ 3. Big Five scores
      │                                         │
      ▼                                         │
[Compute Similarity Scores]                     │
      │                                         │
      ├─► TF-IDF Similarity ────┐              │
      │                          │              │
      ├─► SBERT Similarity ──────┼─► [Hybrid]  │
      │                          │              │
      └──────────────────────────┘              │
      │                                         │
      ▼                                         │
[Calculate Component Scores] ◄─────────────────┘
      │
      │ 1. Similarity score (TF-IDF/SBERT/Hybrid)
      │ 2. Keyword overlap score
      │ 3. Personality match score
      │ 4. Skills match score
      │
      ▼
[Weighted Hybrid Scoring]
      │
      │ Hybrid = 0.40×Sim + 0.25×Kw + 0.20×Pers + 0.15×Skill
      │
      ▼
[Rank & Filter Top-K]
      │
      ▼
[Format Results]
      │
      │ - Career title
      │ - Domain
      │ - Match scores
      │ - Description
      │ - Skills
      │ - Explanation
      │
      ▼
[Display in UI]
```

---

## Module Dependencies

### Dependency Graph

```
app.py (Main UI)
    │
    ├─► domain_predictor.py
    │       └─► nlp_pipeline.py
    │       └─► data_loader.py
    │
    ├─► recommender.py
    │       ├─► nlp_pipeline.py (TF-IDF)
    │       ├─► sbert_semantic.py (SBERT)
    │       ├─► personality.py
    │       └─► data_loader.py
    │
    ├─► personality.py
    │       └─► nlp_pipeline.py
    │
    └─► data_loader.py
            └─► pandas

sbert_semantic.py
    └─► sentence_transformers
            └─► torch

clustering.py
    ├─► nlp_pipeline.py
    ├─► sbert_semantic.py
    └─► scikit-learn

nlp_pipeline.py
    ├─► nltk
    ├─► scikit-learn
    └─► numpy
```

### Import Hierarchy

**Level 0 (No Internal Dependencies)**:
- `data_loader.py`: Only depends on pandas
- `sbert_semantic.py`: Only depends on sentence-transformers

**Level 1 (Depends on Level 0)**:
- `nlp_pipeline.py`: Uses data_loader
- `personality.py`: Uses nlp_pipeline

**Level 2 (Depends on Level 1)**:
- `domain_predictor.py`: Uses nlp_pipeline, data_loader
- `recommender.py`: Uses nlp_pipeline, sbert_semantic, personality, data_loader
- `clustering.py`: Uses nlp_pipeline, sbert_semantic

**Level 3 (Top-level Application)**:
- `app.py`: Uses all above modules

---

## API Documentation

### Data Loader API (`data_loader.py`)

#### `load_domain_dataset()`
```python
def load_domain_dataset() -> pd.DataFrame:
    """
    Load domain training dataset.
    
    Returns:
        DataFrame with columns: ['text', 'label']
    
    Raises:
        FileNotFoundError: If dataset file missing
    """
```

#### `load_careers_dataset()`
```python
def load_careers_dataset() -> pd.DataFrame:
    """
    Load careers database.
    
    Returns:
        DataFrame with columns: ['career', 'domain', 'description', 
                                 'skills', 'keywords', 'personality']
    
    Raises:
        FileNotFoundError: If dataset file missing
    """
```

#### `filter_careers_by_domain(df, domain)`
```python
def filter_careers_by_domain(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """
    Filter careers by domain label.
    
    Args:
        df: Careers DataFrame
        domain: Domain label to filter by
    
    Returns:
        Filtered DataFrame with careers from specified domain
    """
```

---

### NLP Pipeline API (`nlp_pipeline.py`)

#### `preprocess_text(text)`
```python
def preprocess_text(text: str) -> str:
    """
    Full text preprocessing pipeline.
    
    Steps:
        1. Lowercase
        2. Remove punctuation
        3. Tokenize
        4. Remove stopwords
        5. Lemmatize
        6. Clean tokens
    
    Args:
        text: Raw input text
    
    Returns:
        Preprocessed text string
    """
```

#### `tokenize_text(text)`
```python
def tokenize_text(text: str) -> List[str]:
    """
    Tokenize and clean text.
    
    Args:
        text: Raw input text
    
    Returns:
        List of cleaned tokens
    """
```

#### `compute_similarity_scores(query_text, corpus_texts)`
```python
def compute_similarity_scores(
    query_text: str,
    corpus_texts: List[str]
) -> List[float]:
    """
    Compute TF-IDF cosine similarity scores.
    
    Args:
        query_text: User query text
        corpus_texts: List of documents to compare against
    
    Returns:
        List of similarity scores (0-1 range)
    """
```

---

### SBERT Embedder API (`sbert_semantic.py`)

#### `SBERTEmbedder.__init__(model_name)`
```python
def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
    """
    Initialize SBERT embedder.
    
    Args:
        model_name: Hugging Face model name
    """
```

#### `encode_sentences(texts)`
```python
def encode_sentences(self, texts: Union[str, List[str]]) -> np.ndarray:
    """
    Encode texts to embeddings.
    
    Args:
        texts: Single text or list of texts
    
    Returns:
        Normalized embeddings (N × 384)
    """
```

#### `batch_similarity_scores(query_text, corpus_texts)`
```python
def batch_similarity_scores(
    self,
    query_text: str,
    corpus_texts: List[str]
) -> np.ndarray:
    """
    Compute query-corpus similarities.
    
    Args:
        query_text: Query text
        corpus_texts: List of corpus texts
    
    Returns:
        Array of similarity scores (0-1 range)
    """
```

---

### Clustering API (`clustering.py`)

#### `tfidf_kmeans_cluster(texts, k, random_state)`
```python
def tfidf_kmeans_cluster(
    texts: List[str],
    k: int = 8,
    random_state: int = 42
) -> Tuple[np.ndarray, TfidfVectorizer, KMeans]:
    """
    K-Means clustering using TF-IDF features.
    
    Args:
        texts: List of text documents
        k: Number of clusters
        random_state: Random seed
    
    Returns:
        Tuple of (cluster_labels, vectorizer, kmeans_model)
    """
```

#### `find_optimal_k(texts, method, k_range, metric)`
```python
def find_optimal_k(
    texts: List[str],
    method: str = "tfidf",
    k_range: Tuple[int, int] = (2, 20),
    metric: str = "silhouette"
) -> Dict:
    """
    Find optimal number of clusters.
    
    Args:
        texts: List of documents
        method: "tfidf" or "sbert"
        k_range: (min_k, max_k) range to test
        metric: "silhouette", "inertia", or "calinski_harabasz"
    
    Returns:
        Dict with optimal_k, scores, and metric values
    """
```

---

### Domain Predictor API (`domain_predictor.py`)

#### `predict_domain_with_probabilities(text)`
```python
def predict_domain_with_probabilities(text: str) -> Dict:
    """
    Predict domain with full probability distribution.
    
    Args:
        text: User input text
    
    Returns:
        {
            'domain': str,              # Predicted domain label
            'confidence': float,        # Max probability (0-1)
            'probabilities': Dict[str, float]  # All domain probs
        }
    
    Raises:
        ValueError: If models not loaded
    """
```

---

### Recommender API (`recommender.py`)

#### `recommend_careers_for_domain(user_text, domain, top_k, similarity_model)`
```python
def recommend_careers_for_domain(
    user_text: str,
    domain: str,
    top_k: int = 5,
    similarity_model: Literal["tfidf", "sbert", "hybrid"] = "hybrid"
) -> List[Dict]:
    """
    Recommend careers using hybrid scoring.
    
    Args:
        user_text: User description
        domain: Predicted domain
        top_k: Number of recommendations
        similarity_model: Similarity engine choice
    
    Returns:
        List of recommendation dicts:
        {
            'career': str,
            'domain': str,
            'similarity_score': float,
            'tfidf_score': float,
            'sbert_score': float,
            'keyword_score': float,
            'personality_score': float,
            'skills_score': float,
            'hybrid_score': float,
            'description': str,
            'skills': str,
            'personality': str
        }
    """
```

---

## Performance Considerations

### Latency Analysis

| Component | Avg Time | Notes |
|-----------|----------|-------|
| Domain Prediction | 50-100ms | TF-IDF + LogReg |
| TF-IDF Similarity | 30-50ms | For 15 careers |
| SBERT Similarity | 150-300ms | For 15 careers |
| Hybrid Similarity | 200-350ms | TF-IDF + SBERT |
| Personality Analysis | 10-20ms | Keyword matching |
| Full Recommendation | 250-400ms | End-to-end (hybrid) |
| UI Rendering | 100-200ms | Streamlit + Plotly |

**Total User Experience**: ~500-600ms (hybrid mode)

### Optimization Strategies

1. **Lazy Loading**: SBERT model loaded only when selected
2. **Caching**: Streamlit `@st.cache_data` for dataset loading
3. **Batch Processing**: Vectorize all careers at once
4. **Sparse Matrices**: TF-IDF uses scipy sparse arrays
5. **GPU Acceleration**: SBERT supports CUDA (if available)

### Scalability

**Current Scale**:
- 48 domains
- 767 careers
- 187K training samples

**Scaling Bottlenecks**:
- TF-IDF: Linear with corpus size (efficient)
- SBERT: Linear with corpus size (slower)
- Domain Classifier: Constant time (already trained)

**Recommendations for 10× Scale** (7,670 careers):
- Implement approximate nearest neighbors (FAISS, Annoy)
- Pre-compute and cache SBERT embeddings
- Use inverted index for keyword filtering
- Distributed computing for clustering

---

## Deployment Architecture

### Local Development
```
Developer Machine
    └── Streamlit Dev Server (localhost:8501)
        ├── Hot reload on file changes
        ├── Debug mode enabled
        └── Direct file system access
```

### Production Deployment Options

#### Option 1: Streamlit Cloud
```
GitHub Repository
    └── Streamlit Cloud Service
        ├── Auto-deploy on git push
        ├── Free tier: 1GB RAM, 1 CPU
        └── HTTPS + custom domain
```

#### Option 2: Docker Container
```
Dockerfile
    ├── Python 3.11 base image
    ├── Install dependencies
    ├── Copy app + data + models
    ├── Expose port 8501
    └── CMD: streamlit run app/app.py

Deployment Targets:
    ├── AWS ECS / Fargate
    ├── Google Cloud Run
    ├── Azure Container Instances
    └── Heroku
```

#### Option 3: VM / Server
```
Linux Server (Ubuntu 20.04+)
    ├── Python 3.11 + venv
    ├── Nginx reverse proxy
    ├── systemd service
    └── SSL certificate (Let's Encrypt)
```

---

## Security Considerations

### Data Privacy
- No user data stored or logged
- Session-based processing only
- No external API calls (all local)

### Model Security
- Pre-trained models included in repo
- No dynamic model loading from URLs
- Input validation and sanitization

### Dependencies
- Regular security audits (Dependabot)
- Pin exact versions in requirements.txt
- Monitor CVE databases for vulnerabilities

---

## Testing Strategy

### Unit Tests (Planned)
```python
tests/
    ├── test_data_loader.py
    ├── test_nlp_pipeline.py
    ├── test_domain_predictor.py
    ├── test_recommender.py
    ├── test_personality.py
    ├── test_sbert_semantic.py
    └── test_clustering.py
```

### Integration Tests (Planned)
- End-to-end recommendation flow
- Model loading and prediction
- UI component rendering

### Performance Tests (Planned)
- Latency benchmarks
- Memory profiling
- Concurrent user simulation

---

## Future Architectural Enhancements

### Microservices Architecture
```
API Gateway (FastAPI)
    ├── Domain Prediction Service
    ├── Recommendation Service
    ├── Similarity Engine Service (TF-IDF)
    ├── Similarity Engine Service (SBERT)
    └── Personality Analysis Service

Streamlit Frontend ───► API Gateway
```

### Database Integration
- PostgreSQL for careers database
- Redis for caching embeddings
- Vector database (Pinecone/Weaviate) for SBERT

### MLOps Pipeline
- Model versioning (MLflow)
- A/B testing framework
- User feedback collection
- Continuous retraining

---

## Conclusion

This architecture provides:
- **Modularity**: Clear separation of concerns
- **Scalability**: Can handle 10× data growth with optimizations
- **Flexibility**: Easy to swap algorithms or add features
- **Maintainability**: Well-documented, typed Python code
- **Extensibility**: Clean APIs for future enhancements

For implementation details, see individual module docstrings and `README.md`.
