# ⚖️ Legal Document Classification System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

> **AI-Powered Classification of U.S. Supreme Court Opinions using Deep Learning**

Automatically categorize legal documents into 14 predefined legal issue areas using both traditional Machine Learning (TF-IDF + Logistic Regression) and state-of-the-art Deep Learning (Legal-BERT Transformer).

---

## 👥 Team Members

| Name | Student ID | Role |
|------|------------|------|
| **Kirtan Shah** | 22000428 | Deep Learning Model Development |
| **Naisargi Modi** | 22000397 | Data Processing & Baseline Model |
| **Mansha Sanger** | 22000396 | Web Application & Deployment |
| **Archi Prajapati** | 22000378 | Documentation, Testing & Research |

**Course:** Deep Learning  
**Institution:** Navrachana University  
**Submission Date:** December 2024

---

## 🎥 Project Demo Video


**[Click here to watch our complete project walkthrough on Loom →](https://www.loom.com/share/52ac39a201c34220a3be12becd7b2ef2))**

The video includes:
- Problem statement and motivation
- Live demo of the Streamlit application
- Model training process explanation
- Results analysis and comparison
- Technical challenges and solutions

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [Dataset](#-dataset)
- [Complete Development Pipeline](#-complete-development-pipeline)
- [Models Implemented](#-models-implemented)
- [Results](#-results)
- [Technology Stack](#-technology-stack)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Team Contributions](#-team-contributions)
- [Challenges Faced](#-challenges-faced)
- [Future Improvements](#-future-improvements)
- [References](#-references)
- [License](#-license)

---

## 🎯 Problem Statement

### The Challenge

Legal document management is a critical but time-consuming task in the legal industry:

- **Time Waste:** Lawyers spend **30-40% of their time** organizing and categorizing documents
- **High Costs:** Manual classification costs **$25-80 per document**
- **Volume:** Law firms handle **1,000+ documents daily**
- **Human Error:** Manual categorization has a **5-8% error rate**
- **Inefficiency:** 12 documents/hour manually vs 1,000+ documents/hour with AI

### Our Goal

Develop an AI-powered system that **automatically classifies legal documents** into predefined categories, helping lawyers and legal researchers quickly organize and retrieve documents, thereby:
- Reducing classification time by **80%**
- Cutting costs by **$50,000+ annually** for mid-size firms
- Improving accuracy and consistency

---

## 💡 Solution Overview

We built a **dual-model classification system** that:

1. **Compares Traditional ML vs Deep Learning** approaches
2. **Classifies Supreme Court opinions** into 14 legal issue areas
3. **Provides real-time predictions** via an interactive web application
4. **Achieves 78.9% accuracy** using Legal-BERT (state-of-the-art)

### Key Features

✅ **Two Classification Models:**
   - Baseline: TF-IDF + Logistic Regression (74.3% accuracy)
   - Advanced: Fine-tuned Legal-BERT (78.9% accuracy)

✅ **Interactive Web Application:**
   - Real-time document classification
   - Confidence score visualizations
   - Side-by-side model comparison
   - Sample legal documents for testing

✅ **Comprehensive Evaluation:**
   - Accuracy, Macro F1, Micro F1 metrics
   - Per-category performance analysis
   - Confusion matrix and error analysis

---

## 📚 Dataset

### SCOTUS (LexGLUE Benchmark)

We used the **SCOTUS dataset** from the LexGLUE benchmark, containing U.S. Supreme Court opinions.

| Attribute | Details |
|-----------|---------|
| **Source** | LexGLUE Benchmark (`coastalcph/lex_glue`) |
| **Total Documents** | ~8,000 Supreme Court opinions |
| **Training Set** | ~5,000 documents |
| **Validation Set** | ~1,400 documents |
| **Test Set** | ~1,400 documents |
| **Number of Classes** | 14 legal issue areas |
| **Task Type** | Single-label classification |
| **Labels** | Expert-annotated by legal scholars |
| **Average Document Length** | 4,000-6,000 words |

### 14 Legal Categories

1. **Criminal Procedure** - Search/seizure, Miranda rights, defendant rights
2. **Civil Rights** - Discrimination, equal protection, voting rights
3. **First Amendment** - Free speech, press, religion, assembly
4. **Due Process** - Procedural rights under 5th/14th Amendments
5. **Privacy** - Right to privacy, personal autonomy, reproductive rights
6. **Attorneys** - Legal profession regulation, attorney-client privilege
7. **Unions** - Labor unions, collective bargaining, worker rights
8. **Economic Activity** - Business regulation, antitrust, commerce
9. **Judicial Power** - Court jurisdiction, judicial review, federal authority
10. **Federalism** - Federal-state relations, state sovereignty, preemption
11. **Interstate Relations** - Relations between states, full faith and credit
12. **Federal Taxation** - Federal tax law, IRS disputes, tax policy
13. **Miscellaneous** - Cases not fitting other categories
14. **Private Action** - Private party disputes, torts, property rights

### Why SCOTUS Dataset?

- ✅ **Real Ground-Truth Labels:** Expert-annotated by legal professionals
- ✅ **Publicly Available:** Easy access via HuggingFace Datasets
- ✅ **Benchmark Standard:** Part of established LexGLUE benchmark
- ✅ **Diverse Topics:** Covers all major areas of U.S. law
- ✅ **High Quality:** Supreme Court opinions are well-written and structured

---

## 🔄 Complete Development Pipeline

Our end-to-end machine learning pipeline consisted of 6 major stages:

### **Stage 1: Data Collection** 📚
```python
from datasets import load_dataset

# Load SCOTUS dataset from HuggingFace
dataset = load_dataset("coastalcph/lex_glue", "scotus", trust_remote_code=True)
```
- Automatically downloaded 8,000 labeled documents
- Split into train/validation/test sets
- Verified data integrity and label distribution

### **Stage 2: Data Preprocessing** 🔧
```python
# Text preprocessing
- Truncate documents to first 10,000 characters
- Tokenize with Legal-BERT tokenizer (512 token limit)
- Encode labels (0-13) for 14 categories
- Handle missing values and edge cases
```
**Preprocessing Steps:**
- Text cleaning (remove special characters if needed)
- Tokenization for BERT (WordPiece tokenization)
- TF-IDF vectorization for baseline (15,000 features)
- Padding/truncation to fixed length (512 tokens)

### **Stage 3: Baseline Model Development** 📊
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_texts)

# Logistic Regression
clf = LogisticRegression(class_weight='balanced', max_iter=1000)
clf.fit(X_train, y_train)
```
**Baseline Configuration:**
- 15,000 TF-IDF features
- Unigrams + Bigrams (1-2 word phrases)
- Balanced class weights (handle imbalance)
- LBFGS solver, L2 regularization
- Training time: ~30 seconds on CPU

### **Stage 4: Advanced Model Development** 🧠
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load Legal-BERT
model = AutoModelForSequenceClassification.from_pretrained(
    "nlpaueb/legal-bert-base-uncased",
    num_labels=14
)

# Fine-tuning configuration
- Learning rate: 2e-5
- Optimizer: AdamW with weight decay (0.01)
- Epochs: 3
- Batch size: 8
- Scheduler: Linear warmup (10%) + decay
- Loss: Cross-entropy
```
**Legal-BERT Architecture:**
- 12 transformer layers
- 110 million parameters
- 768 hidden units
- Pre-trained on 12GB legal corpus
- Classification head: 768 → 14 classes

### **Stage 5: Model Evaluation** 📈
```python
# Evaluation metrics
- Accuracy: Overall correctness
- Macro F1: Average across all classes (fair to rare classes)
- Micro F1: Global performance
- Per-class Precision/Recall/F1
- Confusion matrix analysis
```
**Test Set Evaluation:**
- Ran both models on 1,400 held-out test documents
- Compared performance across all metrics
- Analyzed per-category strengths/weaknesses
- Identified common misclassification patterns

### **Stage 6: Deployment** 🚀
```python
import streamlit as st

# Streamlit web application
- Interactive UI with text input
- Real-time predictions from both models
- Confidence score visualizations (Plotly)
- Side-by-side model comparison
- Sample document library
```
**Deployment Features:**
- Lightweight web interface (Streamlit)
- Model loading with caching for speed
- Interactive Plotly charts
- Responsive design
- Session state management

---

## 🤖 Models Implemented

### Model 1: TF-IDF + Logistic Regression (Baseline)

**Algorithm Overview:**

**TF-IDF (Term Frequency-Inverse Document Frequency):**
```
TF(term, doc) = (Count of term in doc) / (Total terms in doc)
IDF(term) = log(Total documents / Documents containing term)
TF-IDF = TF × IDF
```

**How it works:**
1. Converts text into numerical vectors (15,000 dimensions)
2. Assigns high scores to important, distinctive words
3. Example: "plaintiff" in legal docs → High TF-IDF ✓
4. Example: "the", "and" → Low TF-IDF (too common)

**Logistic Regression:**
- Learns which TF-IDF features → which category
- Uses one-vs-rest classification for 14 classes
- Balanced class weights to handle imbalanced data

**Strengths:**
- ✅ Fast training (30 seconds)
- ✅ Fast inference (1ms per document)
- ✅ Interpretable (can see feature importance)
- ✅ Low memory footprint (~5MB)
- ✅ No GPU required

**Limitations:**
- ❌ No semantic understanding (keyword-based)
- ❌ Cannot capture context
- ❌ Treats words independently

### Model 2: Legal-BERT (Advanced)

**Algorithm Overview:**

**BERT (Bidirectional Encoder Representations from Transformers):**
- Reads text **bidirectionally** (both left and right context)
- Uses **self-attention mechanism** to understand relationships
- Example: In "rights were violated", BERT knows "rights" relates to "violated"

**Legal-BERT Specifics:**
- Pre-trained on 12GB of legal text (cases, statutes, contracts)
- Understands legal terminology: "habeas corpus", "prima facie", "mens rea"
- 110 million parameters trained to capture legal reasoning patterns

**Our Fine-Tuning Process:**
1. Loaded pre-trained Legal-BERT
2. Added classification head (768 → 14 outputs)
3. Fine-tuned on 5,000 SCOTUS documents
4. Used small learning rate (2e-5) to avoid "forgetting" legal knowledge
5. Trained for 3 epochs with AdamW optimizer
6. Saved best model based on validation F1 score

**Strengths:**
- ✅ Contextual understanding of text
- ✅ Domain-specific legal knowledge
- ✅ Captures semantic relationships
- ✅ State-of-the-art accuracy (78.9%)

**Limitations:**
- ❌ Slow training (18 minutes with GPU)
- ❌ Slow inference (52ms per document)
- ❌ Large model size (~440MB)
- ❌ Requires GPU for training
- ❌ 512 token limit (truncates long documents)

---

## 📊 Results

### Overall Performance Comparison

| Metric | TF-IDF + LogReg (Baseline) | Legal-BERT (Advanced) | Improvement |
|--------|---------------------------|----------------------|-------------|
| **Accuracy** | 74.3% | **78.9%** | +4.6% |
| **Macro F1** | 67.8% | **72.3%** | +4.5% |
| **Micro F1** | 74.3% | **78.9%** | +4.6% |
| **Training Time** | 30 seconds | 18 minutes | 36x slower |
| **Inference Speed** | 1 ms/doc | 52 ms/doc | 52x slower |
| **Model Size** | ~5 MB | ~440 MB | 88x larger |

### Key Insights

✅ **Legal-BERT achieves 5% better F1 score**
   - This means ~70 more documents correctly classified per 1,400 test documents
   - Significant improvement for production use

⚖️ **Trade-off: Accuracy vs Speed**
   - Baseline: Fast but less accurate (good for high-volume, low-stakes)
   - Legal-BERT: Slower but more accurate (good for critical decisions)

💡 **Domain-Specific Pre-training Matters**
   - Legal-BERT outperforms generic BERT by ~9-14%
   - Pre-training on legal text provides substantial domain knowledge

### Per-Category Performance (Legal-BERT F1 Scores)

**Top 5 Best Performing Categories:**
1. Criminal Procedure: **85.4%** F1 (clear vocabulary, most training data)
2. Economic Activity: **82.3%** F1 (well-defined legal domain)
3. First Amendment: **80.1%** F1 (distinctive free speech patterns)
4. Civil Rights: **78.5%** F1 (common in training data)
5. Judicial Power: **76.8%** F1 (clear jurisdictional language)

**Bottom 5 Challenging Categories:**
1. Miscellaneous: **51.2%** F1 (catch-all category, heterogeneous)
2. Interstate Relations: **58.9%** F1 (very few training examples)
3. Private Action: **62.1%** F1 (rare in Supreme Court)
4. Attorneys: **63.4%** F1 (overlaps with multiple areas)
5. Privacy: **65.6%** F1 (overlaps with Due Process)

### Error Analysis

**Most Common Misclassifications:**
- Due Process ↔ Civil Rights (23 cases) - Both involve constitutional rights
- Privacy ↔ Due Process (18 cases) - Privacy is subset of due process
- First Amendment ↔ Civil Rights (15 cases) - Free speech tied to civil liberties
- Attorneys ↔ Judicial Power (12 cases) - Both about legal system

**Root Causes:**
- Semantic overlap between related legal areas
- Limited training data for rare categories
- Long documents truncated to 512 tokens (information loss)

---

## 🛠️ Technology Stack

### Core ML/DL Frameworks
- **Python 3.10+** - Programming language
- **PyTorch 2.0+** - Deep learning framework
- **HuggingFace Transformers 4.35+** - Pre-trained models
- **HuggingFace Datasets** - Dataset loading
- **Scikit-learn 1.3+** - Baseline model & metrics

### Data Processing & Visualization
- **Pandas 2.0+** - Data manipulation
- **NumPy 1.24+** - Numerical computations
- **Matplotlib 3.7+** - Static visualizations
- **Seaborn 0.12+** - Statistical visualizations
- **Plotly 5.18+** - Interactive charts

### Deployment
- **Streamlit 1.28+** - Web application framework
- **Joblib 1.3+** - Model serialization
- **Google Colab** - GPU-accelerated training environment

### Development Tools
- **Git & GitHub** - Version control
- **Jupyter Notebooks** - Experimentation
- **VS Code** - Code editor

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- 8GB+ RAM (16GB recommended)
- GPU with CUDA (optional, for training)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/legal-document-classifier.git
cd legal-document-classifier
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**requirements.txt:**
```
streamlit>=1.28.0
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.18.0
joblib>=1.3.0
tqdm>=4.66.0
```

### Step 4: Download Pre-trained Models

**Option A: Use Our Trained Models**

If you have our `saved_models/` folder:
```bash
# Place the saved_models folder in project root
# Structure should be:
# project-root/
#   ├── saved_models/
#   │   ├── legal_bert_classifier/
#   │   └── baseline_tfidf_logreg.joblib
```

**Option B: Train Models from Scratch**

Open and run the training notebook in Google Colab:
```bash
# Upload legal-doc-classifier.ipynb to Google Colab
# Enable GPU: Runtime > Change runtime type > GPU
# Run all cells
# Download the saved_models/ folder
```

### Step 5: Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📖 Usage Guide

### Using the Web Application

1. **Launch the App:**
   ```bash
   streamlit run app.py
   ```

2. **Classify a Document:**
   - Go to the "🔍 Classify Document" tab
   - Paste your legal document text (up to 10,000 characters)
   - Click "🔍 Classify" button
   - View predictions from both models with confidence scores

3. **Try Sample Documents:**
   - Go to the "📝 Sample Documents" tab
   - Click "Load" on any pre-loaded legal case
   - Return to "Classify Document" tab to see it auto-filled

4. **Explore Project Details:**
   - Go to the "📈 Project Details" tab
   - Read about dataset, models, training process
   - View category descriptions

### Training Models from Scratch

**Open the Jupyter Notebook in Google Colab:**

```python
# 1. Upload legal-doc-classifier.ipynb to Colab
# 2. Enable GPU (Runtime > Change runtime type > GPU)
# 3. Run cells sequentially

# The notebook includes:
# - Dataset loading and exploration
# - Baseline model training (TF-IDF + LogReg)
# - Legal-BERT fine-tuning
# - Comprehensive evaluation
# - Model saving
```

**Key Training Parameters:**

```python
# Baseline
TF-IDF: max_features=15000, ngram_range=(1,2)
LogReg: class_weight='balanced', max_iter=1000

# Legal-BERT
learning_rate = 2e-5
batch_size = 8
epochs = 3
optimizer = AdamW(weight_decay=0.01)
```

### Using Models Programmatically

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib

# Load Legal-BERT
tokenizer = AutoTokenizer.from_pretrained('saved_models/legal_bert_classifier')
model = AutoModelForSequenceClassification.from_pretrained('saved_models/legal_bert_classifier')

# Load Baseline
baseline = joblib.load('saved_models/baseline_tfidf_logreg.joblib')

# Classify a document
text = "Your legal document text here..."

# Baseline prediction
baseline_pred = baseline.predict([text])[0]
baseline_proba = baseline.predict_proba([text])[0]

# Legal-BERT prediction
inputs = tokenizer(text, truncation=True, max_length=512, return_tensors='pt')
outputs = model(**inputs)
bert_pred = outputs.logits.argmax().item()
bert_proba = torch.softmax(outputs.logits, dim=1)[0]

print(f"Baseline: {CATEGORIES[baseline_pred]} ({baseline_proba[baseline_pred]:.2f})")
print(f"Legal-BERT: {CATEGORIES[bert_pred]} ({bert_proba[bert_pred]:.2f})")
```

---

## 📁 Project Structure

```
legal-document-classifier/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
│
├── notebooks/
│   └── legal-doc-classifier.ipynb  # Training notebook (Google Colab)
│
├── saved_models/
│   ├── legal_bert_classifier/      # Fine-tuned Legal-BERT
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── special_tokens_map.json
│   │   └── vocab.txt
│   └── baseline_tfidf_logreg.joblib  # TF-IDF + LogReg model
│
├── results/
│   ├── class_distribution.png      # Dataset visualizations
│   ├── baseline_confusion_matrix.png
│   ├── legalbert_confusion_matrix.png
│   ├── model_comparison.png
│   ├── per_class_comparison.png
│   └── training_curves.png
│
├── docs/
│   ├── presentation.html           # Project presentation (Reveal.js)
│   ├── deployment_guide.md         # Deployment instructions
│   └── algorithm_explanations.md   # Detailed algorithm docs
│
└── tests/
    ├── test_models.py              # Unit tests for models
    └── test_app.py                 # Tests for Streamlit app
```

---

## 👥 Team Contributions

### Kirtan Shah [22000428] - Deep Learning Model Development

**Responsibilities:**
- Legal-BERT model architecture design and implementation
- Fine-tuning pipeline with PyTorch
- Hyperparameter tuning (learning rate, batch size, epochs)
- Training loop with AdamW optimizer and learning rate scheduling
- Gradient clipping and weight decay implementation
- GPU memory management and optimization
- Model evaluation metrics calculation
- Early stopping and checkpoint saving

**Key Achievements:**
- Successfully fine-tuned 110M parameter model
- Achieved 78.9% accuracy on test set
- Optimized training to run efficiently on Colab GPU

---

### Naisargi Modi [22000397] - Data Processing & Baseline Model

**Responsibilities:**
- Dataset loading from HuggingFace
- Exploratory data analysis on 8,000 documents
- Data preprocessing and cleaning
- Train/validation/test split
- TF-IDF vectorization (15,000 features)
- Logistic Regression implementation with scikit-learn
- Performance comparison between models
- Confusion matrix generation and error analysis
- Per-class performance analysis

**Key Achievements:**
- Built strong baseline achieving 74.3% accuracy
- Identified key misclassification patterns
- Documented which categories are hardest to classify

---

### Mansha Sanger [22000396] - Web Application & Deployment

**Responsibilities:**
- Complete Streamlit application design and development
- Interactive UI with three main tabs
- Real-time prediction interface
- Interactive Plotly visualizations (confidence charts)
- Side-by-side model comparison feature
- Sample document library curation
- Session state management
- User experience optimization
- Deployment configuration

**Key Achievements:**
- Built intuitive, responsive web interface
- Implemented real-time dual-model comparison
- Created beautiful interactive visualizations

---

### Archi Prajapati [22000378] - Documentation, Testing & Research

**Responsibilities:**
- Literature review on Legal-BERT and transformers
- Research on LexGLUE benchmark
- Comprehensive code documentation
- README creation with detailed instructions
- Results documentation and report writing
- Code testing and quality assurance
- Unit tests for models and app
- Presentation preparation (slides, talking points)
- Deployment guide creation
- Loom video recording and editing

**Key Achievements:**
- Created comprehensive documentation
- Ensured code quality and reproducibility
- Prepared professional presentation materials

---

### Collaborative Efforts

**All team members participated in:**
- Weekly progress meetings and brainstorming sessions
- Debugging complex issues (GPU errors, label mismatches, model convergence)
- Design decisions (model selection, app features, UI/UX)
- Code reviews and pair programming
- Final integration testing
- Presentation rehearsal and refinement

**Communication Tools:**
- **GitHub:** Version control and code collaboration
- **Google Colab:** Shared notebook development
- **Slack/Discord:** Daily team communication
- **Google Docs:** Shared documentation
- **Zoom:** Weekly team meetings
- **Loom:** Video recording and sharing

---

## 🚧 Challenges Faced

### Technical Challenges

1. **Long Documents (10K+ words)**
   - **Problem:** BERT has 512 token limit, most legal docs are much longer
   - **Solution:** Truncated to first 512 tokens, accepted some information loss
   - **Future:** Use Longformer (4096 tokens) or hierarchical attention

2. **Class Imbalance**
   - **Problem:** Some categories had 1000+ examples, others had <50
   - **Solution:** Used balanced class weights in Logistic Regression
   - **Impact:** Improved minority class performance

3. **GPU Memory Limits**
   - **Problem:** Out-of-memory errors with batch size > 8
   - **Solution:** Reduced batch size to 8, used gradient accumulation
   - **Trade-off:** Slower training but stable convergence

4. **Missing Classes in Validation Set**
   - **Problem:** Scikit-learn metrics failed when not all classes present
   - **Solution:** Explicitly specified `labels=range(14)` in all metric functions
   - **Learning:** Always validate data split includes all classes

5. **Model Size (440MB)**
   - **Problem:** Large model difficult to deploy and share
   - **Solution:** Used model quantization consideration, Git LFS for storage
   - **Future:** Explore model distillation for smaller size

### Team & Coordination Challenges

1. **Learning Curve with Transformers**
   - **Challenge:** Team initially unfamiliar with BERT architecture
   - **Solution:** Studied papers, HuggingFace tutorials, experimented with code
   - **Outcome:** Gained deep understanding of attention mechanisms

2. **Understanding Legal Terminology**
   - **Challenge:** Complex legal jargon difficult to understand
   - **Solution:** Researched legal concepts, consulted online resources
   - **Impact:** Better understanding of why model makes certain predictions

3. **Remote Collaboration**
   - **Challenge:** Coordinating work across different time zones/schedules
   - **Solution:** Daily standup messages, shared Google Docs, GitHub branches
   - **Tools:** Slack, Zoom, GitHub Projects for task tracking

4. **Debugging GPU Errors in Colab**
   - **Challenge:** Intermittent CUDA errors, session timeouts
   - **Solution:** Implemented checkpointing, saved intermediate results
   - **Learning:** Always save progress frequently in cloud environments

5. **Balancing with Coursework**
   - **Challenge:** Managing project alongside other courses and exams
   - **Solution:** Clear task division, realistic deadlines, regular check-ins
   - **Outcome:** Successfully delivered complete project on time

---

## 🚀 Future Improvements

### Model Enhancements

1. **Longformer for Full Documents**
   - Handle 4096+ tokens instead of 512
   - Capture complete document context
   - Expected improvement: +2-3% accuracy

2. **Multi-Label Classification**
   - Allow documents to belong to multiple categories
   - More realistic for complex legal cases
   - Use sigmoid output + binary cross-entropy loss

3. **Ensemble Methods**
   - Combine Legal-BERT, RoBERTa, and baseline predictions
   - Voting or stacking ensemble
   - Expected improvement: +1-2% accuracy

4. **Model Interpretability (LIME/SHAP)**
   - Explain which words/phrases influenced classification
   - Build trust with legal professionals
   - Identify model biases

5. **Active Learning**
   - Identify uncertain predictions for manual review
   - Iteratively improve model with human feedback
   - Reduce labeling costs

### Feature Additions

1. **File Upload Support**
   - Accept PDF and DOCX files directly
   - Extract text using PyPDF2 or python-docx
   - More user-friendly for lawyers

2. **Batch Processing**
   - Classify multiple documents at once
   - Export results to CSV/Excel
   - Process entire case folders

3. **Document Summarization**
   - Generate concise summaries of long documents
   - Use T5 or BART models
   - Help lawyers quickly understand cases

4. **Citation Extraction**
   - Identify and extract cited cases and statutes
   - Link to relevant precedents
   - Build knowledge graph of legal citations

5. **Historical Trend Analysis**
   - Analyze changes in legal topics over time
   - Visualize shifts in Supreme Court focus
   - Research tool for legal scholars

### Dataset Expansion

1. **Train on EUR-Lex (65K multi-label documents)**
2. **Add state court opinions for broader coverage**
3. **Include contracts and legal memoranda**
4. **Multi-jurisdiction support (UK, EU, Canada)**

---

## 📚 References

### Academic Papers

1. **Legal-BERT:**
   - Chalkidis, I., Fergadiotis, M., Malakasiotis, P., Aletras, N., & Androutsopoulos, I. (2020). *LEGAL-BERT: The Muppets straight out of Law School.* Findings of EMNLP 2020.
   - [arXiv:2010.02559](https://arxiv.org/abs/2010.02559)
