# app.py - Legal Document Classification Streamlit App
# Run with: streamlit run app.py

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os
import time

# Page configuration
st.set_page_config(
    page_title="Legal Document Classifier",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        color: #1E3A5F;
    }
    .info-box {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #b8d4e3;
        margin: 0.5rem 0;
        color: #1E3A5F;
    }
    .success-box {
        background: #e8f5e9;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #a5d6a7;
        margin: 0.5rem 0;
        color: #1b5e20;
    }
    .warning-box {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffcc80;
        margin: 0.5rem 0;
        color: #e65100;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .dark-text {
        color: #1E3A5F !important;
    }
    .project-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    .project-section h4 {
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .project-section p, .project-section li {
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Categories
CATEGORIES = [
    "Criminal Procedure", "Civil Rights", "First Amendment", "Due Process",
    "Privacy", "Attorneys", "Unions", "Economic Activity", "Judicial Power",
    "Federalism", "Interstate Relations", "Federal Taxation", "Miscellaneous",
    "Private Action"
]

# Category descriptions
CATEGORY_DESCRIPTIONS = {
    "Criminal Procedure": "Cases involving criminal justice processes, rights of defendants, search and seizure, Miranda rights",
    "Civil Rights": "Discrimination cases, equal protection, voting rights, civil liberties",
    "First Amendment": "Freedom of speech, press, religion, assembly, and petition",
    "Due Process": "Procedural and substantive due process rights under 5th and 14th Amendments",
    "Privacy": "Right to privacy, personal autonomy, reproductive rights",
    "Attorneys": "Legal profession regulation, attorney-client privilege, bar admission",
    "Unions": "Labor unions, collective bargaining, worker organization rights",
    "Economic Activity": "Business regulation, commerce, contracts, antitrust",
    "Judicial Power": "Court jurisdiction, judicial review, federal court authority",
    "Federalism": "Federal-state relations, state sovereignty, preemption",
    "Interstate Relations": "Relations between states, full faith and credit, interstate commerce",
    "Federal Taxation": "Federal tax law, IRS disputes, tax policy",
    "Miscellaneous": "Cases not fitting other categories",
    "Private Action": "Private party disputes, torts, property rights"
}

@st.cache_resource
def load_models():
    """Load the trained models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if saved models exist
    bert_path = 'saved_models/legal_bert_classifier'
    baseline_path = 'saved_models/baseline_tfidf_logreg.joblib'
    
    models_loaded = {'bert': False, 'baseline': False}
    bert_model, bert_tokenizer, baseline_model = None, None, None
    
    # Try to load Legal-BERT
    if os.path.exists(bert_path):
        try:
            bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
            bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path)
            bert_model.to(device)
            bert_model.eval()
            models_loaded['bert'] = True
        except Exception as e:
            st.warning(f"Could not load Legal-BERT: {e}")
    
    # Try to load baseline
    if os.path.exists(baseline_path):
        try:
            baseline_model = joblib.load(baseline_path)
            models_loaded['baseline'] = True
        except Exception as e:
            st.warning(f"Could not load baseline model: {e}")
    
    return bert_model, bert_tokenizer, baseline_model, device, models_loaded

def predict_bert(text, model, tokenizer, device, top_k=5):
    """Make prediction using Legal-BERT"""
    model.eval()
    
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)[0]
    
    probs_np = probs.cpu().numpy()
    top_indices = np.argsort(probs_np)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'category': CATEGORIES[idx],
            'confidence': float(probs_np[idx]),
            'index': idx
        })
    
    return results, probs_np

def predict_baseline(text, model, top_k=5):
    """Make prediction using baseline model"""
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    
    top_indices = np.argsort(proba)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'category': CATEGORIES[idx],
            'confidence': float(proba[idx]),
            'index': idx
        })
    
    return results, proba

def create_confidence_chart(probs, title="Prediction Confidence"):
    """Create a horizontal bar chart for confidence scores"""
    num_categories = len(CATEGORIES)
    
    # Ensure probabilities are numpy array with correct length
    probs = np.array(probs).flatten()
    if len(probs) < num_categories:
        probs = np.pad(probs, (0, num_categories - len(probs)))
    elif len(probs) > num_categories:
        probs = probs[:num_categories]
    
    df = pd.DataFrame({
        'Category': list(CATEGORIES),
        'Confidence': [float(p * 100) for p in probs]
    }).sort_values('Confidence', ascending=True)
    
    fig = px.bar(
        df, x='Confidence', y='Category',
        orientation='h',
        color='Confidence',
        color_continuous_scale='Viridis',
        title=title
    )
    fig.update_layout(
        height=500,
        showlegend=False,
        xaxis_title="Confidence (%)",
        yaxis_title="",
        coloraxis_showscale=False
    )
    fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
    
    return fig

def create_comparison_chart(bert_probs, baseline_probs):
    """Create comparison chart between models"""
    num_categories = len(CATEGORIES)
    
    # Ensure probabilities are numpy arrays and have correct length
    bert_probs = np.array(bert_probs).flatten()
    baseline_probs = np.array(baseline_probs).flatten()
    
    # Pad or truncate to match number of categories
    if len(bert_probs) < num_categories:
        bert_probs = np.pad(bert_probs, (0, num_categories - len(bert_probs)))
    elif len(bert_probs) > num_categories:
        bert_probs = bert_probs[:num_categories]
        
    if len(baseline_probs) < num_categories:
        baseline_probs = np.pad(baseline_probs, (0, num_categories - len(baseline_probs)))
    elif len(baseline_probs) > num_categories:
        baseline_probs = baseline_probs[:num_categories]
    
    # Create dataframe with explicit lists
    categories_list = list(CATEGORIES)
    bert_conf = [float(p * 100) for p in bert_probs]
    baseline_conf = [float(p * 100) for p in baseline_probs]
    
    df = pd.DataFrame({
        'Category': categories_list + categories_list,
        'Confidence': bert_conf + baseline_conf,
        'Model': ['Legal-BERT'] * num_categories + ['Baseline'] * num_categories
    })
    
    fig = px.bar(
        df, x='Category', y='Confidence', color='Model',
        barmode='group',
        title='Model Prediction Comparison',
        color_discrete_map={'Legal-BERT': '#667eea', 'Baseline': '#f093fb'}
    )
    fig.update_layout(
        height=400,
        xaxis_tickangle=-45,
        xaxis_title="",
        yaxis_title="Confidence (%)"
    )
    
    return fig

def main():
    # Header
    st.markdown('<p class="main-header">⚖️ Legal Document Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered classification of Court opinions into 14 legal categories</p>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading models..."):
        bert_model, bert_tokenizer, baseline_model, device, models_loaded = load_models()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        available_models = []
        if models_loaded['bert']:
            available_models.append("Legal-BERT")
        if models_loaded['baseline']:
            available_models.append("TF-IDF + LogReg (Baseline)")
        
        if not available_models:
            st.error("No models found! Please train and save models first.")
            st.info("Run the training notebook to create saved_models/")
            return
        
        if len(available_models) > 1:
            selected_model = st.selectbox("Select Model", available_models)
            compare_models = st.checkbox("Compare both models", value=True)
        else:
            selected_model = available_models[0]
            compare_models = False
        
        st.markdown("---")
        
        # Display device info
        st.subheader("📊 System Info")
        st.write(f"**Device:** {device}")
        st.write(f"**Legal-BERT:** {'✅ Loaded' if models_loaded['bert'] else '❌ Not found'}")
        st.write(f"**Baseline:** {'✅ Loaded' if models_loaded['baseline'] else '❌ Not found'}")
        
        st.markdown("---")
        
        # Category info
        st.subheader("📚 Categories")
        selected_cat = st.selectbox("Learn about category:", CATEGORIES)
        st.info(CATEGORY_DESCRIPTIONS[selected_cat])
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Classify Document", "📝 Sample Documents", "📈 Project Details"])
    
    with tab1:
        st.header("Enter Legal Document Text")
        
        # Initialize session state for text
        if 'text_input' not in st.session_state:
            st.session_state.text_input = ""
        
        # Text input
        text_input = st.text_area(
            "Paste your legal document text here:",
            value=st.session_state.text_input,
            height=250,
            placeholder="Enter the text of a legal document, court opinion, or case summary...",
            key="text_area_input"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            classify_btn = st.button("🔍 Classify", type="primary", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.text_input = ""
                st.rerun()
        
        if classify_btn and text_input.strip():
            with st.spinner("Analyzing document..."):
                start_time = time.time()
                
                # Get predictions
                bert_results, bert_probs = None, None
                baseline_results, baseline_probs = None, None
                
                if models_loaded['bert'] and (selected_model == "Legal-BERT" or compare_models):
                    bert_results, bert_probs = predict_bert(text_input, bert_model, bert_tokenizer, device)
                
                if models_loaded['baseline'] and (selected_model == "TF-IDF + LogReg (Baseline)" or compare_models):
                    baseline_results, baseline_probs = predict_baseline(text_input, baseline_model)
                
                inference_time = time.time() - start_time
            
            st.markdown("---")
            st.header("📊 Classification Results")
            
            # Show primary prediction
            if selected_model == "Legal-BERT" and bert_results:
                primary_results = bert_results
                primary_probs = bert_probs
            else:
                primary_results = baseline_results
                primary_probs = baseline_probs
            
            # Main prediction display
            top_pred = primary_results[0]
            
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="margin:0;">📋 {top_pred['category']}</h2>
                <p style="font-size: 1.5rem; margin: 0.5rem 0;">Confidence: {top_pred['confidence']*100:.1f}%</p>
                <p style="font-size: 0.9rem; opacity: 0.9;">Inference time: {inference_time*1000:.0f}ms</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Category description
            st.info(f"**About this category:** {CATEGORY_DESCRIPTIONS[top_pred['category']]}")
            
            # Top-5 predictions
            st.subheader("Top 5 Predictions")
            
            cols = st.columns(5)
            for i, result in enumerate(primary_results[:5]):
                with cols[i]:
                    st.metric(
                        label=f"#{i+1}",
                        value=f"{result['confidence']*100:.1f}%",
                        delta=result['category'][:15] + "..." if len(result['category']) > 15 else result['category']
                    )
            
            # Confidence chart
            st.subheader("Confidence Distribution")
            fig = create_confidence_chart(primary_probs, f"{selected_model} - All Categories")
            st.plotly_chart(fig, use_container_width=True)
            
            # Model comparison
            if compare_models and bert_probs is not None and baseline_probs is not None:
                st.markdown("---")
                st.subheader("🔄 Model Comparison")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Legal-BERT Prediction:**")
                    st.success(f"🏆 {bert_results[0]['category']} ({bert_results[0]['confidence']*100:.1f}%)")
                
                with col2:
                    st.markdown("**Baseline Prediction:**")
                    st.info(f"📊 {baseline_results[0]['category']} ({baseline_results[0]['confidence']*100:.1f}%)")
                
                # Agreement check
                if bert_results[0]['category'] == baseline_results[0]['category']:
                    st.success("✅ Both models agree on the classification!")
                else:
                    st.warning("⚠️ Models disagree - consider manual review")
                
                # Comparison chart
                fig_comp = create_comparison_chart(bert_probs, baseline_probs)
                st.plotly_chart(fig_comp, use_container_width=True)
        
        elif classify_btn:
            st.warning("Please enter some text to classify.")
    
    with tab2:
        st.header("📝 Sample Legal Documents")
        st.write("Click on a sample to load it for classification:")
        
        samples = {
            "Criminal Procedure Case": """The defendant was convicted of first-degree murder and sentenced to life imprisonment without the possibility of parole. The appellant challenges the conviction on grounds that the trial court erred in admitting certain evidence obtained through a warrantless search of his residence. The Fourth Amendment protects citizens against unreasonable searches and seizures, and the exclusionary rule requires suppression of evidence obtained in violation of constitutional rights. The prosecution argues that the evidence falls within the exigent circumstances exception to the warrant requirement. The Court must determine whether the warrantless entry was justified and whether the subsequent seizure of evidence was constitutionally permissible.""",
            
            "Civil Rights Case": """The plaintiff, an African American employee, alleges that the defendant corporation discriminated against her on the basis of race in violation of Title VII of the Civil Rights Act of 1964. The complaint states that despite satisfactory performance reviews and seniority, the plaintiff was passed over for promotion in favor of less qualified white candidates on multiple occasions. The employer's pattern and practice of discrimination created a hostile work environment that ultimately led to the plaintiff's constructive discharge. The district court granted summary judgment for the defendant, finding insufficient evidence of discriminatory intent.""",
            
            "Federal Taxation Case": """The petitioner seeks review of the Tax Court's decision upholding the Commissioner's determination of a deficiency in federal income tax. The dispute centers on whether certain payments received by the taxpayer constitute taxable income under Section 61 of the Internal Revenue Code or whether they qualify for exclusion under Section 104(a)(2) as damages received on account of personal physical injuries. The Commissioner disallowed the claimed exclusion and assessed additional taxes plus penalties. The taxpayer argues that the settlement payments were specifically allocated to physical injury claims and should be excluded from gross income.""",
            
            "First Amendment Case": """The petitioner, a public university student organization, challenges the university's denial of funding for their publication on grounds that it violates the First Amendment's guarantee of free speech. The university contends that its funding guidelines are viewpoint-neutral and that the denial was based on the publication's religious content rather than any particular viewpoint. The organization argues that the policy discriminates against religious viewpoints in a public forum created by the university's funding program. The Court must determine whether the university's funding decision constitutes unconstitutional viewpoint discrimination.""",
            
            "Economic Activity Case": """The Federal Trade Commission brings this action against the defendant corporation alleging violations of Section 5 of the Federal Trade Commission Act through deceptive trade practices and unfair methods of competition. The complaint alleges that the defendant engaged in price-fixing agreements with competitors, allocated markets among themselves, and made false advertising claims about their products. The defendant argues that its conduct falls within legitimate business practices and that the FTC lacks jurisdiction over certain aspects of the challenged conduct."""
        }
        
        for title, text in samples.items():
            with st.expander(f"📄 {title}"):
                st.write(text)
                if st.button(f"Load '{title}'", key=f"load_{title}"):
                    st.session_state.text_input = text
                    st.success(f"✅ Loaded! Switch to 'Classify Document' tab to analyze.")
                    st.rerun()
    
    with tab3:
        st.header("📈 Project Details: Legal Document Classification")
        
        # Project Overview
        st.markdown("""
        <div class="project-section">
            <h4>🎯 Project Objective</h4>
            <p>This project implements an AI-powered legal document classification system that automatically 
            categorizes U.S. Supreme Court opinions into 14 predefined legal issue areas. The goal is to assist 
            lawyers, legal researchers, and court systems by quickly organizing and retrieving documents, 
            saving time in legal research and document management.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Dataset Information
        st.subheader("📚 Dataset: SCOTUS (LexGLUE Benchmark)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #1E3A5F; margin:0;">Dataset Statistics</h4>
                <ul style="color: #333; margin-top: 0.5rem;">
                    <li><strong>Source:</strong> LexGLUE Benchmark (coastalcph/lex_glue)</li>
                    <li><strong>Total Samples:</strong> ~8,000+ Supreme Court opinions</li>
                    <li><strong>Training Set:</strong> ~5,000 documents</li>
                    <li><strong>Validation Set:</strong> ~1,400 documents</li>
                    <li><strong>Test Set:</strong> ~1,400 documents</li>
                    <li><strong>Classes:</strong> 14 legal issue areas</li>
                    <li><strong>Task Type:</strong> Single-label classification</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #1E3A5F; margin:0;">Why SCOTUS Dataset?</h4>
                <ul style="color: #333; margin-top: 0.5rem;">
                    <li>Publicly available via HuggingFace</li>
                    <li>Real ground-truth labels (not pseudo-labels)</li>
                    <li>Part of established LexGLUE benchmark</li>
                    <li>Covers diverse legal areas</li>
                    <li>High-quality annotated data</li>
                    <li>Suitable for single-label classification</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # The 14 Categories
        st.subheader("📋 The 14 Legal Categories")
        
        cat_data = []
        for i, cat in enumerate(CATEGORIES):
            cat_data.append({
                "ID": i,
                "Category": cat,
                "Description": CATEGORY_DESCRIPTIONS[cat]
            })
        
        st.dataframe(
            pd.DataFrame(cat_data),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        
        # Models Implemented
        st.subheader("🤖 Models Implemented")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="project-section">
                <h4>📊 Baseline: TF-IDF + Logistic Regression</h4>
                <p style="color: #333;"><strong>Architecture:</strong></p>
                <ul style="color: #333;">
                    <li><strong>Vectorizer:</strong> TF-IDF with 15,000 features</li>
                    <li><strong>N-grams:</strong> Unigrams and Bigrams (1,2)</li>
                    <li><strong>Min DF:</strong> 3 (ignore rare terms)</li>
                    <li><strong>Max DF:</strong> 0.95 (ignore very common terms)</li>
                    <li><strong>Sublinear TF:</strong> Enabled (log scaling)</li>
                    <li><strong>Classifier:</strong> Logistic Regression</li>
                    <li><strong>Class Weights:</strong> Balanced (handle imbalance)</li>
                    <li><strong>Solver:</strong> LBFGS</li>
                </ul>
                <p style="color: #333;"><strong>Strengths:</strong></p>
                <ul style="color: #333;">
                    <li>Fast training (~30 seconds)</li>
                    <li>No GPU required</li>
                    <li>Interpretable features</li>
                    <li>Good baseline performance</li>
                </ul>
                <p style="color: #333;"><strong>Limitations:</strong></p>
                <ul style="color: #333;">
                    <li>No semantic understanding</li>
                    <li>Relies on keyword matching</li>
                    <li>Cannot capture context</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="project-section">
                <h4>🧠 Advanced: Fine-tuned Legal-BERT</h4>
                <p style="color: #333;"><strong>Architecture:</strong></p>
                <ul style="color: #333;">
                    <li><strong>Base Model:</strong> nlpaueb/legal-bert-base-uncased</li>
                    <li><strong>Parameters:</strong> ~110 million</li>
                    <li><strong>Pre-training:</strong> Legal domain corpora</li>
                    <li><strong>Max Sequence Length:</strong> 512 tokens</li>
                    <li><strong>Fine-tuning Epochs:</strong> 3</li>
                    <li><strong>Learning Rate:</strong> 2e-5</li>
                    <li><strong>Optimizer:</strong> AdamW with weight decay</li>
                    <li><strong>Scheduler:</strong> Linear warmup (10%)</li>
                </ul>
                <p style="color: #333;"><strong>Strengths:</strong></p>
                <ul style="color: #333;">
                    <li>Understands legal terminology</li>
                    <li>Captures semantic relationships</li>
                    <li>Higher accuracy on complex cases</li>
                    <li>Pre-trained on legal corpora</li>
                </ul>
                <p style="color: #333;"><strong>Limitations:</strong></p>
                <ul style="color: #333;">
                    <li>Slower inference</li>
                    <li>Requires GPU for training</li>
                    <li>512 token limit truncates long documents</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Evaluation Metrics
        st.subheader("📏 Evaluation Metrics")
        
        st.markdown("""
        <div class="info-box">
            <h4 style="color: #1E3A5F;">Metrics Used for Evaluation</h4>
            <ul style="color: #333;">
                <li><strong>Accuracy:</strong> Percentage of correctly classified documents</li>
                <li><strong>Macro F1:</strong> Average F1 score across all classes (treats all classes equally, important for imbalanced datasets)</li>
                <li><strong>Micro F1:</strong> Global F1 score (equivalent to accuracy for single-label classification)</li>
                <li><strong>Per-class Precision/Recall/F1:</strong> Detailed performance for each legal category</li>
                <li><strong>Confusion Matrix:</strong> Visualization of classification errors and patterns</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Results Table
        st.subheader("📊 Model Performance Results")
        
        results_df = pd.DataFrame({
            "Model": ["TF-IDF + Logistic Regression", "Legal-BERT (Fine-tuned)"],
            "Accuracy": ["~74%", "~78%"],
            "Macro F1": ["~67%", "~72%"],
            "Micro F1": ["~74%", "~78%"],
            "Training Time": ["~30 seconds", "~15-20 minutes (GPU)"],
            "Inference Speed": ["~1ms/doc", "~50ms/doc"]
        })
        
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div class="success-box">
            <h4 style="color: #1b5e20;">Key Findings</h4>
            <ul style="color: #333;">
                <li>Legal-BERT outperforms baseline by <strong>~4-5%</strong> in accuracy and F1 scores</li>
                <li>Domain-specific pre-training significantly helps with legal terminology understanding</li>
                <li>Some categories (e.g., Criminal Procedure, Economic Activity) are easier to classify</li>
                <li>Confusion often occurs between semantically related categories (e.g., Civil Rights vs. Due Process)</li>
                <li>Both models struggle with the "Miscellaneous" category due to its heterogeneous nature</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Technical Implementation
        st.subheader("⚙️ Technical Implementation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #1E3A5F;">Training Pipeline</h4>
                <ol style="color: #333;">
                    <li>Load SCOTUS dataset from HuggingFace</li>
                    <li>Preprocess text (truncation to 10K chars)</li>
                    <li>Train TF-IDF baseline model</li>
                    <li>Tokenize with Legal-BERT tokenizer</li>
                    <li>Create PyTorch DataLoaders</li>
                    <li>Fine-tune Legal-BERT with cross-entropy loss</li>
                    <li>Apply gradient clipping (max norm = 1.0)</li>
                    <li>Save best model based on validation F1</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #1E3A5F;">Libraries & Tools Used</h4>
                <ul style="color: #333;">
                    <li><strong>PyTorch:</strong> Deep learning framework</li>
                    <li><strong>HuggingFace Transformers:</strong> Pre-trained models</li>
                    <li><strong>HuggingFace Datasets:</strong> Dataset loading</li>
                    <li><strong>Scikit-learn:</strong> Baseline model & metrics</li>
                    <li><strong>Pandas & NumPy:</strong> Data manipulation</li>
                    <li><strong>Matplotlib & Seaborn:</strong> Visualizations</li>
                    <li><strong>Streamlit:</strong> Web application</li>
                    <li><strong>Plotly:</strong> Interactive charts</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Challenges & Solutions
        st.subheader("🔧 Challenges & Solutions")
        
        st.markdown("""
        <div class="warning-box">
            <h4 style="color: #e65100;">Challenges Faced</h4>
            <table style="width:100%; color: #333;">
                <tr>
                    <th style="text-align:left; padding: 8px;">Challenge</th>
                    <th style="text-align:left; padding: 8px;">Solution</th>
                </tr>
                <tr>
                    <td style="padding: 8px;">Long legal documents (10K+ words)</td>
                    <td style="padding: 8px;">Truncate to 512 tokens for BERT; use first 10K chars for TF-IDF</td>
                </tr>
                <tr>
                    <td style="padding: 8px;">Class imbalance (some categories rare)</td>
                    <td style="padding: 8px;">Use balanced class weights in Logistic Regression</td>
                </tr>
                <tr>
                    <td style="padding: 8px;">Missing classes in validation set</td>
                    <td style="padding: 8px;">Explicitly specify labels parameter in all metrics</td>
                </tr>
                <tr>
                    <td style="padding: 8px;">Complex legal terminology</td>
                    <td style="padding: 8px;">Use domain-specific Legal-BERT instead of generic BERT</td>
                </tr>
                <tr>
                    <td style="padding: 8px;">GPU memory limitations</td>
                    <td style="padding: 8px;">Use batch size of 8; gradient accumulation if needed</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Practical Applications
        st.subheader("💼 Practical Applications")
        
        st.markdown("""
        <div class="success-box">
            <h4 style="color: #1b5e20;">Real-World Use Cases</h4>
            <ul style="color: #333;">
                <li><strong>Law Firms:</strong> Automatic document routing to relevant practice areas</li>
                <li><strong>Legal Research:</strong> Quick categorization of case law for research projects</li>
                <li><strong>Court Systems:</strong> Automated case classification for docket management</li>
                <li><strong>Legal Education:</strong> Help students understand case categorization</li>
                <li><strong>Document Management:</strong> Organize large legal document repositories</li>
                <li><strong>E-Discovery:</strong> Filter and categorize documents during litigation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Future Improvements
        st.subheader("🚀 Future Improvements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h4 style="color: #1E3A5F;">Model Enhancements</h4>
                <ul style="color: #333;">
                    <li>Use Longformer for longer documents</li>
                    <li>Implement multi-label classification</li>
                    <li>Add model interpretability (LIME/SHAP)</li>
                    <li>Ensemble multiple models</li>
                    <li>Fine-tune on more epochs</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h4 style="color: #1E3A5F;">Feature Additions</h4>
                <ul style="color: #333;">
                    <li>PDF/DOCX file upload support</li>
                    <li>Batch processing multiple documents</li>
                    <li>Export results to CSV/Excel</li>
                    <li>Add document summarization</li>
                    <li>Support other legal datasets (EUR-Lex)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Category distribution chart
        st.markdown("---")
        st.subheader("📊 Category Distribution in Dataset")
        
        cat_dist = pd.DataFrame({
            'Category': CATEGORIES,
            'Approximate %': [18, 15, 8, 7, 3, 2, 5, 15, 10, 6, 2, 4, 3, 2]
        })
        
        fig = px.pie(cat_dist, values='Approximate %', names='Category', 
                     title='Approximate Category Distribution in SCOTUS Dataset',
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Built with Streamlit • Models: Legal-BERT & TF-IDF + Logistic Regression</p>
        <p>Dataset: LexGLUE SCOTUS • 14 Legal Categories</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()