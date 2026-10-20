import argparse
import sys
import os
import requests
import re
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from pathlib import Path
import pandas as pd
import numpy as np
import torch
import json
import joblib
from dotenv import load_dotenv

# Workarounds for broken torch companion libraries on Windows
sys.modules['torchvision'] = None
sys.modules['torchaudio'] = None
from transformers import AutoTokenizer, AutoModel
from xgboost import XGBClassifier

def get_mda_text(text):
    doc_match = re.search(r'<DOCUMENT>.*?</DOCUMENT>', text, flags=re.IGNORECASE | re.DOTALL)
    if doc_match:
        text = doc_match.group(0)

    try:
        # Use lxml for better fault tolerance with broken tags
        soup = BeautifulSoup(text, 'lxml')
        # Preserve paragraph breaks by replacing block elements with newlines
        for br in soup.find_all(['br', 'p', 'div']):
            br.replace_with('\n' + br.get_text() + '\n')
        clean_text = soup.get_text(separator=' ', strip=True)
        # Normalize excessive whitespace but keep single newlines
        clean_text = re.sub(r'[ \t]+', ' ', clean_text)
        clean_text = re.sub(r'\n\s*\n', '\n', clean_text)
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return ""
    
    # Semantic Lookahead Regex: By looking for the actual words in the next section's title
    # (e.g. 'quantitative' or 'financial'), we completely avoid the need for \n and ignore inline references.
    pattern_10k = re.compile(r"(?i)item[\s\W]+7[\.\:\s\-\–]+management.*?discussion.*?(?=(item[\s\W]+7a[\.\:\s\-\–]+quantitative|item[\s\W]+8[\.\:\s\-\–]+financial|\Z))", re.DOTALL)
    matches_10k = [m.group(0) for m in pattern_10k.finditer(clean_text) if len(m.group(0)) > 2000]
    
    if matches_10k:
        return max(matches_10k, key=len)
        
    pattern_10q = re.compile(r"(?i)item[\s\W]+2[\.\:\s\-\–]+management.*?discussion.*?(?=(item[\s\W]+3[\.\:\s\-\–]+quantitative|item[\s\W]+4[\.\:\s\-\–]+controls|\Z))", re.DOTALL)
    matches_10q = [m.group(0) for m in pattern_10q.finditer(clean_text) if len(m.group(0)) > 2000]
    
    if matches_10q:
        return max(matches_10q, key=len)
        
    print("Warning: Regex missed. Falling back to chunking.")
    words = clean_text.split()
    return " ".join(words[:max(2000, len(words)//5)])

def load_lm_dictionary(dict_path):
    df = pd.read_csv(dict_path)
    cats = ['Negative', 'Uncertainty', 'Litigious', 'Constraining']
    word_lists = {cat: set() for cat in cats}
    
    for cat in cats:
        if cat in df.columns:
            words = df[df[cat] > 0]['Word'].dropna().astype(str).tolist()
            word_lists[cat] = set([w.lower() for w in words])
            
    return word_lists

def compute_lm_features(text, word_lists):
    if not isinstance(text, str) or not text.strip():
        return [0.0, 0.0, 0.0, 0.0]
        
    words = re.findall(r'\b[A-Za-z]+\b', text.lower())
    total_words = len(words)
    
    if total_words == 0:
        return [0.0, 0.0, 0.0, 0.0]
        
    counts = {cat: 0 for cat in word_lists.keys()}
    for w in words:
        for cat, w_set in word_lists.items():
            if w in w_set:
                counts[cat] += 1
                
    return [
        counts['Negative'] / total_words, 
        counts['Uncertainty'] / total_words, 
        counts['Litigious'] / total_words, 
        counts['Constraining'] / total_words
    ]

def compute_complexity_features(text):
    if not isinstance(text, str) or not text.strip():
        return [0.0, 0.0, 0.0, 0.0]
    sentences = re.split(r'[\.\?\!]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = re.findall(r'\b[A-Za-z]+\b', text)
    total_words = len(words)
    total_sentences = max(1, len(sentences))
    if total_words == 0:
        return [0.0, 0.0, 0.0, 0.0]
    avg_sentence_len = total_words / total_sentences
    complex_words = sum(1 for w in words if len(re.findall(r'[aeiouy]{1,2}', w.lower())) >= 3)
    complex_word_ratio = complex_words / total_words
    fog_index = 0.4 * (avg_sentence_len + 100.0 * complex_word_ratio)
    log_word_count = float(np.log1p(total_words))
    return [avg_sentence_len / 100.0, complex_word_ratio, fog_index / 100.0, log_word_count / 15.0]

def get_finbert_embedding(text):
    print("Loading ProsusAI/finbert model...")
    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    model = AutoModel.from_pretrained('ProsusAI/finbert', use_safetensors=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"Extracting Dual-Pooled FinBERT embedding on {device}...")
    with torch.no_grad():
        if not isinstance(text, str) or not text.strip():
            return np.zeros(1536)
            
        tokens = tokenizer(text, return_tensors='pt', truncation=False, padding=False)
        input_ids = tokens['input_ids'][0]
        
        max_len = 512
        stride = 256
        
        chunk_embeddings = []
        for j in range(0, len(input_ids), stride):
            chunk_ids = input_ids[j:j+max_len].unsqueeze(0).to(device)
            if chunk_ids.size(1) < 10:
                continue
            
            outputs = model(input_ids=chunk_ids)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            chunk_embeddings.append(cls_embedding)
            
        if chunk_embeddings:
            chunk_arr = np.array(chunk_embeddings)
            doc_mean = np.mean(chunk_arr, axis=0)
            doc_max = np.max(chunk_arr, axis=0)
            return np.hstack([doc_mean, doc_max])
        else:
            return np.zeros(1536)

def main():
    parser = argparse.ArgumentParser(description="Predict restatement risk for a given SEC EDGAR filing URL.")
    parser.add_argument("url", type=str, help="The URL to the SEC EDGAR filing (e.g., .htm or .txt)")
    args = parser.parse_args()

    # The shipped model was trained on an amendment proxy with stale features.
    # Fail before requesting an SEC filing or presenting its score as risk.
    raise RuntimeError(
        "Live restatement scoring is unavailable: the existing model uses amendment "
        "proxy labels and unverified feature artifacts."
    )
    
    load_dotenv()
    user_agent = os.getenv("SEC_USER_AGENT", "Your Name contact@domain.com")
    headers = {"User-Agent": user_agent}
    
    print(f"\n--- STEP 1: Fetching Filing ---")
    print(f"URL: {args.url}")
    try:
        resp = requests.get(args.url, headers=headers, timeout=15)
        resp.raise_for_status()
        raw_text = resp.text
    except Exception as e:
        print(f"Failed to fetch URL: {e}")
        sys.exit(1)
        
    mda_text = get_mda_text(raw_text)

    print(f"Extracted {len(mda_text)} characters for analysis.")
    
    print(f"\n--- STEP 2: Feature Engineering ---")
    data_dir = Path("data")
    models_dir = Path("models")
    
    dict_file_exact = data_dir / "dictionaries" / "Loughran-McDonald_MasterDictionary_1993-2023.csv"
    dict_file_short = data_dir / "dictionaries" / "Loughran-McDonald.csv"
    
    if dict_file_exact.exists():
        dict_path = dict_file_exact
    elif dict_file_short.exists():
        dict_path = dict_file_short
    else:
        csvs = list((data_dir / "dictionaries").glob("*.csv"))
        if csvs:
            dict_path = csvs[0]
        else:
            print("Error: Could not find Loughran-McDonald dictionary in data/dictionaries/")
            sys.exit(1)
            
    word_lists = load_lm_dictionary(dict_path)
    lm_features = compute_lm_features(mda_text, word_lists)
    comp_features = compute_complexity_features(mda_text)
    
    # Infer metadata from URL/text
    is_10k = 1.0 if '10-k' in args.url.lower() or '10-k' in raw_text[:2000].lower() else 0.0
    meta_features = [is_10k, 0.5, 0.5, 0.5, 0.5, 0.0] # neutral defaults for live inference
    
    # Load TF-IDF vectorizer
    tfidf_path = models_dir / "tfidf_vectorizer.pkl"
    if tfidf_path.exists():
        tfidf = joblib.load(tfidf_path)
        tfidf_features = tfidf.transform([mda_text]).toarray()[0]
    else:
        print(f"Error: TF-IDF vectorizer not found at {tfidf_path}. Train model first.")
        sys.exit(1)
        
    bert_features = get_finbert_embedding(mda_text)
    
    X = np.hstack([lm_features, comp_features, meta_features, tfidf_features, bert_features]).reshape(1, -1)
    
    print(f"\n--- STEP 3: Model Scoring ---")
    
    meta_path = models_dir / "best_model_meta.json"
    
    model_type = "XGBoost"
    threshold = 0.30
    
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            model_type = meta.get("model_type", "XGBoost")
            threshold = meta.get("threshold", 0.30)
    
    print(f"Loading best model: {model_type} (Decision Threshold: {threshold:.2f})")
    
    if model_type == "XGBoost":
        model_path = models_dir / "best_risk_model.json"
        if not model_path.exists():
            print(f"Error: Model not found at {model_path}")
            sys.exit(1)
        model = XGBClassifier()
        model.load_model(str(model_path))
    else:
        model_path = models_dir / "best_risk_model.pkl"
        if not model_path.exists():
            print(f"Error: Model not found at {model_path}")
            sys.exit(1)
        model = joblib.load(model_path)
    
    # Predict Probability
    prob = model.predict_proba(X)[0, 1]
    
    print(f"\n{'='*50}")
    print("FINANCIAL RISK DETECTOR INFERENCE")
    print(f"{'='*50}")
    print(f"Target URL: {args.url}")
    print(f"Risk Score: {prob * 100:.2f}%")
    
    if prob >= threshold:
        print("STATUS:     HIGH RISK FLAG (Review Required)")
    else:
        print("STATUS:     NORMAL")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
