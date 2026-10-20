import pandas as pd
import numpy as np
from pathlib import Path
import os
import torch
import sys
sys.modules['torchvision'] = None  # Workaround for torchvision::nms RuntimeError
sys.modules['torchaudio'] = None   # Workaround for torchaudio libtorchaudio.pyd crash
from transformers import AutoTokenizer, AutoModel
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

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
    # Syllable approximation: count vowel groups of 3+ letters
    complex_words = sum(1 for w in words if len(re.findall(r'[aeiouy]{1,2}', w.lower())) >= 3)
    complex_word_ratio = complex_words / total_words
    fog_index = 0.4 * (avg_sentence_len + 100.0 * complex_word_ratio)
    log_word_count = float(np.log1p(total_words))
    
    return [avg_sentence_len / 100.0, complex_word_ratio, fog_index / 100.0, log_word_count / 15.0]

def extract_metadata_features(df):
    feats = []
    for _, row in df.iterrows():
        is_10k = 1.0 if str(row.get('form', '')).upper() == '10-K' else 0.0
        wksi = 1.0 if str(row.get('wksi', '0')) == '1' else 0.0
        
        # Accelerated Filer Status
        afs_str = str(row.get('afs', '')).upper()
        afs_val = 0.0
        if '1-LAF' in afs_str: afs_val = 1.0
        elif '2-ACC' in afs_str: afs_val = 0.8
        elif '3-SRA' in afs_str: afs_val = 0.6
        elif '4-NON' in afs_str: afs_val = 0.4
        elif '5-SRO' in afs_str: afs_val = 0.2
        
        # SIC sector division (first digit)
        sic_str = re.sub(r'\D', '', str(row.get('sic', '0')))
        sic_div = float(sic_str[0]) / 10.0 if sic_str else 0.0
        
        # Quarter from period (YYYYMMDD)
        period_str = str(row.get('period', ''))
        quarter = 0.5
        if len(period_str) >= 6:
            month = int(period_str[4:6]) if period_str[4:6].isdigit() else 6
            quarter = (month // 3) / 4.0
            
        changed = 1.0 if pd.notna(row.get('changed', None)) and str(row.get('changed', '')).strip() != '' else 0.0
        
        feats.append([is_10k, wksi, afs_val, sic_div, quarter, changed])
    return np.array(feats)

def get_finbert_embeddings(texts):
    print("Loading ProsusAI/finbert model...")
    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    model = AutoModel.from_pretrained('ProsusAI/finbert', use_safetensors=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    mean_embeddings = []
    max_embeddings = []
    
    print(f"Extracting Dual-Pooled (Mean + Max) FinBERT embeddings on {device}...")
    from tqdm import tqdm
    with torch.no_grad():
        for text in tqdm(texts, desc="Extracting FinBERT Embeddings"):
            if not isinstance(text, str) or not text.strip():
                mean_embeddings.append(np.zeros(768))
                max_embeddings.append(np.zeros(768))
                continue
                
            # Tokenize the whole document
            tokens = tokenizer(text, return_tensors='pt', truncation=False, padding=False)
            input_ids = tokens['input_ids'][0]
            
            # Slide window (max length 512, stride 256 for 50% overlap)
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
                doc_max = np.max(chunk_arr, axis=0) # Peak anomaly spike across the disclosure
            else:
                doc_mean = np.zeros(768)
                doc_max = np.zeros(768)
                
            mean_embeddings.append(doc_mean)
            max_embeddings.append(doc_max)
                
    return np.hstack([np.array(mean_embeddings), np.array(max_embeddings)])

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Force recomputation of embeddings')
    args = parser.parse_args()
    
    print("Running 03_feature_engine.py (Multimodal Forensic Pipeline)...")
    data_dir = Path("data")
    corpus_file = data_dir / "mda_corpus.csv"
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    x_file = data_dir / "X.npy"
    y_file = data_dir / "y.npy"
    
    feature_index = data_dir / "feature_index.csv"
    if x_file.exists() and y_file.exists() and feature_index.exists() and not args.force:
        print("Feature matrix and accession index already exist. Use --force to recompute.")
        return
        
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
            raise FileNotFoundError("Could not find Loughran-McDonald dictionary in data/dictionaries/")
            
    print(f"Loading dictionary from {dict_path}...")
    word_lists = load_lm_dictionary(dict_path)
    
    if not corpus_file.exists():
        raise FileNotFoundError(f"{corpus_file} not found.")
        
    df = pd.read_csv(corpus_file, low_memory=False)
    df = df[df['mda_text'].notna() & (df['mda_text'].astype(str).str.strip() != '') & (df['mda_text'].astype(str) != 'nan')].reset_index(drop=True)
    print(f"Loaded {len(df):,} filings with valid MD&A text.")
    
    print("Computing LM sentiment proportions...")
    lm_features = np.array([compute_lm_features(t, word_lists) for t in df['mda_text']])
    print(f"  -> LM shape: {lm_features.shape}")
    
    print("Computing linguistic complexity & obfuscation metrics...")
    complex_features = np.array([compute_complexity_features(t) for t in df['mda_text']])
    print(f"  -> Complexity shape: {complex_features.shape}")
    
    print("Extracting SEC metadata & sector features...")
    meta_features = extract_metadata_features(df)
    print(f"  -> Metadata shape: {meta_features.shape}")
    
    texts = df['mda_text'].tolist()
    
    print("Extracting expanded 16-phrase Forensic Accounting TF-IDF N-grams...")
    risk_phrases = [
        "material weakness", "internal control", "restatement", "going concern", 
        "non-reliance", "previously issued", "deferred tax asset valuation allowance",
        "reclassification of prior period", "revenue recognition", "impairment of goodwill",
        "related party transaction", "sec investigation", "subpoena", "whistleblower",
        "audit committee", "deficiency in internal control"
    ]
    tfidf = TfidfVectorizer(ngram_range=(1, 3), vocabulary=risk_phrases)
    tfidf_features = tfidf.fit_transform(texts).toarray()
    print(f"  -> TF-IDF shape: {tfidf_features.shape}")
    joblib.dump(tfidf, models_dir / "tfidf_vectorizer.pkl")
    
    bert_features = get_finbert_embeddings(texts)
    print(f"  -> Dual-Pooled FinBERT shape: {bert_features.shape}")
    
    # Concatenate all modalities: 4 (LM) + 4 (Complex) + 6 (Meta) + 16 (TF-IDF) + 1536 (FinBERT) = 1566 dims
    X = np.hstack([lm_features, complex_features, meta_features, tfidf_features, bert_features])
    y = df['y'].values
    
    print(f"\n✅ Final Feature Matrix shape: {X.shape}, Target y shape: {y.shape}")
    
    np.save(data_dir / "X.npy", X)
    np.save(data_dir / "y.npy", y)
    df[["adsh", "cik", "period", "filed", "y"]].to_csv(feature_index, index=False)
    print("Saved updated X.npy and y.npy to data/")

if __name__ == "__main__":
    main()
