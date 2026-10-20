import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, brier_score_loss
from xgboost import XGBClassifier
import joblib
import json
import os

def evaluate_thresholds(y_true, y_proba):
    thresholds = np.arange(0.20, 0.55, 0.05)
    best_f1 = -1
    best_thresh = 0.5
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if precision >= 0.70 and f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    # Fallback to standard 0.5 if no threshold meets precision >= 0.70
    if best_f1 == -1:
        best_thresh = 0.5
        y_pred = (y_proba >= best_thresh).astype(int)
        best_f1 = f1_score(y_true, y_pred, zero_division=0)
        
    return best_thresh, best_f1

def train_and_evaluate(name, model, X, y, cv):
    oof_preds = np.zeros(len(X))
    oof_proba = np.zeros(len(X))
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        
        if hasattr(model, "predict_proba"):
            oof_proba[val_idx] = model.predict_proba(X_val)[:, 1]
        else:
            # For models like SGDClassifier without probability by default
            # SGDClassifier with loss='log_loss' does have predict_proba
            oof_proba[val_idx] = model.predict_proba(X_val)[:, 1]
            
        oof_preds[val_idx] = model.predict(X_val)
        
    try:
        roc_auc = roc_auc_score(y, oof_proba)
    except ValueError:
        roc_auc = 0.5
        
    best_thresh, _ = evaluate_thresholds(y, oof_proba)
    final_preds = (oof_proba >= best_thresh).astype(int)
    
    precision = precision_score(y, final_preds, zero_division=0)
    recall = recall_score(y, final_preds, zero_division=0)
    f1 = f1_score(y, final_preds, zero_division=0)
    
    # Probability calibration score
    try:
        brier = brier_score_loss(y, oof_proba)
    except ValueError:
        brier = 1.0
    
    return {
        "Model": name,
        "ROC-AUC": roc_auc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Brier Score": brier,
        "Best Thresh": best_thresh
    }, model

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
import torch

def main():
    print("Running 04_train.py (Super-Learner Stacking & Deep Benchmark)...")
    data_dir = Path("data")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    X_path = data_dir / "X.npy"
    y_path = data_dir / "y.npy"
    
    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError("X.npy or y.npy not found in data/. Run 03_feature_engine.py first.")
        
    X = np.load(X_path)
    y = np.load(y_path)

    feature_index = data_dir / "feature_index.csv"
    if not feature_index.exists():
        raise ValueError("Feature accession index is missing; regenerate features before training. Existing X.npy is stale.")
    rows = pd.read_csv(feature_index, dtype={"adsh": str, "cik": str, "period": str, "filed": str})
    if len(rows) != len(X) or len(y) != len(X) or X.shape[1] != 1566:
        raise ValueError("Feature rows or width do not match the current feature pipeline; regenerate features.")
    if not np.array_equal(rows["y"].to_numpy(), y):
        raise ValueError("Feature labels are misaligned with the accession index.")
    raise RuntimeError(
        "Training is paused: current labels identify amendments, not confirmed restatements, "
        "and the corpus is a balanced sample. Build verified restatement labels and a "
        "natural-prevalence temporal test set before publishing a risk model."
    )
    
    print(f"Loaded X with shape {X.shape} and y with shape {y.shape}")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    xgb_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tree_method = 'hist'
    
    base_lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight='balanced', random_state=42))
    base_mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, early_stopping=True, random_state=42))
    base_et = ExtraTreesClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
    base_xgb = XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.03, subsample=0.8, 
        colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=1.5, 
        scale_pos_weight=scale_pos_weight, eval_metric='logloss', random_state=42, 
        tree_method=tree_method, device=xgb_device
    )
    
    stacking_ensemble = StackingClassifier(
        estimators=[
            ('lr', base_lr),
            ('mlp', base_mlp),
            ('xgb', base_xgb)
        ],
        final_estimator=LogisticRegression(C=1.0, random_state=42),
        cv=3,
        n_jobs=-1
    )
    
    models_to_evaluate = [
        ("Logistic Regression (L2)", base_lr),
        ("Deep MLP Neural Network", base_mlp),
        ("Extra Trees Forest", base_et),
        ("GPU Tuned XGBoost", base_xgb),
        ("Stacking Super-Learner", stacking_ensemble)
    ]
    
    results = []
    trained_models = {}
    
    for name, model in models_to_evaluate:
        print(f"Evaluating {name} across 5 stratified folds...")
        res, fitted_model = train_and_evaluate(name, model, X, y, cv)
        results.append(res)
        trained_models[name] = fitted_model
        
    # Print Comparison Table
    df_results = pd.DataFrame(results).round(4)
    print("\n" + "="*80)
    print("MULTI-MODEL BENCHMARK RESULTS")
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80)
    
    # Generate Performance Chart
    import matplotlib.pyplot as plt
    
    x_labels = df_results['Model']
    roc_scores = df_results['ROC-AUC']
    f1_scores = df_results['F1-Score']
    
    x = np.arange(len(x_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, roc_scores, width, label='ROC-AUC', color='#1f77b4')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', color='#ff7f0e')
    
    ax.set_ylabel('Scores')
    ax.set_title('Financial Risk Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=15)
    ax.legend(loc='lower right')
    
    ax.bar_label(rects1, padding=3, fmt='%.3f')
    ax.bar_label(rects2, padding=3, fmt='%.3f')
    
    fig.tight_layout()
    chart_path = models_dir / "model_comparison.png"
    plt.savefig(chart_path, dpi=300)
    print(f"\nGenerated performance chart: {chart_path}")
    
    # Select best model (e.g., by highest F1-Score, or ROC-AUC)
    # Since we want to balance both, let's sort by F1-Score primarily, then ROC-AUC
    best_row = df_results.sort_values(by=["F1-Score", "ROC-AUC"], ascending=[False, False]).iloc[0]
    best_name = best_row["Model"]
    
    print(f"\nBest performing model selected: {best_name} (F1: {best_row['F1-Score']})")
    
    best_model = trained_models[best_name]
    
    print("Retraining best model on full dataset for persistence...")
    best_model.fit(X, y)
    
    # Save the best model
    if best_name == "XGBoost":
        out_path = models_dir / "best_risk_model.json"
        best_model.save_model(str(out_path))
    else:
        out_path = models_dir / "best_risk_model.pkl"
        joblib.dump(best_model, out_path)
        
    # Also save the threshold so the inference script can use it
    meta_path = models_dir / "best_model_meta.json"
    with open(meta_path, 'w') as f:
        json.dump({
            "model_type": best_name,
            "threshold": float(best_row["Best Thresh"])
        }, f)
        
    print(f"Saved best model artifact to {out_path}")

if __name__ == "__main__":
    main()
