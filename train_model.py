import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from sklearn.model_selection import cross_val_score
from preprocessing import load_dataset, clean_and_encode, get_features_and_labels, split_data


def train_model(X_train, y_train):
    """Train Random Forest with config from synopsis."""
    print("\n[1/4] Training Random Forest...")
    print("      n_estimators=100 | max_depth=10 | criterion=gini")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        criterion='gini',
        random_state=42,
        n_jobs=-1       # use all CPU cores — faster training
    )
    model.fit(X_train, y_train)
    print("      ✅ Training complete")
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Compute all 6 metrics required by Objective 3."""
    print("\n[2/4] Evaluating model...")

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Core metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_proba)

    # 5-Fold Cross Validation on training set
    print("      Running 5-Fold Cross-Validation (may take ~1 min)...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    cv_mean   = cv_scores.mean()
    cv_std    = cv_scores.std()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        'accuracy':  round(accuracy  * 100, 2),
        'precision': round(precision * 100, 2),
        'recall':    round(recall    * 100, 2),
        'f1':        round(f1        * 100, 2),
        'roc_auc':   round(roc_auc   * 100, 2),
        'cv_mean':   round(cv_mean   * 100, 2),
        'cv_std':    round(cv_std    * 100, 2),
        'cv_scores': [round(s * 100, 2) for s in cv_scores],
        'confusion_matrix': cm.tolist(),
        'y_pred':    y_pred.tolist(),
        'y_proba':   y_proba.tolist(),
    }

    # Print results
    print("\n" + "="*45)
    print("  MODEL PERFORMANCE — OBJECTIVE 3")
    print("="*45)
    print(f"  Accuracy         : {metrics['accuracy']}%")
    print(f"  Precision        : {metrics['precision']}%")
    print(f"  Recall           : {metrics['recall']}%")
    print(f"  F1-Score         : {metrics['f1']}%")
    print(f"  ROC-AUC          : {metrics['roc_auc']}%")
    print(f"  CV (5-Fold Mean) : {metrics['cv_mean']}% ± {metrics['cv_std']}%")
    print("="*45)

    print("\n  Confusion Matrix:")
    print(f"  {'':20} Predicted Normal  Predicted Attack")
    print(f"  {'Actual Normal':20} {cm[0][0]:>16,}  {cm[0][1]:>15,}")
    print(f"  {'Actual Attack':20} {cm[1][0]:>16,}  {cm[1][1]:>15,}")
    print("="*45)

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

    return metrics


def save_artifacts(model, metrics, feature_cols):
    """Save model and metrics for use in Streamlit app."""
    print("\n[3/4] Saving model artifacts...")

    joblib.dump(model, 'model.pkl')
    joblib.dump(metrics, 'metrics.pkl')
    joblib.dump(feature_cols, 'feature_cols.pkl')

    print("      model.pkl        — trained Random Forest")
    print("      metrics.pkl      — all evaluation metrics")
    print("      feature_cols.pkl — feature column list")


def assign_risk_level(probability):
    """
    Map attack probability to risk level (Low / Medium / High).
    Used per-row in predictions.
    """
    if probability < 0.4:
        return 'Low'
    elif probability < 0.75:
        return 'Medium'
    else:
        return 'High'


if __name__ == "__main__":
    train_path = sys.argv[1] if len(sys.argv) > 1 else "KDDTrain+.txt"
    test_path  = sys.argv[2] if len(sys.argv) > 2 else "KDDTest+.txt"

    # Step 1: Preprocess
    df = load_dataset(train_path, test_path)
    df = clean_and_encode(df)
    X, y, feature_cols = get_features_and_labels(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 2: Train
    model = train_model(X_train, y_train)

    # Step 3: Evaluate
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

    # Step 4: Save
    save_artifacts(model, metrics, feature_cols)

    print("\n✅ Objective 1 complete — model trained, saved, and evaluated!")
    print("   Run: streamlit run app.py")
