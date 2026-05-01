import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# NSL-KDD column names (41 features + label + difficulty)
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']


def load_dataset(train_path, test_path=None):
    """Load NSL-KDD dataset from CSV files."""
    print("[1/5] Loading dataset...")

    train_df = pd.read_csv(train_path, header=None, names=COLUMNS)
    print(f"      Train set: {train_df.shape[0]:,} records, {train_df.shape[1]} columns")

    if test_path and os.path.exists(test_path):
        test_df = pd.read_csv(test_path, header=None, names=COLUMNS)
        print(f"      Test set:  {test_df.shape[0]:,} records")
        df = pd.concat([train_df, test_df], ignore_index=True)
    else:
        df = train_df

    # Drop difficulty column (not a feature)
    if 'difficulty' in df.columns:
        df = df.drop(columns=['difficulty'])

    return df


def clean_and_encode(df):
    """Clean data, encode categoricals, create binary label."""
    print("[2/5] Cleaning and encoding...")

    # Handle missing values
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean())
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'label':
            df[col] = df[col].fillna(df[col].mode()[0])

    # Binary label: normal=0, attack=1
    df['binary_label'] = df['label'].apply(lambda x: 0 if x.strip().lower() == 'normal' else 1)

    # Keep original attack type for reference
    df['attack_type'] = df['label'].apply(lambda x: x.strip().lower())

    # Label encode categorical columns
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Save encoders for use in app
    joblib.dump(encoders, 'encoders.pkl')
    print(f"      Encoders saved → encoders.pkl")

    return df


def get_features_and_labels(df):
    """Extract feature matrix X and label vector y."""
    print("[3/5] Extracting features...")

    feature_cols = [c for c in df.columns if c not in ['label', 'binary_label', 'attack_type']]
    X = df[feature_cols]
    y = df['binary_label']

    print(f"      Features: {len(feature_cols)}")
    print(f"      Normal:   {(y == 0).sum():,} records")
    print(f"      Attack:   {(y == 1).sum():,} records")

    return X, y, feature_cols


def split_data(X, y, test_size=0.2, random_state=42):
    """80:20 train-test split."""
    print("[4/5] Splitting dataset (80:20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"      Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test


def preprocess_uploaded_file(filepath):
    """
    Preprocess a user-uploaded CSV log file for prediction.
    Handles both NSL-KDD format and partial uploads.
    """
    df = pd.read_csv(filepath, header=None)

    # Assign column names based on number of columns
    if df.shape[1] >= 42:
        df.columns = COLUMNS[:df.shape[1]]
        if 'difficulty' in df.columns:
            df = df.drop(columns=['difficulty'])
    elif df.shape[1] == 41:
        df.columns = COLUMNS[:41]
    else:
        # Try to assign as many columns as possible
        df.columns = COLUMNS[:df.shape[1]]

    # Store original label if present
    original_labels = None
    if 'label' in df.columns:
        original_labels = df['label'].copy()

    # Clean + encode
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean())

    # Load saved encoders
    if os.path.exists('encoders.pkl'):
        encoders = joblib.load('encoders.pkl')
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                le = encoders[col]
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )

    feature_cols = [c for c in df.columns if c not in ['label', 'binary_label', 'attack_type']]
    X = df[feature_cols]

    return X, original_labels, df


if __name__ == "__main__":
    import sys

    train_path = sys.argv[1] if len(sys.argv) > 1 else "KDDTrain+.txt"
    test_path  = sys.argv[2] if len(sys.argv) > 2 else "KDDTest+.txt"

    df = load_dataset(train_path, test_path)
    df = clean_and_encode(df)
    X, y, feature_cols = get_features_and_labels(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Save processed data
    print("[5/5] Saving processed data...")
    X_train.to_csv("X_train.csv", index=False)
    X_test.to_csv("X_test.csv",  index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_test.to_csv("y_test.csv",  index=False)
    joblib.dump(feature_cols, "feature_cols.pkl")
    print("      Saved: X_train.csv, X_test.csv, y_train.csv, y_test.csv, feature_cols.pkl")
    print("\n✅ Preprocessing complete!")
