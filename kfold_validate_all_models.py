import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
N_SPLITS = 5
RANDOM_STATE = 42

# Dataset paths
MEL_FEATURE_DIR = "features/mel_spectrograms"
MEL_LABEL_FILE = "dataset/labels.csv"
ENGINEERED_FEATURE_FILE = "features/engineered_features.npy"
ENGINEERED_LABEL_FILE = "features/engineered_labels.npy"
SCALER_PATH = "engineered_scaler.pkl"

# Model paths
MODELS = {
    'cnn_lstm_regression': {
        'path': '2_cnn_lstm_regression_model.keras',
        'type': 'regression',
        'feature_type': 'mel'
    },
    'cnn_classification': {
        'path': '2_cnn_classification_model.keras',
        'type': 'classification',
        'feature_type': 'mel'
    },
    'engineered': {
        'path': 'engineered_feature_classifier.keras',
        'type': 'classification',
        'feature_type': 'engineered'
    }
}

# ==================== LOAD DATASETS ====================
def load_mel_data():
    """Load mel spectrograms and regression labels"""
    labels_df = pd.read_csv(MEL_LABEL_FILE, names=["filename", "score"])
    
    X = []
    y = []
    filenames = []
    
    for _, row in labels_df.iterrows():
        filename_npy = row["filename"].replace(".wav", ".npy")
        path = os.path.join(MEL_FEATURE_DIR, filename_npy)
        
        if os.path.exists(path):
            mel = np.load(path)
            mel = (mel - np.mean(mel)) / (np.std(mel) + 1e-6)
            mel = mel[..., np.newaxis]
            X.append(mel)
            y.append(row["score"])
            filenames.append(row["filename"])
    
    X = np.array(X)
    y = np.array(y)
    filenames = np.array(filenames)
    
    print(f"  ✅ Mel data: {X.shape[0]} samples, shape: {X.shape[1:]}")
    return X, y, filenames

def load_engineered_data():
    """Load engineered features and classification labels"""
    X = np.load(ENGINEERED_FEATURE_FILE)
    y = np.load(ENGINEERED_LABEL_FILE)
    
    labels_df = pd.read_csv("dataset/labels_classification.csv", names=["filename", "class"])
    filenames = labels_df['filename'].values
    
    # Scale features
    scaler = joblib.load(SCALER_PATH)
    X = scaler.transform(X)
    
    print(f"  ✅ Engineered data: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, filenames

# ==================== LOAD MODELS ====================
def load_model(model_path, model_type):
    """Load saved model"""
    print(f"  Loading {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"  ❌ Model not found: {model_path}")
        return None
    
    try:
        if model_type == 'regression':
            custom_objects = {
                'mse': tf.keras.losses.MeanSquaredError(),
                'mae': tf.keras.metrics.MeanAbsoluteError()
            }
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        else:
            model = tf.keras.models.load_model(model_path)
        
        print(f"  ✅ Model loaded successfully")
        return model
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        return None

# ==================== K-FOLD VALIDATION ====================
def run_kfold_for_model(model, X, y, model_type, model_name, n_splits=5):
    """Run K-Fold validation for a single model"""
    
    print(f"\n{'='*60}")
    print(f"📊 {model_name} - K-Fold Validation")
    print(f"{'='*60}")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_test = X[test_idx]
        y_test = y[test_idx]
        
        # Predict
        if model_type == 'regression':
            y_pred = model.predict(X_test, verbose=0).flatten()
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Accuracy within threshold
            correct_09 = np.sum(np.abs(y_test - y_pred) <= 0.9)
            correct_15 = np.sum(np.abs(y_test - y_pred) <= 1.5)
            acc_09 = correct_09 / len(y_test)
            acc_15 = correct_15 / len(y_test)
            
            fold_results.append({
                'fold': fold,
                'samples': len(y_test),
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'acc_09': acc_09,
                'acc_15': acc_15
            })
            
            print(f"  Fold {fold}: MAE={mae:.3f}, R²={r2:.3f}, Acc(±0.9)={acc_09:.1%}")
            
        else:  # classification
            y_pred_probs = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            
            fold_results.append({
                'fold': fold,
                'samples': len(y_test),
                'accuracy': acc
            })
            
            print(f"  Fold {fold}: Accuracy={acc:.1%} ({np.sum(y_pred == y_test)}/{len(y_test)})")
    
    return fold_results

# ==================== PRINT SUMMARY ====================
def print_summary(model_name, fold_results, model_type):
    """Print summary statistics"""
    
    print(f"\n{'─'*50}")
    print(f"📈 {model_name} - Summary")
    print(f"{'─'*50}")
    
    if model_type == 'regression':
        mae_list = [r['mae'] for r in fold_results]
        r2_list = [r['r2'] for r in fold_results]
        acc09_list = [r['acc_09'] for r in fold_results]
        
        print(f"  MAE:     {np.mean(mae_list):.3f} ± {np.std(mae_list):.3f}")
        print(f"  R²:      {np.mean(r2_list):.3f} ± {np.std(r2_list):.3f}")
        print(f"  Acc(±0.9): {np.mean(acc09_list):.1%} ± {np.std(acc09_list):.1%}")
        
        # Verdict
        if np.mean(r2_list) < 0.1:
            print(f"  ⚠️  Verdict: Poor generalization (R² near 0)")
        elif np.mean(r2_list) < 0.3:
            print(f"  📊 Verdict: Weak generalization")
        else:
            print(f"  ✅ Verdict: Reasonable generalization")
            
    else:  # classification
        acc_list = [r['accuracy'] for r in fold_results]
        
        print(f"  Accuracy: {np.mean(acc_list):.1%} ± {np.std(acc_list):.1%}")
        
        # Verdict
        if np.mean(acc_list) < 0.4:
            print(f"  ⚠️  Verdict: Poor performance")
        elif np.mean(acc_list) < 0.6:
            print(f"  📊 Verdict: Moderate performance")
        else:
            print(f"  ✅ Verdict: Good performance")
    
    # Per-fold table
    print(f"\n  Per-fold Results:")
    if model_type == 'regression':
        print(f"  {'Fold':<6} {'MAE':<10} {'R²':<10} {'Acc(±0.9)':<12}")
        print(f"  {'-'*40}")
        for r in fold_results:
            print(f"  {r['fold']:<6} {r['mae']:<10.3f} {r['r2']:<10.3f} {r['acc_09']:<12.1%}")
    else:
        print(f"  {'Fold':<6} {'Accuracy':<12} {'Correct/Total':<15}")
        print(f"  {'-'*40}")
        for r in fold_results:
            correct = int(r['accuracy'] * r['samples'])
            print(f"  {r['fold']:<6} {r['accuracy']:<12.1%} {correct}/{r['samples']}")

# ==================== MAIN EXECUTION ====================
def main():
    print("\n" + "="*60)
    print("🎤 K-FOLD VALIDATION FOR ALL MODELS")
    print("="*60)
    
    # Load datasets
    print("\n📂 Loading datasets...")
    mel_X, mel_y, mel_filenames = load_mel_data()
    eng_X, eng_y, eng_filenames = load_engineered_data()
    
    # Store all results
    all_results = {}
    
    # Run validation for each model
    for model_name, model_info in MODELS.items():
        print(f"\n{'='*60}")
        print(f"🔄 Processing: {model_name}")
        print(f"{'='*60}")
        
        # Load model
        model = load_model(model_info['path'], model_info['type'])
        if model is None:
            print(f"  ⚠️ Skipping {model_name} - model not loaded")
            continue
        
        # Select dataset
        if model_info['feature_type'] == 'mel':
            X, y = mel_X, mel_y
        else:
            X, y = eng_X, eng_y
        
        # Run K-Fold
        fold_results = run_kfold_for_model(model, X, y, model_info['type'], model_name, N_SPLITS)
        
        # Print summary
        print_summary(model_name, fold_results, model_info['type'])
        
        # Store results
        all_results[model_name] = {
            'type': model_info['type'],
            'folds': fold_results
        }
    
    # Final comparison table
    print("\n" + "="*60)
    print("🏆 FINAL COMPARISON ACROSS MODELS")
    print("="*60)
    
    print(f"\n{'Model':<30} {'Metric':<15} {'Mean ± Std':<20}")
    print("-" * 65)
    
    for model_name, results in all_results.items():
        if results['type'] == 'regression':
            mae_list = [r['mae'] for r in results['folds']]
            r2_list = [r['r2'] for r in results['folds']]
            print(f"{model_name:<30} {'MAE':<15} {np.mean(mae_list):.3f} ± {np.std(mae_list):.3f}")
            print(f"{model_name:<30} {'R²':<15} {np.mean(r2_list):.3f} ± {np.std(r2_list):.3f}")
        else:
            acc_list = [r['accuracy'] for r in results['folds']]
            print(f"{model_name:<30} {'Accuracy':<15} {np.mean(acc_list):.1%} ± {np.std(acc_list):.1%}")
    
    print("\n✅ K-Fold validation completed for all models!")

if __name__ == "__main__":
    main()