"""
SINGING SKILL PREDICTOR
Place this file in your ASSE root folder
Run with: python predict_score.py path/to/audio.wav
"""

import numpy as np
import librosa
import joblib
import tensorflow as tf
import os
import sys
from sklearn.preprocessing import StandardScaler

class SingingSkillPredictor:
    def __init__(self, model_path="engineered_feature_classifier.keras"):
        """
        Initialize predictor with your trained model
        """
        print(f"🎤 Loading model from {model_path}...")
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            print("Please make sure engineered_feature_classifier.keras is in this folder")
            sys.exit(1)
        
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = None
        self.scaler_path = "features/feature_scaler.pkl"
        
        # Load or create scaler
        self._setup_scaler()
    
    def _setup_scaler(self):
        """Setup scaler from training features"""
        features_path = "features/engineered_features.npy"
        
        # Try to load existing scaler
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            print(f"✅ Loaded scaler from {self.scaler_path}")
            return
        
        # Create new scaler from training features
        print("🔄 Creating scaler from training features...")
        if os.path.exists(features_path):
            X_train = np.load(features_path)
            self.scaler = StandardScaler()
            self.scaler.fit(X_train)
            joblib.dump(self.scaler, self.scaler_path)
            print(f"✅ Scaler created and saved to {self.scaler_path}")
        else:
            print(f"⚠️  Training features not found at {features_path}")
            print("Will use unnormalized features (may be less accurate)")
    
    def extract_features(self, audio_path):
        """
        Extract the SAME 17 features used in training
        """
        try:
            # Load audio (using your preprocessing steps)
            print(f"📊 Loading audio: {audio_path}")
            y, sr = librosa.load(audio_path, sr=22050)
            
            # 1️⃣ MFCC (13 features) - same as your extract_engineered_features.py
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)  # 13 values
            
            # 2️⃣ Pitch variance (1 feature)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = pitches[magnitudes > np.median(magnitudes)]
            pitch_variance = np.var(pitch_values) if len(pitch_values) > 0 else 0
            
            # 3️⃣ Spectral Centroid (1 feature)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_centroid_mean = np.mean(spectral_centroid)
            
            # 4️⃣ Zero Crossing Rate (1 feature)
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_mean = np.mean(zcr)
            
            # 5️⃣ Harmonic Ratio (1 feature)
            harmonic, percussive = librosa.effects.hpss(y)
            harmonic_energy = np.sum(np.abs(harmonic))
            percussive_energy = np.sum(np.abs(percussive)) + 1e-6
            harmonic_ratio = harmonic_energy / percussive_energy
            
            # Combine exactly as in your training code
            features = np.hstack([
                mfcc_mean,              # 13 features
                pitch_variance,          # 1 feature
                spectral_centroid_mean,  # 1 feature
                zcr_mean,                # 1 feature
                harmonic_ratio           # 1 feature
            ])  # Total: 17 features
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting features: {str(e)}")
            return None
    
    def predict(self, audio_path):
        """
        Predict singing skill for an audio file
        """
        # Check if file exists
        if not os.path.exists(audio_path):
            print(f"❌ Audio file not found: {audio_path}")
            return None
        
        print(f"\n{'='*50}")
        print(f"🔍 Analyzing: {os.path.basename(audio_path)}")
        print(f"{'='*50}")
        
        # Extract features
        features = self.extract_features(audio_path)
        if features is None:
            return None
        
        # Reshape for model
        features = features.reshape(1, -1)
        
        # Normalize if scaler exists
        if self.scaler:
            features = self.scaler.transform(features)
        
        # Predict
        predictions = self.model.predict(features, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        
        # Class mapping (same as your training)
        class_names = {
            0: "🎵 BAD (0-3)",
            1: "🎵 INTERMEDIATE (4-7)",
            2: "🎵 GOOD (8-10)"
        }
        
        # Calculate approximate score
        score_ranges = {0: (0,3), 1: (4,7), 2: (8,10)}
        min_score, max_score = score_ranges[predicted_class]
        
        # Adjust confidence to score
        adj_confidence = max(0, min(1, (confidence - 0.33) / 0.67))
        approx_score = min_score + (adj_confidence * (max_score - min_score))
        
        # Results
        results = {
            'class': predicted_class,
            'level': class_names[predicted_class],
            'confidence': float(confidence),
            'score': round(approx_score, 1),
            'probabilities': {
                'bad': float(predictions[0]),
                'intermediate': float(predictions[1]),
                'good': float(predictions[2])
            }
        }
        
        # Display results
        self._display_results(results, audio_path)
        
        return results
    
    def _display_results(self, results, audio_path):
        """Pretty print results"""
        print(f"\n📊 PREDICTION RESULTS")
        print(f"{'='*50}")
        print(f"🎯 Skill Level: {results['level']}")
        print(f"📝 Score: {results['score']}/10")
        print(f"📈 Confidence: {results['confidence']:.1%}")
        print(f"\n📊 Probability Breakdown:")
        print(f"   • Bad (0-3):       {results['probabilities']['bad']:.1%}")
        print(f"   • Intermediate (4-7): {results['probabilities']['intermediate']:.1%}")
        print(f"   • Good (8-10):     {results['probabilities']['good']:.1%}")
        print(f"{'='*50}")


def quick_predict():
    """Simple function to predict from command line"""
    if len(sys.argv) < 2:
        print("\n🎤 SINGING SKILL PREDICTOR")
        print("="*50)
        print("Usage: python predict_score.py <audio_file>")
        print("\nExamples:")
        print("  python predict_score.py test_singing.wav")
        print("  python predict_score.py preprocessing/3_rms_normalized/song.wav")
        print("  python predict_score.py ../Downloads/my_recording.wav")
        return
    
    audio_file = sys.argv[1]
    
    # Initialize predictor
    predictor = SingingSkillPredictor()
    
    # Make prediction
    predictor.predict(audio_file)


def batch_mode():
    """Predict all audio files in a folder"""
    if len(sys.argv) < 3 or sys.argv[1] != "--batch":
        return False
    
    folder_path = sys.argv[2]
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return True
    
    # Initialize predictor
    predictor = SingingSkillPredictor()
    
    # Find all audio files
    audio_files = []
    for file in os.listdir(folder_path):
        if file.endswith(('.wav', '.mp3', '.m4a', '.flac')):
            audio_files.append(file)
    
    if not audio_files:
        print(f"❌ No audio files found in {folder_path}")
        return True
    
    print(f"\n📁 Found {len(audio_files)} audio files")
    print("="*50)
    
    results = []
    for audio_file in audio_files:
        file_path = os.path.join(folder_path, audio_file)
        result = predictor.predict(file_path)
        if result:
            results.append({
                'file': audio_file,
                'level': result['level'],
                'score': result['score'],
                'confidence': result['confidence']
            })
    
    # Summary
    if results:
        print("\n📊 BATCH SUMMARY")
        print("="*50)
        levels = {}
        for r in results:
            level_name = r['level'].split('(')[0].strip()
            levels[level_name] = levels.get(level_name, 0) + 1
        
        for level, count in levels.items():
            print(f"{level}: {count} files")
        
        avg_score = sum(r['score'] for r in results) / len(results)
        print(f"\nAverage Score: {avg_score:.1f}/10")
    
    return True


if __name__ == "__main__":
    # Check for batch mode
    if not batch_mode():
        # Single file mode
        quick_predict()