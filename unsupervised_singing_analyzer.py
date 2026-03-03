"""
Unsupervised Singing Analysis System with Labeled Dataset
For automatic singing skill evaluation - pattern diagnosis in vocal recordings
Dataset: 92 singing audios with scores (labels)
Author: Based on guidance from Joe Cheri Ross
"""

import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from scipy.stats import entropy, pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import essentia.standard as es
import os
import warnings
warnings.filterwarnings('ignore')

class UnsupervisedSingingAnalyzer:
    """
    Main class for unsupervised singing analysis
    Analyzes vocal recordings without reference songs to identify patterns
    in good vs. poor singing
    """
    
    def __init__(self, dataset_path, labels_path, sample_rate=22050):
        self.sample_rate = sample_rate
        self.dataset_path = dataset_path
        self.labels_path = labels_path
        self.results = {}
        
        # Load labels
        self.load_labels()
        
    def load_labels(self):
        """
        Load the classification labels from CSV
        """
        try:
            self.labels_df = pd.read_csv(self.labels_path)
            print(f"Loaded labels for {len(self.labels_df)} songs")
            print(f"Columns in labels file: {list(self.labels_df.columns)}")
            
            # Display label statistics
            if 'score' in self.labels_df.columns:
                print(f"\nScore statistics:")
                print(f"  Min: {self.labels_df['score'].min()}")
                print(f"  Max: {self.labels_df['score'].max()}")
                print(f"  Mean: {self.labels_df['score'].mean():.2f}")
                print(f"  Std: {self.labels_df['score'].std():.2f}")
            else:
                print("Warning: 'score' column not found in labels file")
                print(f"Available columns: {list(self.labels_df.columns)}")
                
        except Exception as e:
            print(f"Error loading labels: {e}")
            self.labels_df = None
    
    def get_audio_files(self):
        """
        Get list of audio files from dataset path
        """
        audio_files = []
        for file in os.listdir(self.dataset_path):
            if file.endswith(('.wav', '.mp3', '.m4a', '.flac')):
                audio_files.append(os.path.join(self.dataset_path, file))
        
        print(f"Found {len(audio_files)} audio files in dataset")
        return audio_files
    
    def extract_pitch_fftnet(self, audio_file):
        """
        Extract pitch values using FFTNet approach
        Note: Replace with your actual FFTNet implementation
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # Placeholder for FFTNet pitch extraction
            # Replace this with your actual FFTNet code
            # For now, using librosa's pitch detection as placeholder
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, 
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            
            # Replace NaN with 0 for unvoiced segments
            f0 = np.nan_to_num(f0, nan=0.0)
            
            return f0, voiced_probs
            
        except Exception as e:
            print(f"Error extracting pitch from {audio_file}: {e}")
            return None, None
    
    def identify_tonic(self, audio_file):
        """
        Identify tonic using essentia library
        """
        try:
            # Load audio for essentia
            loader = es.MonoLoader(filename=audio_file, 
                                  sampleRate=self.sample_rate)
            audio = loader()
            
            # Use Indian Art Music tonic extractor
            tonic_extractor = es.TonicIndianArtMusic()
            tonic = tonic_extractor(audio)
            
            return tonic
            
        except Exception as e:
            print(f"Error identifying tonic: {e}")
            # Fallback: use mean of pitch as rough tonic
            pitch, _ = self.extract_pitch_fftnet(audio_file)
            if pitch is not None:
                valid_pitch = pitch[pitch > 0]
                if len(valid_pitch) > 0:
                    return np.median(valid_pitch)
            return 220.0  # Default to A3 if all fails
    
    def convert_to_cents(self, pitch_values, tonic):
        """
        Convert Hz pitch values to cents relative to tonic
        cents = 1200 * log2(pitch/tonic)
        """
        # Avoid division by zero and log of zero
        valid_mask = pitch_values > 0
        cents_values = np.full_like(pitch_values, np.nan)
        
        if np.any(valid_mask):
            cents_values[valid_mask] = 1200 * np.log2(pitch_values[valid_mask] / tonic)
        
        # Remove invalid values
        cents_values = cents_values[~np.isnan(cents_values)]
        cents_values = cents_values[np.isfinite(cents_values)]
        
        return cents_values
    
    def create_pitch_histogram(self, cents_values, bin_width=10):
        """
        Create pitch histogram with specified bin width (in cents)
        """
        # Define histogram range (typically 2 octaves above and below tonic)
        hist_range = (-1200, 1200)
        bins = np.arange(hist_range[0], hist_range[1] + bin_width, bin_width)
        
        # Create histogram
        hist, bin_edges = np.histogram(cents_values, bins=bins)
        
        # Normalize to probability distribution
        if np.sum(hist) > 0:
            hist_normalized = hist / np.sum(hist)
        else:
            hist_normalized = hist
        
        return hist_normalized, bin_edges
    
    def extract_histogram_features(self, hist_normalized, bin_edges):
        """
        Extract meaningful features from pitch histogram
        These features indicate singing quality
        """
        features = {}
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 1. Peak prominence (highest peak value)
        if len(hist_normalized) > 0 and np.max(hist_normalized) > 0:
            main_peak_idx = np.argmax(hist_normalized)
            features['peak_prominence'] = hist_normalized[main_peak_idx]
            features['peak_position'] = bin_centers[main_peak_idx]
        else:
            features['peak_prominence'] = 0
            features['peak_position'] = 0
            main_peak_idx = 0
        
        # 2. Number of significant peaks (using relative threshold)
        if np.max(hist_normalized) > 0:
            peak_threshold = 0.05 * np.max(hist_normalized)
            peaks, properties = find_peaks(hist_normalized, height=peak_threshold)
            features['num_peaks'] = len(peaks)
        else:
            features['num_peaks'] = 0
        
        # 3. Peak sharpness (energy around main peak)
        if len(hist_normalized) > 0:
            peak_window = 5  # 5 bins on each side
            start_idx = max(0, main_peak_idx - peak_window)
            end_idx = min(len(hist_normalized), main_peak_idx + peak_window + 1)
            features['peak_sharpness'] = np.sum(hist_normalized[start_idx:end_idx])
        else:
            features['peak_sharpness'] = 0
        
        # 4. Entropy (measure of dispersion)
        # Add small epsilon to avoid log(0)
        if np.sum(hist_normalized) > 0:
            features['entropy'] = -np.sum(hist_normalized * np.log(hist_normalized + 1e-10))
        else:
            features['entropy'] = 0
        
        # 5. Tonic energy (energy around 0 cents)
        if len(bin_centers) > 0:
            tonic_bin = np.argmin(np.abs(bin_centers))
            tonic_window = 5
            tonic_start = max(0, tonic_bin - tonic_window)
            tonic_end = min(len(hist_normalized), tonic_bin + tonic_window + 1)
            features['tonic_energy'] = np.sum(hist_normalized[tonic_start:tonic_end])
        else:
            features['tonic_energy'] = 0
        
        # 6. Pitch range (spread of distribution)
        if len(hist_normalized) > 0 and np.sum(hist_normalized) > 0:
            cumulative = np.cumsum(hist_normalized)
            percentile_5 = bin_centers[np.searchsorted(cumulative, 0.05)]
            percentile_95 = bin_centers[np.searchsorted(cumulative, 0.95)]
            features['pitch_range'] = percentile_95 - percentile_5
        else:
            features['pitch_range'] = 0
        
        # 7. Skewness (asymmetry of distribution)
        if len(hist_normalized) > 0 and np.sum(hist_normalized) > 0:
            mean_pos = np.sum(bin_centers * hist_normalized)
            variance = np.sum(hist_normalized * (bin_centers - mean_pos) ** 2)
            if variance > 0:
                features['skewness'] = np.sum(hist_normalized * ((bin_centers - mean_pos) ** 3)) / (variance ** 1.5)
            else:
                features['skewness'] = 0
        else:
            features['skewness'] = 0
        
        # 8. Kurtosis (peakedness of distribution)
        if len(hist_normalized) > 0 and variance > 0:
            features['kurtosis'] = np.sum(hist_normalized * ((bin_centers - mean_pos) ** 4)) / (variance ** 2) - 3
        else:
            features['kurtosis'] = 0
        
        return features
    
    def extract_additional_features(self, pitch_values, cents_values):
        """
        Extract additional time-domain features
        """
        features = {}
        
        # Pitch stability features
        valid_pitch = pitch_values[pitch_values > 0]
        if len(valid_pitch) > 0:
            features['pitch_mean_hz'] = np.mean(valid_pitch)
            features['pitch_std_hz'] = np.std(valid_pitch)
            features['pitch_cv'] = features['pitch_std_hz'] / features['pitch_mean_hz']  # Coefficient of variation
        else:
            features['pitch_mean_hz'] = 0
            features['pitch_std_hz'] = 0
            features['pitch_cv'] = 0
        
        # Voicing features
        features['voiced_ratio'] = np.sum(pitch_values > 0) / len(pitch_values)
        
        # Pitch contour features (if we have enough valid pitch)
        if len(valid_pitch) > 5:
            # Rate of pitch change
            pitch_diff = np.diff(valid_pitch)
            features['mean_pitch_change'] = np.mean(np.abs(pitch_diff))
            features['max_pitch_change'] = np.max(np.abs(pitch_diff))
            
            # Smoothness of pitch contour
            features['pitch_contour_smoothness'] = 1.0 / (1.0 + np.std(pitch_diff))
        else:
            features['mean_pitch_change'] = 0
            features['max_pitch_change'] = 0
            features['pitch_contour_smoothness'] = 0
        
        # Cents-based features
        if len(cents_values) > 0:
            features['cents_std'] = np.std(cents_values)
            features['cents_range'] = np.max(cents_values) - np.min(cents_values)
        else:
            features['cents_std'] = 0
            features['cents_range'] = 0
        
        return features
    
    def analyze_single_song(self, audio_file):
        """
        Complete analysis pipeline for a single song
        """
        print(f"\nAnalyzing: {os.path.basename(audio_file)}")
        
        # Step 1: Extract pitch
        pitch_values, confidence = self.extract_pitch_fftnet(audio_file)
        if pitch_values is None:
            return None
        
        # Step 2: Identify tonic
        tonic = self.identify_tonic(audio_file)
        print(f"  Tonic identified: {tonic:.2f} Hz")
        
        # Step 3: Convert to cents
        cents_values = self.convert_to_cents(pitch_values, tonic)
        print(f"  Valid pitch frames: {len(cents_values)}")
        
        # Step 4: Create histogram
        hist_normalized, bin_edges = self.create_pitch_histogram(cents_values)
        
        # Step 5: Extract histogram features
        hist_features = self.extract_histogram_features(hist_normalized, bin_edges)
        
        # Step 6: Extract additional features
        time_features = self.extract_additional_features(pitch_values, cents_values)
        
        # Combine all features
        features = {**hist_features, **time_features}
        
        # Add metadata
        features['file_name'] = os.path.basename(audio_file)
        features['full_path'] = audio_file
        features['tonic_hz'] = tonic
        
        return features, hist_normalized, bin_edges
    
    def analyze_dataset(self):
        """
        Analyze all songs in the dataset
        """
        audio_files = self.get_audio_files()
        all_features = []
        all_histograms = []
        
        for audio_file in audio_files:
            result = self.analyze_single_song(audio_file)
            if result is not None:
                features, hist, bins = result
                
                # Add label if available
                if self.labels_df is not None:
                    file_name = os.path.basename(audio_file)
                    label_row = self.labels_df[self.labels_df['file_name'] == file_name]
                    if len(label_row) > 0:
                        features['true_score'] = label_row.iloc[0]['score']
                    else:
                        features['true_score'] = np.nan
                        print(f"  Warning: No label found for {file_name}")
                
                all_features.append(features)
                all_histograms.append({
                    'file': audio_file,
                    'histogram': hist,
                    'bins': bins
                })
        
        # Convert to DataFrame
        self.results['features_df'] = pd.DataFrame(all_features)
        self.results['histograms'] = all_histograms
        
        print(f"\nSuccessfully analyzed {len(all_features)} songs")
        print(f"Features extracted: {list(self.results['features_df'].columns)}")
        
        return self.results['features_df']
    
    def analyze_correlations_with_scores(self):
        """
        Analyze how extracted features correlate with actual scores
        """
        if 'features_df' not in self.results:
            print("No data available. Run analyze_dataset first.")
            return
        
        df = self.results['features_df']
        
        # Check if true_score column exists
        if 'true_score' not in df.columns:
            print("No true_score column found. Labels may not be properly loaded.")
            return
        
        # Remove rows with NaN scores
        df_clean = df.dropna(subset=['true_score'])
        
        if len(df_clean) == 0:
            print("No valid scores found")
            return
        
        print("\n" + "="*70)
        print("CORRELATION ANALYSIS WITH ACTUAL SCORES")
        print("="*70)
        
        # Select numeric features for correlation (exclude non-numeric and metadata)
        exclude_cols = ['file_name', 'full_path', 'true_score']
        feature_cols = [col for col in df_clean.columns 
                       if col not in exclude_cols 
                       and pd.api.types.is_numeric_dtype(df_clean[col])]
        
        # Calculate correlations
        correlations = []
        for col in feature_cols:
            corr, p_value = pearsonr(df_clean[col], df_clean['true_score'])
            correlations.append({
                'feature': col,
                'correlation': corr,
                'p_value': p_value,
                'abs_correlation': abs(corr)
            })
        
        # Convert to DataFrame and sort
        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values('abs_correlation', ascending=False)
        
        # Display top correlations
        print("\nTop features correlated with singing scores:")
        print("-" * 60)
        for idx, row in corr_df.head(10).iterrows():
            significance = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
            print(f"{row['feature']:25s}: {row['correlation']:6.3f} (p={row['p_value']:.4f}{significance})")
        
        self.results['correlations'] = corr_df
        
        # Create correlation plot
        self.plot_correlations(corr_df)
        
        return corr_df
    
    def plot_correlations(self, corr_df):
        """
        Plot correlation heatmap
        """
        plt.figure(figsize=(12, 8))
        
        # Get top 15 features for visualization
        top_features = corr_df.head(15)['feature'].tolist()
        
        # Create correlation matrix for these features
        df = self.results['features_df']
        feature_corr = df[top_features + ['true_score']].corr()
        
        # Plot heatmap
        sns.heatmap(feature_corr, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, fmt='.2f')
        plt.title('Feature Correlations with Singing Scores')
        plt.tight_layout()
        plt.show()
        
        # Bar plot of top correlations
        plt.figure(figsize=(12, 6))
        top_10 = corr_df.head(10)
        colors = ['green' if x > 0 else 'red' for x in top_10['correlation']]
        plt.barh(top_10['feature'], top_10['correlation'], color=colors)
        plt.xlabel('Correlation with Score')
        plt.title('Top 10 Features Correlated with Singing Scores')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def discover_patterns(self, n_clusters=2):
        """
        Use clustering to discover natural patterns in the data
        Compare clusters with actual scores
        """
        if 'features_df' not in self.results:
            print("No data available. Run analyze_dataset first.")
            return None
        
        df = self.results['features_df'].copy()
        
        # Select features for clustering (exclude metadata and true_score)
        exclude_cols = ['file_name', 'full_path', 'true_score', 'tonic_hz']
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols 
                       and pd.api.types.is_numeric_dtype(df[col])]
        
        X = df[feature_cols].values
        
        # Handle any NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Find optimal number of clusters (elbow method)
        inertias = []
        K_range = range(1, min(6, len(df)))
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Perform clustering with specified number
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to dataframe
        df['cluster'] = cluster_labels
        
        # Perform PCA for visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        df['pca1'] = pca_result[:, 0]
        df['pca2'] = pca_result[:, 1]
        
        self.results['clustered_df'] = df
        self.results['cluster_centers'] = kmeans.cluster_centers_
        self.results['scaler'] = scaler
        self.results['pca'] = pca
        self.results['inertias'] = inertias
        self.results['K_range'] = list(K_range)
        
        # Analyze clusters against true scores
        self.analyze_clusters_vs_scores()
        
        return df
    
    def analyze_clusters_vs_scores(self):
        """
        Analyze how clusters correspond to actual scores
        """
        if 'clustered_df' not in self.results:
            return
        
        df = self.results['clustered_df']
        
        if 'true_score' not in df.columns:
            print("No true_score column for cluster analysis")
            return
        
        print("\n" + "="*70)
        print("CLUSTER ANALYSIS VS ACTUAL SCORES")
        print("="*70)
        
        # Calculate score statistics by cluster
        cluster_stats = df.groupby('cluster')['true_score'].agg(['count', 'mean', 'std', 'min', 'max'])
        print("\nScore statistics by cluster:")
        print("-" * 50)
        print(cluster_stats.round(2))
        
        # Perform ANOVA to see if clusters significantly differ in scores
        from scipy.stats import f_oneway
        
        cluster_groups = [df[df['cluster'] == c]['true_score'].dropna() 
                         for c in df['cluster'].unique()]
        
        if len(cluster_groups) > 1:
            f_stat, p_value = f_oneway(*cluster_groups)
            print(f"\nANOVA test for score differences between clusters:")
            print(f"  F-statistic: {f_stat:.3f}")
            print(f"  p-value: {p_value:.4f}")
            if p_value < 0.05:
                print("  ✅ Clusters show significant differences in scores")
            else:
                print("  ❌ No significant score differences between clusters")
        
        # Visualize cluster scores
        plt.figure(figsize=(10, 6))
        
        # Box plot
        plt.subplot(1, 2, 1)
        sns.boxplot(x='cluster', y='true_score', data=df)
        plt.title('Score Distribution by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        
        # Scatter plot with PCA
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(df['pca1'], df['pca2'], 
                             c=df['true_score'], cmap='viridis', 
                             s=100, alpha=0.7)
        plt.colorbar(scatter, label='True Score')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('PCA Colored by Actual Score')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def train_prediction_model(self):
        """
        Train a simple model to predict scores from features
        """
        if 'features_df' not in self.results:
            print("No data available")
            return
        
        df = self.results['features_df'].dropna(subset=['true_score'])
        
        if len(df) < 10:
            print("Not enough data for training")
            return
        
        # Prepare features
        exclude_cols = ['file_name', 'full_path', 'true_score', 'tonic_hz']
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols 
                       and pd.api.types.is_numeric_dtype(df[col])]
        
        X = df[feature_cols].fillna(0)
        y = df['true_score']
        
        # Train Random Forest
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Cross-validation
        cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
        
        print("\n" + "="*70)
        print("PREDICTION MODEL PERFORMANCE")
        print("="*70)
        print(f"Cross-validation R² scores: {cv_scores}")
        print(f"Mean R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Feature importance
        rf.fit(X, y)
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 most important features:")
        print("-" * 40)
        print(importance_df.head(10).to_string(index=False))
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df.head(10)['feature'], 
                importance_df.head(10)['importance'])
        plt.xlabel('Importance')
        plt.title('Top 10 Feature Importances for Score Prediction')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        self.results['feature_importance'] = importance_df
        self.results['prediction_model'] = rf
    
    def interpret_results(self):
        """
        Comprehensive interpretation of all results
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE RESULTS INTERPRETATION")
        print("="*70)
        
        # 1. Check correlations with scores
        if 'correlations' in self.results:
            corr_df = self.results['correlations']
            print("\n1. KEY FEATURES FOR SINGING QUALITY:")
            print("-" * 40)
            
            # Find features that strongly correlate with scores
            strong_corrs = corr_df[abs(corr_df['correlation']) > 0.3]
            if len(strong_corrs) > 0:
                for _, row in strong_corrs.iterrows():
                    direction = "positively" if row['correlation'] > 0 else "negatively"
                    print(f"   • {row['feature']}: {direction} correlated with scores")
            else:
                print("   No strong correlations found with individual features")
        
        # 2. Cluster analysis
        if 'clustered_df' in self.results:
            df = self.results['clustered_df']
            if 'true_score' in df.columns:
                print("\n2. CLUSTER PATTERNS:")
                print("-" * 40)
                
                # Get mean scores per cluster
                cluster_means = df.groupby('cluster')['true_score'].mean()
                best_cluster = cluster_means.idxmax()
                worst_cluster = cluster_means.idxmin()
                
                print(f"   • Best singing (highest scores): Cluster {best_cluster}")
                print(f"   • Poorest singing (lowest scores): Cluster {worst_cluster}")
                
                # Characterize each cluster
                print("\n3. CLUSTER CHARACTERISTICS:")
                feature_cols = ['peak_prominence', 'entropy', 'num_peaks', 'tonic_energy']
                for col in feature_cols:
                    if col in df.columns:
                        print(f"\n   {col}:")
                        for cluster in sorted(df['cluster'].unique()):
                            mean_val = df[df['cluster'] == cluster][col].mean()
                            print(f"     Cluster {cluster}: {mean_val:.3f}")
        
        # 3. Recommendations
        print("\n" + "="*70)
        print("RECOMMENDATIONS FOR AUTOMATIC EVALUATION:")
        print("="*70)
        print("""
Based on the analysis, consider these features for your automatic singing evaluation system:

1. PRIMARY METRICS (most indicative):
   • Peak prominence (higher is better - focused pitch)
   • Entropy (lower is better - stable singing)
   • Tonic energy (higher is better - good tonic adherence)
   • Number of peaks (lower is better - clean transitions)

2. SECONDARY METRICS:
   • Pitch stability (coefficient of variation)
   • Pitch range (appropriate range for the song)
   • Voiced ratio (amount of singing vs silence)

3. IMPLEMENTATION APPROACH:
   • Combine these features in a weighted model
   • Use cluster analysis to identify quality tiers
   • Validate against human scores
        """)

# Main execution function
def main():
    """
    Main function to run the complete analysis pipeline
    """
    print("="*70)
    print("UNSUPERVISED SINGING ANALYSIS SYSTEM")
    print("WITH LABELED DATASET")
    print("="*70)
    
    # Set your dataset paths
    dataset_path = r"D:\project_D\ASSE\dataset\raw_audio"
    labels_path = r"D:\project_D\ASSE\dataset\labels_classification.csv"
    
    # Verify paths exist
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset path not found: {dataset_path}")
        return
    
    if not os.path.exists(labels_path):
        print(f"ERROR: Labels file not found: {labels_path}")
        return
    
    # Initialize analyzer with your dataset
    analyzer = UnsupervisedSingingAnalyzer(dataset_path, labels_path)
    
    # Step 1: Analyze all songs in dataset
    print("\n" + "="*70)
    print("STEP 1: ANALYZING DATASET")
    print("="*70)
    analyzer.analyze_dataset()
    
    # Step 2: Analyze correlations with actual scores
    print("\n" + "="*70)
    print("STEP 2: CORRELATION ANALYSIS")
    print("="*70)
    analyzer.analyze_correlations_with_scores()
    
    # Step 3: Discover patterns through clustering
    print("\n" + "="*70)
    print("STEP 3: PATTERN DISCOVERY")
    print("="*70)
    analyzer.discover_patterns(n_clusters=2)
    
    # Step 4: Train prediction model
    print("\n" + "="*70)
    print("STEP 4: PREDICTION MODEL")
    print("="*70)
    analyzer.train_prediction_model()
    
    # Step 5: Comprehensive interpretation
    analyzer.interpret_results()
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Save features with scores
    if 'clustered_df' in analyzer.results:
        output_file = 'singing_features_with_scores.csv'
        analyzer.results['clustered_df'].to_csv(output_file, index=False)
        print(f"✅ Features saved to: {output_file}")
    
    # Save correlation results
    if 'correlations' in analyzer.results:
        corr_file = 'feature_correlations.csv'
        analyzer.results['correlations'].to_csv(corr_file, index=False)
        print(f"✅ Correlations saved to: {corr_file}")
    
    # Save feature importance
    if 'feature_importance' in analyzer.results:
        imp_file = 'feature_importance.csv'
        analyzer.results['feature_importance'].to_csv(imp_file, index=False)
        print(f"✅ Feature importance saved to: {imp_file}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()