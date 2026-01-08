"""
Model definitions and training functions for hybrid commodity price forecasting.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


def prepare_features(df: pd.DataFrame, target_column: str = 'target_price') -> Tuple:
    """
    Prepare feature matrices and target for model training.
    
    Args:
        df: DataFrame with all features created
        target_column: Name of target column
        
    Returns:
        Tuple of (X_price, X_news, X_hybrid, y, feature_names)
    """
    # Price features (extended with technical indicators)
    price_feature_cols = [
        # Basic lags
        'price_lag1', 'price_lag2', 'price_lag3', 'price_lag5', 'price_lag7', 'price_lag10',
        # Price differences (momentum)
        'price_diff_1_2', 'price_diff_1_5', 'price_diff_5_10',
        # Returns
        'return_lag1', 'return_lag2', 'return_lag5', 'return_lag7',
        # Moving averages
        'ma_5', 'ma_10', 'ma_20', 'ma_50',
        # Price relative to MA
        'price_to_ma5', 'price_to_ma10', 'price_to_ma20', 'price_to_ma50',
        # MA crossovers
        'ma5_ma10_cross', 'ma10_ma20_cross',
        # Volatility
        'volatility_5', 'volatility_10', 'volatility_20',
        # Technical indicators
        'rsi',  # Relative Strength Index
        'bb_width', 'bb_position',  # Bollinger Bands
        'momentum_5', 'momentum_10',  # Momentum
        'roc_5', 'roc_10',  # Rate of change
        'abs_return'
    ]
    
    # Stock features if available
    stock_feature_cols = [
        'lme_copper_stock_lag1', 'lme_copper_stock_lag2', 'lme_copper_stock_lag5',
        'lme_copper_stock_change_lag1', 'lme_copper_stock_change_lag5',
        'lme_copper_stock_ma_5', 'lme_copper_stock_ma_10',
        'lme_copper_stock_to_ma5', 'lme_copper_stock_to_ma10'
    ]
    
    # News features - all columns that start with 'news_'
    news_feature_cols = [col for col in df.columns 
                        if col.startswith('news_') 
                        and col != target_column
                        and not col.startswith('news_count')  # Keep news_count
                        and 'date' not in col.lower()]
    
    # Filter to only existing columns
    price_feature_cols = [col for col in price_feature_cols if col in df.columns]
    stock_feature_cols = [col for col in stock_feature_cols if col in df.columns]
    news_feature_cols = [col for col in news_feature_cols if col in df.columns]
    
    # Combine feature sets
    all_price_features = price_feature_cols + stock_feature_cols
    all_news_features = news_feature_cols
    all_hybrid_features = all_price_features + all_news_features
    
    # Extract feature matrices
    X_price = df[all_price_features].values if all_price_features else np.zeros((len(df), 1))
    X_news = df[all_news_features].values if all_news_features else np.zeros((len(df), 1))
    X_hybrid = df[all_hybrid_features].values
    
    # Target
    if target_column not in df.columns:
        available_targets = [c for c in df.columns if c.startswith('target_') or c == 'price_shock']
        raise ValueError(f"Target column '{target_column}' not found. Available: {available_targets}")
    y = df[target_column].values
    
    # Fill remaining NaN values (defensive approach)
    # Price features: fill with 0 or median
    if X_price.shape[1] > 0:
        price_nan_mask = np.isnan(X_price)
        if price_nan_mask.any():
            # Fill with column medians (or 0 if all NaN)
            for col_idx in range(X_price.shape[1]):
                col_data = X_price[:, col_idx]
                if np.isnan(col_data).any():
                    median_val = np.nanmedian(col_data)
                    if np.isnan(median_val):
                        median_val = 0.0
                    X_price[np.isnan(col_data), col_idx] = median_val
    
    # News features: fill with 0 (news features are often sparse)
    if X_news.shape[1] > 0:
        X_news = np.nan_to_num(X_news, nan=0.0)
    
    # Hybrid features: combine filled price and news
    X_hybrid = np.hstack([X_price, X_news]) if X_news.shape[1] > 0 else X_price
    
    # Remove rows with NaN in target only (features are now filled)
    valid_mask = ~np.isnan(y)
    X_price = X_price[valid_mask]
    X_news = X_news[valid_mask]
    X_hybrid = X_hybrid[valid_mask]
    y = y[valid_mask]
    
    if len(y) == 0:
        raise ValueError(f"No valid samples after filtering. Original shape: {len(df)}, NaN in target: {np.isnan(df[target_column].values).sum()}")
    
    feature_names = {
        'price': all_price_features,
        'news': all_news_features,
        'hybrid': all_hybrid_features
    }
    
    return X_price, X_news, X_hybrid, y, feature_names


def train_models(X_price, X_news, X_hybrid, y, test_size=0.2, random_state=42, 
                 use_walk_forward=False, n_windows=5, model_type='gbr', 
                 tune_hyperparams=False, save_models=False, save_dir='artifacts',
                 feature_names=None, select_topk_news_features=False, topk_news_features=15):
    """
    Train baseline and hybrid models for price/return prediction.
    
    This is a placeholder - implement based on your regression needs.
    """
    # Placeholder implementation
    results = {
        'model_price': None,
        'model_hybrid': None,
        'split_idx': int(len(y) * (1 - test_size))
    }
    return results


def train_shock_detection_model(
    X_price_train, X_news_train, X_hybrid_train, y_shock_train,
    X_price_test, X_news_test, X_hybrid_test, y_shock_test,
    model_type='gbr', random_state=42,
    calibrate_probabilities=True,
    tune_threshold=True,
    val_fraction=0.2,
    select_topk_news_features=True,
    topk_news_features=40,
    apply_pca_to_news=False  # Disabled by default - strict selection already reduces features
) -> Dict:
    """
    Train models for shock detection (binary classification).
    
    Trains Logistic Regression, Random Forest, SVM, and Gradient Boosting
    on both price-only and hybrid features.
    """
    from sklearn.model_selection import train_test_split as sk_train_test_split
    
    # Scale features
    scaler_price = StandardScaler()
    scaler_news = StandardScaler()
    scaler_hybrid = StandardScaler()
    
    X_price_train_scaled = scaler_price.fit_transform(X_price_train)
    X_price_test_scaled = scaler_price.transform(X_price_test)
    
    X_news_train_scaled = scaler_news.fit_transform(X_news_train) if X_news_train.shape[1] > 0 else X_news_train
    X_news_test_scaled = scaler_news.transform(X_news_test) if X_news_test.shape[1] > 0 else X_news_test
    
    # Feature selection for news features - keep most valuable while preserving signal
    # Balance between signal and noise: increased topk to allow more news features
    selected_news_indices = None
    if select_topk_news_features and X_news_train.shape[1] > 0 and topk_news_features < X_news_train.shape[1]:
        print(f"   🔍 Selecting top {topk_news_features} most valuable news features from {X_news_train.shape[1]}...")
        try:
            from sklearn.feature_selection import mutual_info_classif, f_classif
            from scipy.stats import pearsonr
            
            # Method 1: Mutual Information (captures non-linear relationships)
            mi_scores = mutual_info_classif(X_news_train, y_shock_train, random_state=random_state)
            
            # Method 2: F-statistic (linear relationships)
            f_scores, _ = f_classif(X_news_train, y_shock_train)
            f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Method 3: Correlation with target (direct predictive power)
            corr_scores = np.array([
                abs(pearsonr(X_news_train[:, i], y_shock_train)[0]) 
                if not np.isnan(X_news_train[:, i]).any() else 0.0
                for i in range(X_news_train.shape[1])
            ])
            corr_scores = np.nan_to_num(corr_scores, nan=0.0)
            
            # Method 4: Train a quick baseline model to get feature importance
            try:
                from sklearn.ensemble import RandomForestClassifier
                rf_quick = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=random_state, n_jobs=-1)
                rf_quick.fit(X_news_train, y_shock_train)
                importance_scores = rf_quick.feature_importances_
            except:
                importance_scores = np.zeros(X_news_train.shape[1])
            
            # Normalize all scores
            mi_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-10)
            f_norm = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min() + 1e-10)
            corr_norm = (corr_scores - corr_scores.min()) / (corr_scores.max() - corr_scores.min() + 1e-10)
            imp_norm = (importance_scores - importance_scores.min()) / (importance_scores.max() - importance_scores.min() + 1e-10)
            
            # Weighted combination: prioritize correlation and MI (strongest signals)
            # Even more weight on correlation for direct predictive power
            final_scores = 0.6 * corr_norm + 0.3 * mi_norm + 0.07 * imp_norm + 0.03 * f_norm
            
            # Very strict filtering: require top 10% by correlation OR top 10% by MI
            # This ensures we only keep the absolute best features
            min_corr_threshold = np.percentile(corr_scores, 90)  # Top 10% by correlation
            min_mi_threshold = np.percentile(mi_scores, 90)  # Top 10% by MI
            min_combined_threshold = np.percentile(final_scores, 80)  # Top 20% by combined score
            
            # Combined filter: must be in top 10% by correlation OR top 10% by MI, AND top 20% by combined
            valid_mask = (
                (corr_scores >= min_corr_threshold) | 
                (mi_scores >= min_mi_threshold)
            ) & (final_scores >= min_combined_threshold)
            
            if valid_mask.sum() > topk_news_features:
                # Take top K from valid features (prioritize strongest signals)
                top_indices = np.argsort(final_scores[valid_mask])[-topk_news_features:]
                # Map back to original indices
                valid_indices = np.where(valid_mask)[0]
                top_indices = valid_indices[top_indices]
            elif valid_mask.sum() > 0:
                # Use all valid features if fewer than topk
                top_indices = np.where(valid_mask)[0]
                print(f"   ⚠️  Only {len(top_indices)} features passed quality filters (requested {topk_news_features})")
            else:
                # Fallback: take top K overall if no features pass strict filters
                top_indices = np.argsort(final_scores)[-topk_news_features:]
                print(f"   ⚠️  No features passed strict filters, using top {topk_news_features} by combined score")
            
            selected_news_indices = top_indices
            X_news_train = X_news_train[:, top_indices]
            X_news_test = X_news_test[:, top_indices]
            
            print(f"   ✅ Selected {len(top_indices)} most valuable news features")
            print(f"      Score range: {final_scores[top_indices].min():.3f} - {final_scores[top_indices].max():.3f}")
            print(f"      MI range: {mi_scores[top_indices].min():.3f} - {mi_scores[top_indices].max():.3f}")
        except Exception as e:
            print(f"   ⚠️  Feature selection failed: {e}. Using all news features.")
    
    X_hybrid_train_scaled = scaler_hybrid.fit_transform(X_hybrid_train)
    X_hybrid_test_scaled = scaler_hybrid.transform(X_hybrid_test)
    
    # Reconstruct hybrid features if news features were selected
    if selected_news_indices is not None:
        X_news_train_scaled = scaler_news.fit_transform(X_news_train) if X_news_train.shape[1] > 0 else X_news_train
        X_news_test_scaled = scaler_news.transform(X_news_test) if X_news_test.shape[1] > 0 else X_news_test
    else:
        X_news_train_scaled = scaler_news.fit_transform(X_news_train) if X_news_train.shape[1] > 0 else X_news_train
        X_news_test_scaled = scaler_news.transform(X_news_test) if X_news_test.shape[1] > 0 else X_news_test
    
    # Apply PCA to news features only if still high-dimensional after strict selection
    # With stricter selection (25 features), PCA is usually not needed
    if apply_pca_to_news and X_news_train_scaled.shape[1] > 30:
        try:
            from sklearn.decomposition import PCA
            # More conservative PCA: keep 95% variance (preserve more signal)
            n_components = min(25, int(X_news_train_scaled.shape[1] * 0.9))  # Keep 90% of features or max 25
            pca_news = PCA(n_components=n_components, random_state=random_state)
            X_news_train_pca = pca_news.fit_transform(X_news_train_scaled)
            X_news_test_pca = pca_news.transform(X_news_test_scaled)
            explained_var = pca_news.explained_variance_ratio_.sum()
            print(f"   📉 Applied PCA to news features: {X_news_train_scaled.shape[1]} → {X_news_train_pca.shape[1]} (explained variance: {explained_var:.2%})")
            # Only use PCA if it preserves >90% variance
            if explained_var >= 0.90:
                X_news_train_scaled = X_news_train_pca
                X_news_test_scaled = X_news_test_pca
            else:
                print(f"   ⚠️  PCA explained variance too low ({explained_var:.2%}), using original features")
        except Exception as e:
            print(f"   ⚠️  PCA failed: {e}. Using original news features.")
    else:
        print(f"   ✅ Using {X_news_train_scaled.shape[1]} selected news features (no PCA needed)")
    
    # 2. Create price-news interaction features (enhanced with multiple interaction types)
    # Use smarter feature selection and diverse interaction types
    if X_news_train_scaled.shape[1] > 0 and X_price_train_scaled.shape[1] > 0:
        try:
            from scipy.stats import pearsonr
            from sklearn.feature_selection import mutual_info_classif
            
            # Select top price features by correlation with target (smarter selection)
            price_corr = np.array([
                abs(pearsonr(X_price_train_scaled[:, i], y_shock_train[:len(X_price_train_scaled)])[0])
                if not np.isnan(X_price_train_scaled[:, i]).any() else 0.0
                for i in range(X_price_train_scaled.shape[1])
            ])
            price_corr = np.nan_to_num(price_corr, nan=0.0)
            
            # Also use mutual information for price features
            try:
                price_mi = mutual_info_classif(X_price_train_scaled, y_shock_train[:len(X_price_train_scaled)], random_state=random_state)
                price_mi = np.nan_to_num(price_mi, nan=0.0)
                # Combine correlation and MI (weighted)
                price_scores = 0.6 * price_corr + 0.4 * (price_mi / (price_mi.max() + 1e-10))
            except:
                price_scores = price_corr
            
            # Select top 4 price features (further reduced for strongest interactions only)
            top_price_idx = np.argsort(price_scores)[-min(4, X_price_train_scaled.shape[1]):]
            
            # Select top news features by correlation and MI (focus on strongest signals)
            news_corr = np.array([
                abs(pearsonr(X_news_train_scaled[:, i], y_shock_train[:len(X_news_train_scaled)])[0])
                if not np.isnan(X_news_train_scaled[:, i]).any() else 0.0
                for i in range(X_news_train_scaled.shape[1])
            ])
            news_corr = np.nan_to_num(news_corr, nan=0.0)
            
            try:
                news_mi = mutual_info_classif(X_news_train_scaled, y_shock_train[:len(X_news_train_scaled)], random_state=random_state)
                news_mi = np.nan_to_num(news_mi, nan=0.0)
                # Combine correlation and MI (prioritize correlation for direct signal)
                news_scores = 0.75 * news_corr + 0.25 * (news_mi / (news_mi.max() + 1e-10))
            except:
                news_scores = news_corr
            
            # Select top 4 news features (further reduced for strongest interactions only)
            top_news_idx = np.argsort(news_scores)[-min(4, X_news_train_scaled.shape[1]):]
            
            # Create diverse interaction types (multiplication, division, difference)
            # This captures different types of relationships between price and news
            n_price_top = len(top_price_idx)
            n_news_top = len(top_news_idx)
            n_interactions_per_type = n_price_top * n_news_top
            
            # Initialize interaction matrices (3 types: multiply, divide, difference)
            interactions_train = np.zeros((X_price_train_scaled.shape[0], n_interactions_per_type * 3))
            interactions_test = np.zeros((X_price_test_scaled.shape[0], n_interactions_per_type * 3))
            
            idx = 0
            for p_idx in top_price_idx:
                for n_idx in top_news_idx:
                    price_feat = X_price_train_scaled[:, p_idx]
                    news_feat = X_news_train_scaled[:, n_idx]
                    
                    # Type 1: Multiplication (captures joint effects)
                    interactions_train[:, idx] = price_feat * news_feat
                    interactions_test[:, idx] = X_price_test_scaled[:, p_idx] * X_news_test_scaled[:, n_idx]
                    idx += 1
                    
                    # Type 2: Division (captures relative effects, avoid division by zero)
                    news_feat_safe = news_feat + np.sign(news_feat) * 1e-8  # Add small epsilon
                    interactions_train[:, idx] = price_feat / (news_feat_safe + 1e-8)
                    interactions_test[:, idx] = X_price_test_scaled[:, p_idx] / (X_news_test_scaled[:, n_idx] + 1e-8)
                    idx += 1
                    
                    # Type 3: Difference (captures absolute differences)
                    interactions_train[:, idx] = price_feat - news_feat
                    interactions_test[:, idx] = X_price_test_scaled[:, p_idx] - X_news_test_scaled[:, n_idx]
                    idx += 1
            
            # Scale interactions
            scaler_interactions = StandardScaler()
            interactions_train_scaled = scaler_interactions.fit_transform(interactions_train)
            interactions_test_scaled = scaler_interactions.transform(interactions_test)
            
            # Concatenate: price + news + interactions
            X_hybrid_train_scaled = np.hstack([X_price_train_scaled, X_news_train_scaled, interactions_train_scaled])
            X_hybrid_test_scaled = np.hstack([X_price_test_scaled, X_news_test_scaled, interactions_test_scaled])
            print(f"   ✅ Created {interactions_train.shape[1]} price-news interaction features (top {n_price_top} price × top {n_news_top} news × 3 types: multiply/divide/difference)")
            print(f"      Total hybrid features: {X_hybrid_train_scaled.shape[1]} (price: {X_price_train_scaled.shape[1]}, news: {X_news_train_scaled.shape[1]}, interactions: {interactions_train_scaled.shape[1]})")
        except Exception as e:
            print(f"   ⚠️  Interaction features failed: {e}. Using simple concatenation.")
            X_hybrid_train_scaled = np.hstack([X_price_train_scaled, X_news_train_scaled]) if X_news_train_scaled.shape[1] > 0 else X_price_train_scaled
            X_hybrid_test_scaled = np.hstack([X_price_test_scaled, X_news_test_scaled]) if X_news_test_scaled.shape[1] > 0 else X_price_test_scaled
    else:
        X_hybrid_train_scaled = np.hstack([X_price_train_scaled, X_news_train_scaled]) if X_news_train_scaled.shape[1] > 0 else X_price_train_scaled
        X_hybrid_test_scaled = np.hstack([X_price_test_scaled, X_news_test_scaled]) if X_news_test_scaled.shape[1] > 0 else X_price_test_scaled
    
    # Time-aware train/val split
    split_idx = int(len(y_shock_train) * (1 - val_fraction))
    X_price_val = X_price_train_scaled[split_idx:]
    X_hybrid_val = X_hybrid_train_scaled[split_idx:]
    y_val = y_shock_train[split_idx:]
    
    X_price_tr = X_price_train_scaled[:split_idx]
    X_hybrid_tr = X_hybrid_train_scaled[:split_idx]
    y_tr = y_shock_train[:split_idx]
    
    # Class balancing with SMOTE if needed
    use_smote = SMOTE_AVAILABLE and y_tr.mean() < 0.1
    if use_smote:
        try:
            smote = SMOTE(random_state=random_state, sampling_strategy=0.3)
            X_price_tr, y_tr_price = smote.fit_resample(X_price_tr, y_tr)
            X_hybrid_tr, y_tr_hybrid = smote.fit_resample(X_hybrid_tr, y_tr)
        except:
            use_smote = False
            y_tr_price = y_tr
            y_tr_hybrid = y_tr
    else:
        y_tr_price = y_tr
        y_tr_hybrid = y_tr
    
    results = {}
    all_models = {}
    all_metrics = {}
    all_test_proba = {}
    all_test_pred = {}
    
    models_to_train = [
        ('Logistic Regression', 'lr'),
        ('Random Forest', 'rf'),
        ('SVM', 'svm'),
        ('Gradient Boosting', 'gbr')
    ]
    
    # Train on Price-only
    print("\n📊 Price-Only Features:")
    for model_name, model_code in models_to_train:
        print(f"  🔵 Training {model_name}...")
        
        if model_code == 'lr':
            model = LogisticRegression(
                random_state=random_state, max_iter=1000, class_weight='balanced',
                C=0.001,  # Strong L2 regularization
                penalty='l2',
                solver='lbfgs'
            )
            model.fit(X_price_tr, y_tr_price)
        elif model_code == 'rf':
            # Improved Random Forest: balanced depth to avoid overfitting
            # Reduced depth and increased min_samples to prevent overfitting on imbalanced data
            model = RandomForestClassifier(
                n_estimators=300,  # Sufficient trees
                max_depth=10,  # Reduced depth to prevent overfitting
                min_samples_split=10,  # Increased to prevent overfitting
                min_samples_leaf=5,  # Increased to prevent overfitting
                random_state=random_state, class_weight='balanced', n_jobs=-1,
                max_features='sqrt',  # Feature sampling for diversity
                bootstrap=True,
                oob_score=True  # Out-of-bag score for monitoring
            )
            model.fit(X_price_tr, y_tr_price)
        elif model_code == 'svm':
            if len(X_price_tr) > 5000:
                sample_idx = np.random.choice(len(X_price_tr), 5000, replace=False)
                X_train_svm = X_price_tr[sample_idx]
                y_train_svm = y_tr[sample_idx]
            else:
                X_train_svm = X_price_tr
                y_train_svm = y_tr_price
            model = SVC(kernel='rbf', probability=True, random_state=random_state, class_weight='balanced', C=1.0, gamma='scale')
            model.fit(X_train_svm, y_train_svm)
        elif model_code == 'gbr':
            # Optimized Gradient Boosting: best model (Proposed Model in article)
            # Key: more trees, balanced depth, optimal learning rate
            model = GradientBoostingClassifier(
                n_estimators=400,  # More trees for better learning
                max_depth=7,  # Balanced depth - deep enough but not too deep
                learning_rate=0.07,  # Balanced learning rate
                random_state=random_state,
                subsample=0.85,  # Row sampling for regularization
                min_samples_split=8,  # Moderate regularization
                min_samples_leaf=4,  # Moderate regularization
                max_features='sqrt',  # Column sampling
                loss='log_loss',
                # Early stopping to prevent overfitting
                validation_fraction=0.1,
                n_iter_no_change=25,  # Stop early if no improvement
                tol=1e-4
            )
            model.fit(X_price_tr, y_tr_price)
        
        # Calibration
        calibrated_flag = False
        cal_model = None
        if calibrate_probabilities and len(np.unique(y_val)) == 2:
            try:
                # Use cross-validation for calibration (newer sklearn versions)
                cal_model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
                cal_model.fit(X_price_val, y_val)
                proba_test = cal_model.predict_proba(X_price_test_scaled)[:, 1]
                calibrated_flag = True
            except Exception as e:
                # Fallback: try prefit if available, otherwise skip calibration
                try:
                    from sklearn.calibration import CalibratedClassifierCV
                    # For older sklearn or if prefit works
                    cal_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
                    cal_model.fit(X_price_val, y_val)
                    proba_test = cal_model.predict_proba(X_price_test_scaled)[:, 1]
                    calibrated_flag = True
                except:
                    proba_test = model.predict_proba(X_price_test_scaled)[:, 1]
        else:
            proba_test = model.predict_proba(X_price_test_scaled)[:, 1]
        
        # Threshold tuning
        if tune_threshold and len(np.unique(y_val)) == 2:
            if calibrated_flag and cal_model is not None:
                proba_val = cal_model.predict_proba(X_price_val)[:, 1]
            else:
                proba_val = model.predict_proba(X_price_val)[:, 1]
            
            # Expand threshold range, especially for models that predict low probabilities (like RF)
            # Check if max probability is low - if so, use lower thresholds
            max_proba = proba_val.max()
            if max_proba < 0.3:
                # For models with very low probabilities (e.g., RF), search lower thresholds
                thresholds = np.concatenate([
                    np.arange(0.01, 0.1, 0.01),  # Very low thresholds
                    np.arange(0.1, 0.5, 0.01)   # Low-medium thresholds
                ])
            else:
                thresholds = np.arange(0.05, 0.95, 0.01)  # Standard range
            
            best_thr = 0.5
            best_f1 = -1
            for thr in thresholds:
                y_pred_val = (proba_val >= thr).astype(int)
                if y_pred_val.sum() == 0:  # Skip if no positive predictions
                    continue
                f1v = f1_score(y_val, y_pred_val, zero_division=0)
                if f1v > best_f1:
                    best_f1 = f1v
                    best_thr = thr
            
            # Fallback: if no threshold found, use a very low one
            if best_f1 == -1 or best_f1 == 0:
                # Try even lower thresholds
                for thr in np.arange(0.001, 0.1, 0.001):
                    y_pred_val = (proba_val >= thr).astype(int)
                    if y_pred_val.sum() > 0:
                        f1v = f1_score(y_val, y_pred_val, zero_division=0)
                        if f1v > best_f1:
                            best_f1 = f1v
                            best_thr = thr
                            break
            
            thr = best_thr if best_f1 > 0 else 0.01  # Fallback to very low threshold
        else:
            thr = 0.5
        
        y_pred = (proba_test >= thr).astype(int)
        
        # Metrics
        metrics = {
            'auc': roc_auc_score(y_shock_test, proba_test) if len(np.unique(y_shock_test)) > 1 else 0.0,
            'pr_auc': average_precision_score(y_shock_test, proba_test) if len(np.unique(y_shock_test)) > 1 else 0.0,
            'precision': precision_score(y_shock_test, y_pred, zero_division=0),
            'recall': recall_score(y_shock_test, y_pred, zero_division=0),
            'f1': f1_score(y_shock_test, y_pred, zero_division=0),
            'threshold': thr,
            'calibrated': calibrated_flag
        }
        
        all_models[f'{model_code}_price'] = model
        all_metrics[f'{model_code}_price'] = metrics
        all_test_proba[f'{model_code}_price'] = proba_test
        all_test_pred[f'{model_code}_price'] = y_pred
        
        print(f"     AUC: {metrics['auc']:.3f}, F1: {metrics['f1']:.3f} (thr={thr:.2f})")
    
    # Train on Hybrid features
    print("\n📊 Hybrid Features (Price + News):")
    for model_name, model_code in models_to_train:
        print(f"  🟢 Training {model_name}...")
        
        if model_code == 'lr':
            model = LogisticRegression(
                random_state=random_state, max_iter=3000, class_weight='balanced',
                C=0.15,
                penalty='elasticnet', l1_ratio=0.5, solver='saga'
            )
            model.fit(X_hybrid_tr, y_tr_hybrid)
        elif model_code == 'rf':
            # Optimized for hybrid features: balanced depth to avoid overfitting
            # Reduced depth and increased min_samples to prevent overfitting on imbalanced data
            model = RandomForestClassifier(
                n_estimators=300,  # Sufficient trees
                max_depth=12,  # Reduced depth to prevent overfitting while capturing interactions
                min_samples_split=8,  # Increased to prevent overfitting
                min_samples_leaf=4,  # Increased to prevent overfitting
                random_state=random_state, class_weight='balanced', n_jobs=-1,
                max_features='sqrt',  # Feature sampling for diversity
                bootstrap=True,
                oob_score=True  # Out-of-bag score for monitoring
            )
            model.fit(X_hybrid_tr, y_tr_hybrid)
        elif model_code == 'svm':
            if len(X_hybrid_tr) > 5000:
                sample_idx = np.random.choice(len(X_hybrid_tr), 5000, replace=False)
                X_train_svm = X_hybrid_tr[sample_idx]
                y_train_svm = y_tr_hybrid[sample_idx]
            else:
                X_train_svm = X_hybrid_tr
                y_train_svm = y_tr_hybrid
            model = SVC(kernel='rbf', probability=True, random_state=random_state, class_weight='balanced', C=1.0, gamma='scale')
            model.fit(X_train_svm, y_train_svm)
        elif model_code == 'gbr':
            # Optimized Gradient Boosting for hybrid features (Proposed Model in article)
            n_negative = (y_tr_hybrid == 0).sum()
            n_positive = (y_tr_hybrid == 1).sum()
            if n_positive > 0 and n_negative > 0:
                # Best hyperparameters for hybrid model - optimized for performance with high-quality features
                # Increased capacity to learn from strong news signals
                model = GradientBoostingClassifier(
                    n_estimators=600,  # More trees for better learning
                    max_depth=10,  # Deeper to capture complex interactions
                    learning_rate=0.05,  # Slightly lower LR for more stable learning
                    random_state=random_state,
                    subsample=0.8,  # More regularization
                    min_samples_split=10,  # Balanced regularization
                    min_samples_leaf=5,  # Balanced regularization
                    max_features='sqrt',  # Column sampling
                    loss='log_loss',
                    # Early stopping via validation_fraction
                    validation_fraction=0.1,
                    n_iter_no_change=25,  # Stop early if no improvement
                    tol=1e-4
                )
                model.fit(X_hybrid_tr, y_tr_hybrid)
            else:
                # Fallback if no positive samples
                model = GradientBoostingClassifier(
                    n_estimators=400, max_depth=7, learning_rate=0.07,
                    random_state=random_state, subsample=0.85, min_samples_split=8, min_samples_leaf=4,
                    max_features='sqrt', loss='log_loss'
                )
                model.fit(X_hybrid_tr, y_tr_hybrid)
        
        # Calibration
        calibrated_flag = False
        cal_model = None
        if calibrate_probabilities and len(np.unique(y_val)) == 2:
            try:
                # Use cross-validation for calibration (newer sklearn versions)
                cal_model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
                cal_model.fit(X_hybrid_val, y_val)
                proba_test = cal_model.predict_proba(X_hybrid_test_scaled)[:, 1]
                calibrated_flag = True
            except Exception as e:
                # Fallback: try prefit if available, otherwise skip calibration
                try:
                    from sklearn.calibration import CalibratedClassifierCV
                    # For older sklearn or if prefit works
                    cal_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
                    cal_model.fit(X_hybrid_val, y_val)
                    proba_test = cal_model.predict_proba(X_hybrid_test_scaled)[:, 1]
                    calibrated_flag = True
                except:
                    proba_test = model.predict_proba(X_hybrid_test_scaled)[:, 1]
        else:
            proba_test = model.predict_proba(X_hybrid_test_scaled)[:, 1]
        
        # Threshold tuning
        if tune_threshold and len(np.unique(y_val)) == 2:
            if calibrated_flag and cal_model is not None:
                proba_val = cal_model.predict_proba(X_hybrid_val)[:, 1]
            else:
                proba_val = model.predict_proba(X_hybrid_val)[:, 1]
            
            # Expand threshold range, especially for models that predict low probabilities (like RF)
            # Check if max probability is low - if so, use lower thresholds
            max_proba = proba_val.max()
            if max_proba < 0.3:
                # For models with very low probabilities (e.g., RF), search lower thresholds
                thresholds = np.concatenate([
                    np.arange(0.01, 0.1, 0.01),  # Very low thresholds
                    np.arange(0.1, 0.5, 0.01)   # Low-medium thresholds
                ])
            else:
                thresholds = np.arange(0.05, 0.95, 0.01)  # Standard range
            
            best_thr = 0.5
            best_f1 = -1
            for thr in thresholds:
                y_pred_val = (proba_val >= thr).astype(int)
                if y_pred_val.sum() == 0:  # Skip if no positive predictions
                    continue
                f1v = f1_score(y_val, y_pred_val, zero_division=0)
                if f1v > best_f1:
                    best_f1 = f1v
                    best_thr = thr
            
            # Fallback: if no threshold found, use a very low one
            if best_f1 == -1 or best_f1 == 0:
                # Try even lower thresholds
                for thr in np.arange(0.001, 0.1, 0.001):
                    y_pred_val = (proba_val >= thr).astype(int)
                    if y_pred_val.sum() > 0:
                        f1v = f1_score(y_val, y_pred_val, zero_division=0)
                        if f1v > best_f1:
                            best_f1 = f1v
                            best_thr = thr
                            break
            
            thr = best_thr if best_f1 > 0 else 0.01  # Fallback to very low threshold
        else:
            thr = 0.5
        
        y_pred = (proba_test >= thr).astype(int)
        
        # Metrics
        metrics = {
            'auc': roc_auc_score(y_shock_test, proba_test) if len(np.unique(y_shock_test)) > 1 else 0.0,
            'pr_auc': average_precision_score(y_shock_test, proba_test) if len(np.unique(y_shock_test)) > 1 else 0.0,
            'precision': precision_score(y_shock_test, y_pred, zero_division=0),
            'recall': recall_score(y_shock_test, y_pred, zero_division=0),
            'f1': f1_score(y_shock_test, y_pred, zero_division=0),
            'threshold': thr,
            'calibrated': calibrated_flag
        }
        
        all_models[f'{model_code}_hybrid'] = model
        all_metrics[f'{model_code}_hybrid'] = metrics
        all_test_proba[f'{model_code}_hybrid'] = proba_test
        all_test_pred[f'{model_code}_hybrid'] = y_pred
        
        print(f"     AUC: {metrics['auc']:.3f}, F1: {metrics['f1']:.3f} (thr={thr:.2f})")
    
    # Build feature names for SHAP (after all transformations)
    # Track what transformations were applied
    final_feature_names = []
    
    # Price features (always included, no transformation except scaling)
    n_price = X_price_train_scaled.shape[1]
    final_feature_names.extend([f'price_feat_{i}' for i in range(n_price)])
    
    # News features (after selection and possibly PCA)
    n_news = X_news_train_scaled.shape[1]
    if n_news > 0:
        # Check if PCA was applied (we can't know for sure, but if n_news is small, likely PCA)
        if selected_news_indices is not None and n_news <= topk_news_features:
            # Features were selected, use generic names
            final_feature_names.extend([f'news_selected_{i}' for i in range(n_news)])
    else:
            # Might be PCA or original, use generic names
            final_feature_names.extend([f'news_feat_{i}' for i in range(n_news)])
    
    # Interaction features (if created)
    # Check if interactions were created by comparing total features
    n_total_expected = n_price + n_news
    n_total_actual = X_hybrid_train_scaled.shape[1]
    if n_total_actual > n_total_expected:
        # Interactions were added
        n_interactions = n_total_actual - n_total_expected
        final_feature_names.extend([f'price_news_interaction_{i}' for i in range(n_interactions)])
    
    # Ensure we have the right number of names
    if len(final_feature_names) != X_hybrid_train_scaled.shape[1]:
        # Fallback: just use generic names
        final_feature_names = [f'feature_{i}' for i in range(X_hybrid_train_scaled.shape[1])]
    
    return {
        'all_models': all_models,
        'all_metrics': all_metrics,
        'all_test_proba': all_test_proba,
        'all_test_pred': all_test_pred,
        'X_hybrid_test_scaled': X_hybrid_test_scaled,
        'y_test': y_shock_test,
        'feature_names_hybrid': final_feature_names  # Store feature names for SHAP
    }
