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
    # Price features
    price_feature_cols = [
        'price_lag1', 'price_lag2', 'price_lag3', 'price_lag5',
        'return_lag1', 'return_lag2', 'return_lag5',
        'ma_5', 'ma_10', 'ma_20',
        'price_to_ma5', 'price_to_ma10',
        'volatility_5', 'volatility_10',
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
    
    # Remove rows with NaN
    valid_mask = ~(np.isnan(X_hybrid).any(axis=1) | np.isnan(y))
    X_price = X_price[valid_mask]
    X_news = X_news[valid_mask]
    X_hybrid = X_hybrid[valid_mask]
    y = y[valid_mask]
    
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
    topk_news_features=40
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
            
            # Weighted combination: prioritize MI and importance, then correlation, then F-stat
            final_scores = 0.4 * mi_norm + 0.3 * imp_norm + 0.2 * corr_norm + 0.1 * f_norm
            
            # Filter out features with very low scores (noise)
            min_score_threshold = np.percentile(final_scores, max(0, 100 - (topk_news_features * 100 / X_news_train.shape[1])))
            valid_mask = final_scores >= min_score_threshold
            
            if valid_mask.sum() > topk_news_features:
                # Take top K from valid features
                top_indices = np.argsort(final_scores[valid_mask])[-topk_news_features:]
                # Map back to original indices
                valid_indices = np.where(valid_mask)[0]
                top_indices = valid_indices[top_indices]
            else:
                # Take top K overall
                top_indices = np.argsort(final_scores)[-topk_news_features:]
            
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
    
    # Enhanced hybrid feature engineering inspired by dual-stream architecture
    # Skip PCA if we already have few features (aggressive selection already done)
    # Only apply PCA if we have many features to reduce noise
    if X_news_train_scaled.shape[1] > 30:
        try:
            from sklearn.decomposition import PCA
            # Keep 90% variance for news features (less aggressive to preserve more signal)
            n_components = min(35, int(X_news_train_scaled.shape[1] * 0.85))  # Keep more components
            pca_news = PCA(n_components=n_components, random_state=random_state)
            X_news_train_pca = pca_news.fit_transform(X_news_train_scaled)
            X_news_test_pca = pca_news.transform(X_news_test_scaled)
            print(f"   📉 Applied PCA to news features: {X_news_train_scaled.shape[1]} → {X_news_train_pca.shape[1]} (explained variance: {pca_news.explained_variance_ratio_.sum():.2%})")
            X_news_train_scaled = X_news_train_pca
            X_news_test_scaled = X_news_test_pca
        except Exception as e:
            print(f"   ⚠️  PCA failed: {e}. Using original news features.")
    else:
        print(f"   ✅ Using {X_news_train_scaled.shape[1]} selected news features (no PCA needed)")
    
    # 2. Create price-news interaction features (inspired by attention mechanism)
    # Multiply top price features with top news features - but be selective
    if X_news_train_scaled.shape[1] > 0 and X_price_train_scaled.shape[1] > 0:
        try:
            from scipy.stats import pearsonr
            # Select top 5 price features by correlation with target (increased from 3)
            price_corr = np.array([
                abs(pearsonr(X_price_train_scaled[:, i], y_shock_train[:len(X_price_train_scaled)])[0])
                if not np.isnan(X_price_train_scaled[:, i]).any() else 0.0
                for i in range(X_price_train_scaled.shape[1])
            ])
            price_corr = np.nan_to_num(price_corr, nan=0.0)
            top_price_idx = np.argsort(price_corr)[-min(5, X_price_train_scaled.shape[1]):]
            
            # Select top 5 news features (increased from 3 to capture more interactions)
            top_news_idx = list(range(min(5, X_news_train_scaled.shape[1])))
            
            # Create interactions (5x5 = 25 interactions for richer feature space)
            interactions_train = np.zeros((X_price_train_scaled.shape[0], len(top_price_idx) * len(top_news_idx)))
            interactions_test = np.zeros((X_price_test_scaled.shape[0], len(top_price_idx) * len(top_news_idx)))
            
            idx = 0
            for p_idx in top_price_idx:
                for n_idx in top_news_idx:
                    interactions_train[:, idx] = X_price_train_scaled[:, p_idx] * X_news_train_scaled[:, n_idx]
                    interactions_test[:, idx] = X_price_test_scaled[:, p_idx] * X_news_test_scaled[:, n_idx]
                    idx += 1
            
            # Scale interactions
            scaler_interactions = StandardScaler()
            interactions_train_scaled = scaler_interactions.fit_transform(interactions_train)
            interactions_test_scaled = scaler_interactions.transform(interactions_test)
            
            # Concatenate: price + news + interactions
            X_hybrid_train_scaled = np.hstack([X_price_train_scaled, X_news_train_scaled, interactions_train_scaled])
            X_hybrid_test_scaled = np.hstack([X_price_test_scaled, X_news_test_scaled, interactions_test_scaled])
            print(f"   ✅ Created {interactions_train.shape[1]} price-news interaction features (top 5 price × top 5 news)")
        except Exception as e:
            print(f"   ⚠️  Interaction features failed: {e}. Using simple concatenation.")
            X_hybrid_train_scaled = np.hstack([X_price_train_scaled, X_news_train_scaled])
            X_hybrid_test_scaled = np.hstack([X_price_test_scaled, X_news_test_scaled])
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
            model = LogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced')
            model.fit(X_price_tr, y_tr_price)
        elif model_code == 'rf':
            # Baseline for price-only
            model = RandomForestClassifier(
                n_estimators=150, max_depth=12, min_samples_split=5, min_samples_leaf=2,
                random_state=random_state, class_weight='balanced', n_jobs=-1, max_features='sqrt'
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
            # Optimized for price-only with imbalanced data
            model = GradientBoostingClassifier(
                n_estimators=600, max_depth=7, learning_rate=0.04,
                random_state=random_state, subsample=0.85, min_samples_split=12, min_samples_leaf=6,
                max_features='sqrt', loss='log_loss',
                validation_fraction=0.1,
                n_iter_no_change=20,
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
            thresholds = np.arange(0.1, 0.9, 0.01)
            best_thr = 0.5
            best_f1 = -1
            for thr in thresholds:
                y_pred_val = (proba_val >= thr).astype(int)
                f1v = f1_score(y_val, y_pred_val, zero_division=0)
                if f1v > best_f1:
                    best_f1 = f1v
                    best_thr = thr
            thr = best_thr
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
            # Improved for hybrid: Elastic Net regularization to handle many features
            model = LogisticRegression(
                random_state=random_state, max_iter=3000, class_weight='balanced',
                C=0.5, penalty='elasticnet', l1_ratio=0.5, solver='saga'
            )
            model.fit(X_hybrid_tr, y_tr_hybrid)
        elif model_code == 'rf':
            # Optimized for hybrid features: more trees, deeper, better regularization
            # Inspired by article: more capacity to learn price-news interactions
            model = RandomForestClassifier(
                n_estimators=300, max_depth=18, min_samples_split=2, min_samples_leaf=1,
                random_state=random_state, class_weight='balanced', n_jobs=-1,
                max_features='sqrt', bootstrap=True
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
            # Optimized for imbalanced data to beat Logistic Regression
            # Calculate class weight for imbalanced data
            n_negative = (y_tr_hybrid == 0).sum()
            n_positive = (y_tr_hybrid == 1).sum()
            if n_positive > 0 and n_negative > 0:
                # Optimized hyperparameters for imbalanced classification
                # Use early stopping and stronger regularization to prevent overfitting
                # Deeper trees to capture complex patterns, but with strong regularization
                model = GradientBoostingClassifier(
                    n_estimators=800,  # More trees for better learning
                    max_depth=8,  # Slightly reduced to prevent overfitting
                    learning_rate=0.03,  # Lower LR for more stable learning
                    random_state=random_state,
                    subsample=0.85,  # Row sampling
                    min_samples_split=15,  # Stronger regularization
                    min_samples_leaf=8,  # Stronger regularization
                    max_features='sqrt',  # Column sampling
                    loss='log_loss',
                    # Early stopping via validation_fraction
                    validation_fraction=0.1,
                    n_iter_no_change=20,  # Stop if no improvement for 20 iterations
                    tol=1e-4
                )
                model.fit(X_hybrid_tr, y_tr_hybrid)
            else:
                # Fallback if no positive samples
                model = GradientBoostingClassifier(
                    n_estimators=600, max_depth=10, learning_rate=0.05,
                    random_state=random_state, subsample=0.8, min_samples_split=8, min_samples_leaf=3,
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
            thresholds = np.arange(0.1, 0.9, 0.01)
            best_thr = 0.5
            best_f1 = -1
            for thr in thresholds:
                y_pred_val = (proba_val >= thr).astype(int)
                f1v = f1_score(y_val, y_pred_val, zero_division=0)
                if f1v > best_f1:
                    best_f1 = f1v
                    best_thr = thr
            thr = best_thr
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
