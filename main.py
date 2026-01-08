"""
Main script for hybrid commodity price forecasting.

This script demonstrates the complete pipeline:
1. Load and align price and news data
2. Create price-based features
3. Create news-based features (TF-IDF + heuristics)
4. Train baseline and hybrid models
5. Evaluate and compare results

Research Question:
Does incorporating news sentiment improve copper price forecasting
compared to using price data alone?
"""

from src.data.data_loader import align_price_and_news
from src.features.price_features import create_price_features
from src.features.sentiment_features import create_news_features, validate_sentiment_features
from src.models import prepare_features, train_models, train_shock_detection_model
from src.evaluation import (
    compare_models, compare_models_walk_forward,
    plot_forecasts, plot_feature_importance, plot_correlation_heatmap,
    create_summary_table, explain_with_shap, analyze_prediction_errors,
    print_shock_detection_metrics, plot_shock_detection_results,
    plot_feature_importance_shock, plot_price_with_shocks, plot_top_news_events,
    plot_news_statistics
)
from pathlib import Path
import numpy as np


def main():
    """
    Main pipeline: load data → create features → train models → evaluate
    """
    print("="*80)
    print("🚀 Hybrid Commodity Price Forecasting")
    print("="*80)
    print("\nResearch Question:")
    print("Does incorporating news sentiment improve copper price forecasting")
    print("compared to using price data alone?")
    
    # Step 1: Load and align data
    print("\n" + "="*80)
    print("📥 Step 1: Loading and aligning data...")
    print("="*80)
    # Load news data with relaxed filters for better coverage
    # Use all sources and all news (not just "sufficient") to maximize signal
    df = align_price_and_news(
        filter_sufficient_news=False,  # Include all news, not just "sufficient"
        allowed_sources=None,  # Use all available sources for better coverage
        drop_price_recap_only=True # Drop price recap only news
    )
    print(f"✅ Loaded {len(df)} records")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Shock definition parameters (adjustable to increase/decrease shock rate)
    # Defined early so they're available for both feature creation and reporting
    shock_window = 2  # Number of days for cumulative return (reduced from 3)
    k_sigma = 1.25  # Threshold in std devs (reduced from 1.5) - lower = more shocks
    require_same_direction = True  # If False, allows mixed-direction moves (more shocks)
    
    # Step 2: Create price features
    print("\n" + "="*80)
    print("🔧 Step 2: Creating price features...")
    print("="*80)
    df = create_price_features(df, shock_window=shock_window, k_sigma=k_sigma, require_same_direction=require_same_direction)
    print(f"✅ Created price and stock features")
    
    # Step 3: Create news features
    print("\n" + "="*80)
    print("🔧 Step 3: Creating news features...")
    print("="*80)
    # Prefer compact FinBERT sentiment scores over high-dimensional embeddings
    df = create_news_features(df, use_finbert=True, finbert_mode='sentiment', verbose=True)
    print(f"✅ Created news features (FinBERT/TF-IDF embeddings + heuristics + rolling/lags)")
    
    # Step 3.5: Validate sentiment features
    print("\n" + "="*80)
    print("🔍 Step 3.5: Validating sentiment features...")
    print("="*80)
    sentiment_validation = validate_sentiment_features(df, verbose=True)
    
    # Step 4: Prepare feature matrices
    print("\n" + "="*80)
    print("🔧 Step 4: Preparing feature matrices...")
    print("="*80)
    # Target mode:
    # - 'price'  -> predict price[t+1] level (default, easier but news signal often weak)
    # - 'return' -> predict return[t+1] = price[t+1]/price[t]-1 (usually better for news signal)
    # - 'shock'  -> detect price shocks (binary classification of extreme movements)
    target_mode = 'shock'  # Options: 'return', 'price', 'shock'
    
    if target_mode == 'shock':
        # Check if shock labels exist
        if 'price_shock' not in df.columns:
            print("⚠️  Shock labels not found. Re-running create_price_features to generate them...")
            # Re-run create_price_features to ensure shock labels are created with the latest definition
            # (multi-day cumulative returns exceeding threshold)
            df = create_price_features(df, shock_window=shock_window, k_sigma=k_sigma, require_same_direction=require_same_direction)
            if 'price_shock' not in df.columns:
                raise ValueError("Cannot create shock labels: create_price_features failed")
        
        target_column = 'price_shock'
        print(f"🎯 Target: Shock Detection (binary classification)")
        print(f"   Shock definition: Cumulative return over {shock_window} days > {k_sigma}σ, {'same direction' if require_same_direction else 'any direction'}")
        print(f"   Shock rate: {df['price_shock'].mean():.1%}")
    else:
        target_column = 'target_return' if target_mode == 'return' else 'target_price'
        print(f"🎯 Target: {target_mode.upper()} prediction")

    X_price, X_news, X_hybrid, y, feature_names = prepare_features(df, target_column=target_column)
    print(f"✅ Price features: {X_price.shape[1]} features")
    print(f"✅ News features: {X_news.shape[1]} features")
    print(f"✅ Hybrid features: {X_hybrid.shape[1]} features")
    print(f"✅ Target samples: {len(y)}")
    
    # Step 5: Train models
    print("\n" + "="*80)
    print("🎯 Step 5: Training models...")
    print("="*80)
    
    # Use walk-forward validation (set use_walk_forward=True) or single split
    use_walk_forward = True  # Set to False for single split
    n_windows = 5  # Number of walk-forward windows
    model_type = 'gbr'  # Options: 'gbr', 'xgb', 'lgb'
    tune_hyperparams = False  # Set to True for hyperparameter tuning (slower)
    run_shap = True  # Set True only if you installed shap (optional)
    select_topk_news_features = True  # Select top-K news features per window (TRAIN-only)
    topk_news_features = 15  # Very strict selection - only the strongest news signals
    
    if target_mode == 'shock':
        # Shock detection: binary classification
        print("🔍 Running SHOCK DETECTION mode (binary classification)")
        print(f"   Target distribution: {np.bincount(y.astype(int))}")
        from sklearn.model_selection import train_test_split
        # Use time-based split instead of random to preserve chronological order
        # This is more realistic for time-series forecasting
        split_idx = int(len(y) * 0.8)
        X_price_train, X_price_test = X_price[:split_idx], X_price[split_idx:]
        X_news_train, X_news_test = X_news[:split_idx], X_news[split_idx:]
        X_hybrid_train, X_hybrid_test = X_hybrid[:split_idx], X_hybrid[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Also store indices for visualization
        train_indices = np.arange(split_idx)
        test_indices = np.arange(split_idx, len(y))
        
        print(f"   Train: {len(y_train)} samples ({y_train.sum()} shocks, {y_train.mean():.1%})")
        print(f"   Test: {len(y_test)} samples ({y_test.sum()} shocks, {y_test.mean():.1%})")
        results = train_shock_detection_model(
            X_price_train, X_news_train, X_hybrid_train, y_train,
            X_price_test, X_news_test, X_hybrid_test, y_test,
            model_type=model_type, random_state=42,
            calibrate_probabilities=True,  # Enable probability calibration
            tune_threshold=True,  # Enable threshold tuning for better F1/precision/recall
            val_fraction=0.2,  # Use 20% of training data for validation (calibration + threshold tuning)
            select_topk_news_features=select_topk_news_features,  # Select top-K news features
            topk_news_features=topk_news_features  # Number of top news features to keep
        )
        # Store test data and indices for later use in visualizations
        results['X_hybrid_test'] = X_hybrid_test
        results['test_indices'] = test_indices  # Store indices for proper date mapping
    else:
        # Regression: return/price prediction
        results = train_models(
            X_price, X_news, X_hybrid, y,
            test_size=0.2,
            random_state=42,
            use_walk_forward=use_walk_forward,
            n_windows=n_windows,
            model_type=model_type,
            tune_hyperparams=tune_hyperparams,
            save_models=True,
            save_dir='artifacts',
            feature_names=feature_names,
            select_topk_news_features=select_topk_news_features,
            topk_news_features=topk_news_features,
        )
    print("✅ All models trained")
    
    # Step 6: Evaluate and compare
    print("\n" + "="*80)
    print("📊 Step 6: Evaluating models...")
    print("="*80)
    
    if target_mode == 'shock':
        # Print formatted metrics table
        print_shock_detection_metrics(results, verbose=True)
        
        # Create visualizations
        output_dir = Path('artifacts')
        output_dir.mkdir(exist_ok=True)
        figures_dir = Path('figures')
        figures_dir.mkdir(exist_ok=True)
        
        # Main shock detection results plot (saved to both artifacts and figures)
        plot_shock_detection_results(results, save_dir='artifacts')
        
        # Price with shocks visualization
        plot_price_with_shocks(df, results, save_dir='figures')
        
        # Top news events visualization (reduced to 12 to avoid overlap)
        plot_top_news_events(df, results, save_dir='figures', top_n=12)
        
        # Comprehensive news statistics
        print("\n" + "="*80)
        print("📰 Generating comprehensive news statistics...")
        print("="*80)
        plot_news_statistics(save_dir='figures')
        
        # Feature importance plot (replaces SHAP) - use best hybrid model
        if run_shap:
            print("\n" + "="*80)
            print("🔍 Generating feature importance plot for shock detection...")
            print("="*80)
            # Use best hybrid model if available, otherwise fallback to default
            if 'all_models' in results and results['all_models']:
                # Find best hybrid model by AUC
                best_auc = 0
                best_model_key = None
                for key, metrics in results['all_metrics'].items():
                    if 'hybrid' in key and metrics['auc'] > best_auc:
                        best_auc = metrics['auc']
                        best_model_key = key
                
                if best_model_key and best_model_key in results['all_models']:
                    best_model = results['all_models'][best_model_key]
                    print(f"   Using {best_model_key} (AUC={best_auc:.3f}) for feature importance")
                    # Use stored feature names if available, otherwise fallback
                    importance_feature_names = results.get('feature_names_hybrid', feature_names.get('hybrid', None))
                    if importance_feature_names is None:
                        # Try to get from original feature names
                        importance_feature_names = feature_names.get('hybrid', None)
                    if importance_feature_names is None:
                        print("   ⚠️  Feature names not available, using default names")
                        importance_feature_names = None
                    plot_feature_importance_shock(
                        best_model,
                        importance_feature_names,
                        top_n=20,
                        save_dir='artifacts/shap'  # Will be saved to both artifacts and figures
                    )
                else:
                    print("⚠️  Best hybrid model not found for feature importance")
            else:
                # Fallback to default model
                if 'model_hybrid' in results:
                    importance_feature_names = results.get('feature_names_hybrid', feature_names.get('hybrid', None))
                    plot_feature_importance_shock(
                        results['model_hybrid'],
                        importance_feature_names,
                        top_n=20,
                        save_dir='artifacts/shap'  # Will be saved to both artifacts and figures
                    )
                else:
                    print("⚠️  Hybrid model not available for feature importance")
        else:
            print("\nℹ️  Feature importance disabled (set run_shap=True to enable)")
        
        print("\n✅ Shock detection evaluation complete")
        return results, None, None, None
    elif use_walk_forward:
        aggregated, window_metrics = compare_models_walk_forward(results)
        # Use aggregated metrics for winner determination
        metrics_arima = {
            'RMSE': aggregated['arima']['RMSE']['median'],
            'MAE': aggregated['arima']['MAE']['median'],
            'R²': aggregated['arima']['R²']['median'],
            'Directional Accuracy': aggregated['arima']['Directional Accuracy']['median']
        }
        metrics_arimax = None
        if 'arimax' in aggregated:
            metrics_arimax = {
                'RMSE': aggregated['arimax']['RMSE']['median'],
                'MAE': aggregated['arimax']['MAE']['median'],
                'R²': aggregated['arimax']['R²']['median'],
                'Directional Accuracy': aggregated['arimax']['Directional Accuracy']['median']
            }
        metrics_price = {
            'RMSE': aggregated['price']['RMSE']['median'],
            'MAE': aggregated['price']['MAE']['median'],
            'R²': aggregated['price']['R²']['median'],
            'Directional Accuracy': aggregated['price']['Directional Accuracy']['median']
        }
        metrics_hybrid = {
            'RMSE': aggregated['hybrid']['RMSE']['median'],
            'MAE': aggregated['hybrid']['MAE']['median'],
            'R²': aggregated['hybrid']['R²']['median'],
            'Directional Accuracy': aggregated['hybrid']['Directional Accuracy']['median']
        }
    else:
        metrics_arima, metrics_price, metrics_hybrid, metrics_arimax = compare_models(results)
    
    # Step 7: Visualizations
    print("\n" + "="*80)
    print("📈 Step 7: Creating visualizations...")
    print("="*80)
    
    # Create output directory
    output_dir = Path('artifacts')
    output_dir.mkdir(exist_ok=True)
    
    # Plot forecasts
    plot_forecasts(results, save_path=output_dir / 'forecasts.png')
    
    # Plot feature importance (use last window or single split)
    if use_walk_forward:
        last_model = results['models_hybrid'][-1]
        last_window_test_start = results['window_info'][-1]['test_start']
        last_window_test_end = results['window_info'][-1]['test_end']
        X_hybrid_test = X_hybrid[last_window_test_start:last_window_test_end]
    else:
        last_model = results['model_hybrid']
        split_idx = results['split_idx']
        X_hybrid_test = X_hybrid[split_idx:]
    
    plot_feature_importance(
        last_model, 
        feature_names['hybrid'], 
        top_n=20,
        save_path=output_dir / 'feature_importance.png'
    )
    
    # Correlation heatmap
    plot_correlation_heatmap(
        df,
        feature_names['hybrid'][:50],  # Limit to 50 features
        save_path=output_dir / 'correlation_heatmap.png'
    )
    
    # Summary table
    summary_df = create_summary_table(results, save_path=output_dir / 'summary_metrics.csv')
    print(f"\n✅ Summary table saved")
    
    # Step 8: SHAP explanations (optional, can be slow)
    print("\n" + "="*80)
    print("🔍 Step 8: Generating SHAP explanations...")
    print("="*80)
    if run_shap:
        if use_walk_forward:
            explain_with_shap(
                last_model,
                X_hybrid_test,
                feature_names['hybrid'],
                top_n=20,
                save_dir=str(output_dir / 'shap')
            )
        else:
            explain_with_shap(
                results['model_hybrid'],
                X_hybrid_test,
                feature_names['hybrid'],
                top_n=20,
                save_dir=str(output_dir / 'shap')
            )
    else:
        print("ℹ️  SHAP disabled (set run_shap=True to enable; requires `pip install shap`).")
    
    # Analyze prediction errors
    if not use_walk_forward:
        analyze_prediction_errors(results, feature_names, X_hybrid_test)
    
    # Conclusion
    print("\n" + "="*80)
    print("✅ Pipeline completed successfully!")
    print("="*80)
    
    # Determine overall winner (by RMSE)
    all_rmse = {
        'ARIMA': metrics_arima['RMSE'],
        **({'ARIMAX': metrics_arimax['RMSE']} if 'metrics_arimax' in locals() and metrics_arimax is not None else {}),
        'ML Baseline': metrics_price['RMSE'],
        'Hybrid': metrics_hybrid['RMSE']
    }
    winner = min(all_rmse, key=all_rmse.get)
    winner_rmse = all_rmse[winner]
    
    print(f"\n🏆 Overall Winner (by RMSE): {winner}")
    print(f"   RMSE: {winner_rmse:.4f}")
    
    if winner == 'Hybrid':
        improvement_vs_ml = (metrics_price['RMSE'] - metrics_hybrid['RMSE']) / metrics_price['RMSE'] * 100
        improvement_vs_arima = (metrics_arima['RMSE'] - metrics_hybrid['RMSE']) / metrics_arima['RMSE'] * 100
        print(f"   Improvement vs ML Baseline: {improvement_vs_ml:.2f}%")
        print(f"   Improvement vs ARIMA: {improvement_vs_arima:.2f}%")
    
    return results, metrics_arima, metrics_price, metrics_hybrid


if __name__ == "__main__":
    results, metrics_arima, metrics_price, metrics_hybrid = main()
