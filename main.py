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
from src.features.sentiment_features import create_news_features
from src.models import prepare_features, train_models
from src.evaluation import compare_models


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
    df = align_price_and_news()
    print(f"✅ Loaded {len(df)} records")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Step 2: Create price features
    print("\n" + "="*80)
    print("🔧 Step 2: Creating price features...")
    print("="*80)
    df = create_price_features(df)
    print(f"✅ Created price and stock features")
    
    # Step 3: Create news features
    print("\n" + "="*80)
    print("🔧 Step 3: Creating news features...")
    print("="*80)
    df = create_news_features(df, use_finbert=True, verbose=True)
    print(f"✅ Created news features (FinBERT/TF-IDF embeddings + heuristics)")
    
    # Step 4: Prepare feature matrices
    print("\n" + "="*80)
    print("🔧 Step 4: Preparing feature matrices...")
    print("="*80)
    X_price, X_news, X_hybrid, y, feature_names = prepare_features(df)
    print(f"✅ Price features: {X_price.shape[1]} features")
    print(f"✅ News features: {X_news.shape[1]} features")
    print(f"✅ Hybrid features: {X_hybrid.shape[1]} features")
    print(f"✅ Target samples: {len(y)}")
    
    # Step 5: Train models
    print("\n" + "="*80)
    print("🎯 Step 5: Training models...")
    print("="*80)
    results = train_models(X_price, X_news, X_hybrid, y, test_size=0.2, random_state=42)
    print("✅ All models trained")
    
    # Step 6: Evaluate and compare
    print("\n" + "="*80)
    print("📊 Step 6: Evaluating models...")
    print("="*80)
    metrics_arima, metrics_price, metrics_hybrid = compare_models(results)
    
    # Conclusion
    print("\n" + "="*80)
    print("✅ Pipeline completed successfully!")
    print("="*80)
    
    # Determine overall winner (by RMSE)
    all_rmse = {
        'ARIMA': metrics_arima['RMSE'],
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
