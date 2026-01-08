# Hybrid Commodity Forecasting with News Data

This project combines time-series forecasting with news sentiment analysis to predict commodity price movements.

## Setup Instructions

### Prerequisites
- Conda (Miniconda or Anaconda)
- Git

### Environment Setup

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd hec-dsap-hybrid-commodity-forecasting
   ```

2. **Create the conda environment** from `environment.yml`:
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment**:
   ```bash
   conda activate commodity-forecast
   ```

4. **Verify installation**:
   ```bash
   python -c "import sklearn, pandas, numpy, matplotlib, seaborn; print('✅ All packages installed successfully')"
   ```

### Running the Analysis

1. **Activate the environment** (if not already active):
   ```bash
   conda activate commodity-forecast
   ```

2. **Run the main pipeline**:
   ```bash
   python main.py
   ```

   This will:
   - Load and align price and news data
   - Create price-based features (moving averages, returns, volatility)
   - Create news-based features (FinBERT sentiment scores if available, otherwise TF-IDF embeddings + keyword heuristics)
   - Train baseline (price-only) and hybrid (price + news) models with walk-forward validation
   - Compare model performance and show results

3. **Expected output**:
   - Model comparison table with RMSE, MAE, R², and Directional Accuracy (or AUC, F1 for shock detection)
   - Winner model identification
   - Performance improvement metrics
   - Saved artifacts in `artifacts/` (models, predictions, detailed plots)
   - Saved figures in `figures/` (visualizations for reports: shock detection results, price with shocks, top news events)

### Target Definition (Important)

The pipeline supports three prediction modes:

1. **Return Prediction** (default): Predicts next-day return
   - `target_return = price[t+1]/price[t] - 1`
   - Generally works better for capturing news-driven price movements

2. **Price Prediction**: Predicts next-day price level
   - `target_price = price[t+1]`

3. **Shock Detection**: Binary classification of extreme price movements
   - Detects price "shocks" (returns exceeding 2 standard deviations)
   - Useful for early warning of rare but impactful price spikes
   - Combines price trends with news sentiment for improved detection

You can switch the target mode in `main.py`:
- `target_mode = 'return'` (default) - regression
- `target_mode = 'price'` - regression  
- `target_mode = 'shock'` - binary classification

### Project Structure

```
.
├── main.py                      # Main entry point - run this!
├── environment.yml              # Conda environment definition
├── README.md                    # This file
├── PROPOSAL.md                  # Project proposal
├── artifacts/                   # Generated outputs (models, predictions, plots)
│   └── shap/                    # SHAP explanations
├── figures/                     # Visualization outputs for reports
│   └── shock_detection_results.png
│   └── price_with_shocks.png
│   └── top_news_events.png
├── src/
│   ├── data/
│   │   ├── data_loader.py       # Unified data loading functions
│   │   ├── copper/
│   │   │   └── data_copper_lme_all_years.csv  # Copper price data
│   │   └── news/
│   │       ├── news_data.py     # News data collection
│   │       └── copper_news_all_sources.csv  # Collected news data
│   ├── features/
│   │   ├── price_features.py     # Price and stock feature creation
│   │   └── sentiment_features.py # News sentiment and heuristic features
│   ├── models.py                # Model training functions
│   └── evaluation.py            # Model evaluation and comparison
```

### Dependencies

All dependencies are specified in `environment.yml`:
- Python 3.11
- numpy, pandas
- scikit-learn (for machine learning models)
- matplotlib, seaborn (for visualization)
- jupyter (for notebooks)
- statsmodels (ARIMA baseline)
- optional: xgboost, lightgbm (if you set `model_type='xgb'/'lgb'`)
- optional: shap (if you set `run_shap=True`)

### Troubleshooting
**Issue**: `FinBERT sentiment failed` / TF-IDF fallback
- **Reason**: `transformers` and/or `torch` not installed in your environment.
- **Fix**:
  - `pip install transformers torch`
  - or update the conda env: `conda env update -f environment.yml --prune`

**Issue**: XGBoost/LightGBM not available
- **Fix**: Install optional deps:
  - `conda install -c conda-forge xgboost lightgbm`

**Issue**: SHAP not available
- **Fix**:
  - `pip install shap`
  - then set `run_shap=True` in `main.py`


**Issue**: `ModuleNotFoundError: No module named 'sklearn'`
- **Solution**: Make sure you've activated the conda environment: `conda activate commodity-forecast`
- If the environment doesn't exist, create it: `conda env create -f environment.yml`

**Issue**: Jupyter notebook uses wrong kernel
- **Solution**: In Jupyter, go to Kernel → Change Kernel → Select "Python 3 (commodity-forecast)" or "commodity-forecast"

**Issue**: Environment creation fails
- **Solution**: Make sure you have conda installed and updated: `conda update conda`

