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
   - All outputs saved to `results/`:
     - Plots and visualizations in `results/figures/`
     - Model files (if enabled) in `results/models/`

## Report and Presentation (PDF)
The project report and presentation are written in LaTeX for professional formatting. To compile to PDF:

```bash
# Navigate to the reports directory
cd reports/reports

# Compile the report (run twice for references/TOC)
pdflatex report.tex
pdflatex report.tex

# Compile the presentation
pdflatex presentation.tex
pdflatex presentation.tex

# Clean up auxiliary files
rm -f *.aux *.log *.out *.toc *.nav *.snm *.vrb
```

**Requirements**: `pdflatex` (install via `brew install basictex` on Mac, or `apt-get install texlive-latex-base` on Linux)

The compiled PDFs should be:
- `report.pdf`: ~8-9 pages with all figures, tables, and detailed methodology/results sections
- `presentation.pdf`: ~20 slides for project presentation

## Figures for the report
All visualizations are saved in `results/figures/`:
- Shock detection plots: `shock_detection_results.png`, `price_with_shocks.png`, `top_news_events_table.png`, `news_statistics.png`
- Regression plots (when `target_mode` is `price` or `return`):
  - `forecasts_price.png`, `summary_metrics_price.png` (and `.csv`)
  - `forecasts_return.png`, `summary_metrics_return.png` (and `.csv`)

### Target Definition (Important)

The pipeline supports three prediction modes:

1. **Shock Detection** (default): Binary classification of extreme price movements
   - Detects price "shocks" (multi-day cumulative returns exceeding statistical threshold)
   - Useful for early warning of rare but impactful price movements
   - Combines price trends with news sentiment for improved detection
   - Focus of the final project implementation

2. **Return Prediction**: Predicts next-day return
   - `target_return = price[t+1]/price[t] - 1`
   - Generally works better for capturing news-driven price movements
   - Attempted but showed poor quality, leading to focus on shock detection

3. **Price Prediction**: Predicts next-day price level
   - `target_price = price[t+1]`
   - Attempted but showed poor quality, leading to focus on shock detection

You can switch the target mode in `main.py`:
- `target_mode = 'shock'` (default) - binary classification
- `target_mode = 'return'` - regression
- `target_mode = 'price'` - regression

### Project Structure

```
.
├── main.py                      # Main entry point - run this!
├── environment.yml              # Conda environment definition
├── README.md                    # This file
├── PROPOSAL.md                  # Project proposal
├── ai_log.md                    # AI usage log
├── data/
│   ├── raw/                      # Raw datasets used by main.py
│   │   ├── copper/
│   │   │   └── data_copper_lme_all_years.csv
│   │   └── news/
│   │       └── copper_news_all_sources.csv
│   ├── news/
│   │   └── news_data.py          # News collection utility (writes to data/raw/news)
│   └── copper/
│       ├── copper_data_parsing.py        # Price collection utility (optional)
│       ├── copper_visualization.py       # Price/stock plot utility (optional)
│       └── copper_price_stock_timeseries.png
├── results/                     # Outputs created by running main.py
│   ├── figures/                 # All plots and visualizations
│   └── models/                  # Saved model files (if save_models=True)
├── reports/                     # LaTeX sources and compiled PDFs
│   └── reports/
│       ├── report.tex           # Main project report (IEEE format)
│       ├── presentation.tex     # Presentation slides (Beamer)
│       ├── IEEEtran.cls         # IEEE template class file
│       ├── report.pdf           # Compiled report (generated)
│       └── presentation.pdf     # Compiled presentation (generated)
├── src/
│   ├── data_loader.py           # Data loading / alignment
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

