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

2. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

3. **Open the analysis notebook**:
   - Navigate to `src/data/plain_analysis.ipynb`
   - Make sure the kernel is set to `Python 3 (commodity-forecast)` or `commodity-forecast`
   - Run all cells

### Project Structure

```
.
├── environment.yml              # Conda environment definition
├── README.md                    # This file
├── src/
│   └── data/
│       ├── data_loader.py       # Unified data loading functions
│       ├── plain_analysis.ipynb # Main analysis notebook
│       ├── copper/
│       │   └── copper_data_parsing.py  # Copper price data parser
│       └── news/
│           ├── news_data.py     # News data collection
│           └── copper_news_all_sources.csv  # Collected news data
```

### Dependencies

All dependencies are specified in `environment.yml`:
- Python 3.11
- numpy, pandas
- scikit-learn (for machine learning models)
- matplotlib, seaborn (for visualization)
- jupyter (for notebooks)
- yfinance (for financial data, via pip)

### Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'sklearn'`
- **Solution**: Make sure you've activated the conda environment: `conda activate commodity-forecast`
- If the environment doesn't exist, create it: `conda env create -f environment.yml`

**Issue**: Jupyter notebook uses wrong kernel
- **Solution**: In Jupyter, go to Kernel → Change Kernel → Select "Python 3 (commodity-forecast)" or "commodity-forecast"

**Issue**: Environment creation fails
- **Solution**: Make sure you have conda installed and updated: `conda update conda`

