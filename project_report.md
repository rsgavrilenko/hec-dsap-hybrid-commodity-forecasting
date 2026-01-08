---
title: Hybrid Commodity Forecasting with News Data
author: 
date: 
---

## Abstract
Briefly summarize the research question, data, models, and the main findings (150–200 words).

## 1. Introduction
- **Research question**: Does incorporating news sentiment improve forecasting of copper price movements compared to using price data alone?
- Motivation and practical relevance.
- What you implement and evaluate in this repository.

## 2. Data
### 2.1 Price data
- Source: LME copper historical data (`src/data/copper/data_copper_lme_all_years.csv`)
- Date range and frequency.

### 2.2 News data
- Source: RSS/Google News scraping pipeline (`src/data/news/news_data.py`)
- Dataset: `src/data/news/copper_news_all_sources.csv`
- Deduplication, cleaning, and alignment to trading days.

## 3. Methodology
### 3.1 Feature engineering
- Price features: lags, moving averages, volatility, technical indicators.
- News features: FinBERT sentiment (if available locally), TF‑IDF fallback, heuristics, rolling aggregations, lags.
- Interaction features: price × news interactions.

### 3.2 Models
Describe the model families and why they’re used:
- Time-series baseline (ARIMA for regression mode).
- ML baselines (Logistic Regression / Random Forest / SVM / Gradient Boosting).
- Hybrid models (price + news + interactions).

### 3.3 Evaluation protocol
- Time-based splits to avoid leakage.
- Metrics:
  - Regression: RMSE / MAE / R² / Directional Accuracy
  - Shock detection: AUC / PR‑AUC / Precision / Recall / F1

## 4. Results
Include:
- One regression run (price or return) with a comparison table + at least one plot from `figures/`.
- Shock detection results with AUC/PR‑AUC/F1 and the key plots from `figures/`.

## 5. Discussion
- Why some models outperform others.
- When and why news features help (and when they add noise).
- Limitations (data coverage, noisy labels, model assumptions).

## 6. Conclusion & Future Work
- Summary of findings.
- Practical recommendations.
- Future improvements.

## How to generate PDF
If you use pandoc:

```bash
pandoc project_report.md -o project_report.pdf --pdf-engine=xelatex --toc --number-sections
```


