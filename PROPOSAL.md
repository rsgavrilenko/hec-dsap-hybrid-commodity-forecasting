# **Forecasting Commodity Price Movements with News and Time Series Data**

**Category:** Data Analysis & Machine Learning  

---

## **Problem Statement / Motivation**

Commodity trading is one of the largest global industries — covering the movement of oil, gas, metals, and agricultural goods around the world. Switzerland plays an especially important role in this sector, hosting many major trading companies.

Yet, the industry still hasn’t fully embraced data-driven decision-making. **Traders often rely on their intuition** and experience to react to new information — for example, when a copper mine shuts down, or new sanctions affect oil exports. These news events can move prices dramatically, and traders must quickly decide what to do next.

The idea of this project is to explore whether a simple model can **predict the next-time price movement** (`price[t+1]`) based on two sources of information:

- recent price history, and  
- what kind of news is happening at that moment.

In other words, can we teach a model to “sense” how news and price data interact — similar to how a junior trader learns to interpret market signals?

---

## **Approach & Technologies**

The project combines basic **time-series forecasting** and **news sentiment analysis**.

### **Data Collection**
- Historical price data for commodities such as oil, gas, or copper (from Yahoo Finance or Quandl).  
- News headlines from free online sources (Investing.com RSS, NewsAPI, or Reuters).  

### **Data Preparation**
- Align price and news data by date, being careful to **avoid lookahead bias** (e.g. news published late in the evening should not influence same-day prices).  
- Convert news headlines into a **financial sentiment score** using lightweight open-source models such as **FLANG** ([FinLLMs project](https://github.com/adlnlp/FinLLMs)) or **FinBERT-20**, which are specifically trained on financial text.  
- Validate sentiment quality separately — check whether positive headlines are generally followed by upward price movements (and vice versa).  
- Create features from price data — e.g. moving averages, percentage changes, and volatility indicators.  

### **Modeling**
- Use three explicit model types for comparison:  
  1. **Time-series baseline** (e.g. ARIMA or Prophet) using price-only data.  
  2. **Machine learning baseline** (e.g. XGBoost) trained only on price-based features.  
  3. **Hybrid model** combining both numerical and sentiment-derived features in a unified feature matrix.  

  The hybrid model will integrate time-series indicators (returns, volatility, lagged features) together with aggregated sentiment scores over rolling windows.  
- Evaluate how much the inclusion of sentiment improves the forecast compared to using price data alone.  

### **Evaluation & Visualization**
- **Temporal validation** using real chronological splits — e.g. training on 2020–2022 and testing on 2023–2024, or a walk-forward validation scheme.  
- Metrics: **RMSE** (forecast accuracy) and **directional accuracy** (whether the model predicts the right trend).  
- Visualizations: forecast plots, feature importance graphs, and correlation between sentiment and price movements to interpret model behavior.  

### **Tools**
- Python, pandas, NumPy, scikit-learn, matplotlib, seaborn.  
- Jupyter Notebooks for analysis and modular `.py` files for reproducible code.  

---

## **Expected Challenges & Mitigations**

| Challenge | Mitigation |
|------------|-------------|
| Aligning news and price data | Use clear date-based merging and filter unrelated news |
| Limited amount of clean news data | Focus on one or two well-covered commodities (oil, copper) |
| Overfitting with small data | Use simple models and cross-validation |
| Measuring impact of news | Validate sentiment independently and compare model performance with and without sentiment features |
| Local compute limits | Use lightweight financial NLP models such as **FLANG** or **FinBERT-20**, and run on Colab or Kaggle if needed |

---

## **Success Criteria**

- A reproducible pipeline to collect, clean, and merge commodity prices with daily news data.  
- A working model that shows **better forecast accuracy** when using both price and sentiment information.  
- Clear visual results demonstrating how news affects short-term price movements.  
- Well-structured and documented code suitable for a DSAP portfolio project.  

---

## **Stretch Goals**

- Try multiple commodities and compare their sensitivity to news.  
- Add an **explainability layer** (e.g. SHAP or feature importance plots).  
- Test how the model reacts to **hypothetical “shock” events**, similar to what traders face during interviews.  

---
