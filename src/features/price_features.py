"""
Module for creating price and stock-based features.

This module provides functions to create technical indicators and features
from price and stock data, including:
- Lagged prices and returns
- Moving averages
- Volatility indicators
- Stock level features
"""

import pandas as pd
from typing import Optional


def create_price_features(
    df: pd.DataFrame,
    price_column: str = 'price',
    stock_column: Optional[str] = 'lme_copper_stock'
) -> pd.DataFrame:
    """
    Create price-based features from a DataFrame with price data.
    
    Features created:
    - Lagged prices (1, 2, 3, 5 days)
    - Returns (percentage change) with lags
    - Moving averages (5, 10, 20 days)
    - Price relative to moving averages
    - Volatility (rolling standard deviation)
    - Stock features (if stock_column provided):
      - Lagged stock levels
      - Stock changes
      - Stock moving averages
      - Stock relative to moving averages
    
    Args:
        df: DataFrame with 'date' and price_column columns.
            Should be sorted by date.
        price_column: Name of the price column. Default: 'price'
        stock_column: Name of the stock column. If None, stock features
                     are not created. Default: 'lme_copper_stock'
    
    Returns:
        DataFrame with original columns plus new feature columns.
        The last row is removed (no target available for it).
    """
    df = df.copy()
    
    # Sort by date to ensure proper lag calculation
    df = df.sort_values('date').reset_index(drop=True)
    
    # Target: next day's price (price[t+1])
    df['target_price'] = df[price_column].shift(-1)
    
    # Remove last row (no target available)
    df = df[:-1].copy()
    
    # Lagged prices
    df['price_lag1'] = df[price_column].shift(1)
    df['price_lag2'] = df[price_column].shift(2)
    df['price_lag3'] = df[price_column].shift(3)
    df['price_lag5'] = df[price_column].shift(5)
    
    # Returns (percentage change)
    df['return_lag1'] = df[price_column].pct_change(1, fill_method=None).shift(1)
    df['return_lag2'] = df[price_column].pct_change(2, fill_method=None).shift(1)
    df['return_lag5'] = df[price_column].pct_change(5, fill_method=None).shift(1)
    
    # Moving averages
    df['ma_5'] = df[price_column].rolling(window=5, min_periods=1).mean().shift(1)
    df['ma_10'] = df[price_column].rolling(window=10, min_periods=1).mean().shift(1)
    df['ma_20'] = df[price_column].rolling(window=20, min_periods=1).mean().shift(1)
    
    # Price relative to moving averages
    df['price_to_ma5'] = df[price_column] / df['ma_5'] - 1
    df['price_to_ma10'] = df[price_column] / df['ma_10'] - 1
    
    # Volatility (rolling standard deviation of returns)
    df['volatility_5'] = df[price_column].pct_change(fill_method=None).rolling(window=5, min_periods=1).std().shift(1)
    df['volatility_10'] = df[price_column].pct_change(fill_method=None).rolling(window=10, min_periods=1).std().shift(1)
    
    # Absolute return (for spike detection)
    df['abs_return'] = df[price_column].pct_change(fill_method=None).abs()
    
    # Stock features (LME stock levels)
    if stock_column and stock_column in df.columns:
        # Lagged stock
        df['lme_copper_stock_lag1'] = df[stock_column].shift(1)
        df['lme_copper_stock_lag2'] = df[stock_column].shift(2)
        df['lme_copper_stock_lag5'] = df[stock_column].shift(5)
        
        # Stock changes (percentage change)
        df['lme_copper_stock_change_lag1'] = df[stock_column].pct_change(1, fill_method=None).shift(1)
        df['lme_copper_stock_change_lag5'] = df[stock_column].pct_change(5, fill_method=None).shift(1)
        
        # Moving averages of stock
        df['lme_copper_stock_ma_5'] = df[stock_column].rolling(window=5, min_periods=1).mean().shift(1)
        df['lme_copper_stock_ma_10'] = df[stock_column].rolling(window=10, min_periods=1).mean().shift(1)
        
        # Stock relative to moving average
        df['lme_copper_stock_to_ma5'] = df[stock_column] / df['lme_copper_stock_ma_5'] - 1
        df['lme_copper_stock_to_ma10'] = df[stock_column] / df['lme_copper_stock_ma_10'] - 1
    
    return df

