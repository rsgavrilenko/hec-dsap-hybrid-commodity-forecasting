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
    stock_column: Optional[str] = 'lme_copper_stock',
        shock_window: int = 3,
        k_sigma: float = 1.5,  # Reduced from 2.0 to get more balanced classes (~5-10% shocks)
        require_same_direction: bool = True  # If False, allows mixed-direction cumulative moves
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
    - Shock labels (price_shock): Binary indicator for extreme price movements
    
    Args:
        df: DataFrame with 'date' and price_column columns.
            Should be sorted by date.
        price_column: Name of the price column. Default: 'price'
        stock_column: Name of the stock column. If None, stock features
                     are not created. Default: 'lme_copper_stock'
        shock_window: Number of days to consider for cumulative return in shock detection.
                     Default: 3
        k_sigma: Threshold in standard deviations for shock detection.
                 Default: 1.5 (reduced from 2.0 for better class balance, ~6-7% shocks)
                 Lower values = more shocks (less strict), higher values = fewer shocks (more strict)
        require_same_direction: If True, all days in window must have returns in same direction.
                               If False, allows any extreme cumulative move regardless of direction.
                               Default: True
    
    Returns:
        DataFrame with original columns plus new feature columns.
        The last row is removed (no target available for it).
    """
    df = df.copy()
    
    # Sort by date to ensure proper lag calculation
    df = df.sort_values('date').reset_index(drop=True)
    
    # Targets:
    # - next day's price level (price[t+1])
    # - next day's return from t to t+1 (return[t+1]) = price[t+1] / price[t] - 1
    df['target_price'] = df[price_column].shift(-1)
    df['target_return'] = df[price_column].shift(-1) / df[price_column] - 1
    
    # Remove last row (no target available)
    df = df[:-1].copy()
    
    # Lagged prices (extended set) - fill NaN with forward fill (use last known price)
    first_price = df[price_column].iloc[0] if len(df) > 0 else 0
    df['price_lag1'] = df[price_column].shift(1).bfill().fillna(first_price)
    df['price_lag2'] = df[price_column].shift(2).bfill().fillna(first_price)
    df['price_lag3'] = df[price_column].shift(3).bfill().fillna(first_price)
    df['price_lag5'] = df[price_column].shift(5).bfill().fillna(first_price)
    df['price_lag7'] = df[price_column].shift(7).bfill().fillna(first_price)  # Weekly lag
    df['price_lag10'] = df[price_column].shift(10).bfill().fillna(first_price)  # Extended lag
    
    # Price differences (momentum indicators) - fill NaN with 0
    df['price_diff_1_2'] = (df['price_lag1'] - df['price_lag2']).fillna(0)  # 1-day momentum
    df['price_diff_1_5'] = (df['price_lag1'] - df['price_lag5']).fillna(0)  # 5-day momentum
    df['price_diff_5_10'] = (df['price_lag5'] - df['price_lag10']).fillna(0)  # Longer-term momentum
    
    # Returns (percentage change) - extended (fill NaN with 0 - no return)
    df['return_lag1'] = df[price_column].pct_change(1, fill_method=None).shift(1).fillna(0)
    df['return_lag2'] = df[price_column].pct_change(2, fill_method=None).shift(1).fillna(0)
    df['return_lag5'] = df[price_column].pct_change(5, fill_method=None).shift(1).fillna(0)
    df['return_lag7'] = df[price_column].pct_change(7, fill_method=None).shift(1).fillna(0)  # Weekly return
    
    # Moving averages (extended) - fill NaN with current price
    df['ma_5'] = df[price_column].rolling(window=5, min_periods=1).mean().shift(1).fillna(df[price_column])
    df['ma_10'] = df[price_column].rolling(window=10, min_periods=1).mean().shift(1).fillna(df[price_column])
    df['ma_20'] = df[price_column].rolling(window=20, min_periods=1).mean().shift(1).fillna(df[price_column])
    df['ma_50'] = df[price_column].rolling(window=50, min_periods=1).mean().shift(1).fillna(df[price_column])  # Longer-term trend
    
    # Price relative to moving averages (fill NaN with 0 - no deviation)
    df['price_to_ma5'] = (df[price_column] / (df['ma_5'] + 1e-10) - 1).fillna(0)
    df['price_to_ma10'] = (df[price_column] / (df['ma_10'] + 1e-10) - 1).fillna(0)
    df['price_to_ma20'] = (df[price_column] / (df['ma_20'] + 1e-10) - 1).fillna(0)
    df['price_to_ma50'] = (df[price_column] / (df['ma_50'] + 1e-10) - 1).fillna(0)
    
    # MA crossovers (trend indicators) - fill NaN with 0
    df['ma5_ma10_cross'] = ((df['ma_5'] - df['ma_10']) / (df['ma_10'] + 1e-10)).fillna(0)  # Short-term vs medium-term
    df['ma10_ma20_cross'] = ((df['ma_10'] - df['ma_20']) / (df['ma_20'] + 1e-10)).fillna(0)  # Medium-term vs long-term
    
    # Volatility (rolling standard deviation of returns) - extended (fill NaN with 0)
    returns = df[price_column].pct_change(fill_method=None)
    df['volatility_5'] = returns.rolling(window=5, min_periods=1).std().shift(1).fillna(0)
    df['volatility_10'] = returns.rolling(window=10, min_periods=1).std().shift(1).fillna(0)
    df['volatility_20'] = returns.rolling(window=20, min_periods=1).std().shift(1).fillna(0)  # Longer-term volatility
    
    # RSI (Relative Strength Index) - momentum indicator
    delta = returns
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean().shift(1)
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean().shift(1)
    rs = gain / (loss + 1e-10)  # Avoid division by zero
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50.0)  # Fill NaN with neutral RSI (50)
    
    # Bollinger Bands
    df['bb_middle'] = df['ma_20']  # Middle band (MA)
    bb_std = returns.rolling(window=20, min_periods=1).std().shift(1).fillna(0)
    df['bb_upper'] = df['bb_middle'] + (2 * bb_std * df[price_column])
    df['bb_lower'] = df['bb_middle'] - (2 * bb_std * df[price_column])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)  # Band width (volatility)
    df['bb_width'] = df['bb_width'].fillna(0)  # Fill NaN with 0
    df['bb_position'] = (df[price_column] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)  # Position in band
    df['bb_position'] = df['bb_position'].fillna(0.5)  # Fill NaN with middle position
    
    # Momentum indicators (fill NaN with 0 - no momentum)
    df['momentum_5'] = (df[price_column] / df[price_column].shift(5) - 1).fillna(0)  # 5-day momentum
    df['momentum_10'] = (df[price_column] / df[price_column].shift(10) - 1).fillna(0)  # 10-day momentum
    
    # Rate of change (fill NaN with 0 - no change)
    df['roc_5'] = ((df[price_column] - df[price_column].shift(5)) / (df[price_column].shift(5) + 1e-10)).fillna(0)  # 5-day ROC
    df['roc_10'] = ((df[price_column] - df[price_column].shift(10)) / (df[price_column].shift(10) + 1e-10)).fillna(0)  # 10-day ROC
    
    # Absolute return (for spike detection)
    df['abs_return'] = returns.abs()
    
    # Shock detection label (extreme price movements over multiple days)
    # A "shock" is defined as a cumulative return over N days that exceeds k standard deviations
    # This captures sustained price movements rather than single-day spikes
    if 'target_return' in df.columns:
        returns = df['target_return'].dropna()
        if len(returns) > 0:
            # Parameters for multi-day shock detection (now configurable)
            # Default: shock_window=3, k_sigma=1.5 (reduced from 2.0 for better class balance)
            
            # Calculate cumulative returns over the shock window
            cumulative_returns = df['target_return'].rolling(window=shock_window, min_periods=1).sum()
            
            # Calculate statistics for cumulative returns
            cum_returns_clean = cumulative_returns.dropna()
            if len(cum_returns_clean) > 0:
                mean_cum_return = cum_returns_clean.mean()
                std_cum_return = cum_returns_clean.std()
                
                # Shock = cumulative return exceeds k*std threshold
                # Also check that all days in window have returns in the same direction
                df['price_shock'] = 0
                
                for i in range(len(df)):
                    if i < shock_window - 1:
                        continue  # Skip first few rows where cumulative return is not meaningful
                    
                    # Get returns in the window (last N days including current)
                    window_start = max(0, i - shock_window + 1)
                    window_returns = df['target_return'].iloc[window_start:i+1]
                    
                    # Check if cumulative return exceeds threshold
                    cum_ret = cumulative_returns.iloc[i]
                    if pd.isna(cum_ret):
                        continue
                    exceeds_threshold = abs(cum_ret) > (k_sigma * std_cum_return)
                    
                    # Check if all returns in window are in the same direction (sustained movement)
                    window_returns_clean = window_returns.dropna()
                    if len(window_returns_clean) < shock_window:
                        continue  # Skip if not enough valid returns
                    
                    # Check direction requirement (if enabled)
                    if require_same_direction:
                        all_positive = (window_returns_clean > 0).all()
                        all_negative = (window_returns_clean < 0).all()
                        same_direction = all_positive or all_negative
                    else:
                        same_direction = True  # Skip direction check if disabled
                    
                    # Mark as shock if threshold exceeded and (if required) same direction
                    if exceeds_threshold and same_direction:
                        df.loc[df.index[i], 'price_shock'] = 1
            else:
                df['price_shock'] = 0
        else:
            df['price_shock'] = 0
    
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

