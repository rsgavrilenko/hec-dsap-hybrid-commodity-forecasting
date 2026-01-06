"""
Module for loading and aligning price and news data.

This module provides functions to:
- Load copper price data from CSV files
- Load news data from CSV files
- Align price and news data while avoiding lookahead bias
- Prepare combined datasets for modeling

Lookahead bias occurs when we use information from the future to predict
the past. For example, news published late in the evening should not
affect the closing price of the same day.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import time, timedelta


def load_copper_prices(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load copper price data from CSV file.

    The CSV should contain columns:
    - date: date string (format: YYYY-MM-DD)
    - lme_copper_cash: cash price (float)
    - lme_copper_3m: 3-month forward price (float)
    - lme_copper_stock: stock level (float)

    Args:
        file_path: Path to CSV file. If None, uses default location
                   in src/data/copper/data_copper_lme_all_years.csv

    Returns:
        DataFrame with price data, indexed by date

    Example:
        >>> prices = load_copper_prices()
        >>> print(prices.head())
    """
    if file_path is None:
        # Default path relative to this file
        data_dir = Path(__file__).parent / "copper"
        file_path = data_dir / "data_copper_lme_all_years.csv"

    # Load CSV
    df = pd.read_csv(file_path)

    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Set date as index for easier merging later
    df = df.set_index('date')

    # Sort by date to ensure chronological order
    df = df.sort_index()

    # Ensure numeric columns are numeric (in case of parsing issues)
    numeric_cols = ['lme_copper_cash', 'lme_copper_3m', 'lme_copper_stock']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def load_news_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load news data from CSV file.

    The CSV should contain columns:
    - date: datetime string
    - title: article title
    - text: article text
    - source, publication, link, article_url, etc.

    Args:
        file_path: Path to CSV file. If None, uses default location
                   in src/data/news/copper_news_all_sources.csv

    Returns:
        DataFrame with news data, with date column as datetime
    """
    if file_path is None:
        # Default path relative to this file
        data_dir = Path(__file__).parent / "news"
        file_path = data_dir / "copper_news_all_sources.csv"

    # Load CSV
    df = pd.read_csv(file_path)

    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Sort by date
    df = df.sort_values('date')

    return df


def align_price_and_news(
    price_df: Optional[pd.DataFrame] = None,
    news_df: Optional[pd.DataFrame] = None,
    cutoff_time: time = time(17, 0),  # 5:00 PM
    price_column: str = 'lme_copper_cash'
) -> pd.DataFrame:
    """
    Align price and news data while avoiding lookahead bias.

    The alignment strategy:
    - News published before cutoff_time (e.g., 5 PM) on day D affects price on day D
    - News published after cutoff_time on day D affects price on day D+1
    - This ensures we don't use future information to predict past prices

    Args:
        price_df: DataFrame with price data (from load_copper_prices).
                  If None, loads default price data.
        news_df: DataFrame with news data (from load_news_data).
                 If None, loads default news data.
        cutoff_time: Time of day after which news affects next day's price.
                     Default: 5:00 PM (17:00)
        price_column: Column name from price_df to use as target variable.
                      Default: 'lme_copper_cash'

    Returns:
        DataFrame with columns:
        - date: date (daily, no time component)
        - price: price value for that date
        - lme_copper_stock: LME stock level (if available in price data)
        - news_count: number of news articles for that date
        - combined_text: all news text combined (for sentiment analysis)
        - titles: list of all article titles
        - sources: list of unique news sources
        - publications: list of unique publication names
        - links: list of all article links
        - article_urls: list of original article URLs
        - original_links: list of Google News links
        - domains: list of unique domains
        - queries: list of unique search queries used

    Example:
        >>> aligned_df = align_price_and_news()
        >>> print(aligned_df.head())
    """
    # Load data if not provided
    if price_df is None:
        price_df = load_copper_prices()

    if news_df is None:
        news_df = load_news_data()

    # Extract target price column
    if price_column not in price_df.columns:
        raise ValueError(f"Price column '{price_column}' not found. "
                         f"Available columns: {list(price_df.columns)}")

    price_series = price_df[price_column].copy()
    price_series.name = 'price'

    # Extract stock column if available
    stock_series = None
    if 'lme_copper_stock' in price_df.columns:
        stock_series = price_df['lme_copper_stock'].copy()
        stock_series.name = 'lme_copper_stock'

    # Convert price index to date (remove time component if present)
    price_series.index = pd.to_datetime(price_series.index).normalize()
    if stock_series is not None:
        stock_series.index = pd.to_datetime(stock_series.index).normalize()

    # Create date column for news (to match with price dates)
    news_df = news_df.copy()
    news_df['news_date'] = pd.to_datetime(news_df['date']).dt.date

    # Assign news to appropriate date based on cutoff_time
    # News before cutoff_time affects same day, after cutoff_time affects next day
    news_df['date_for_price'] = news_df['date'].apply(
        lambda dt: dt.date() if dt.time() <= cutoff_time else (dt.date() + timedelta(days=1))
    )

    # Convert to datetime for merging
    news_df['date_for_price'] = pd.to_datetime(news_df['date_for_price'])

    # Helper functions for aggregation
    def combine_text(x):
        """Combine text fields, filtering out empty strings."""
        texts = [str(t) for t in x if pd.notna(t) and str(t).strip()]
        return ' '.join(texts) if texts else ''

    def unique_list(x):
        """Get unique non-null values as a list."""
        unique_vals = [str(v) for v in x.unique() if pd.notna(v) and str(v).strip()]
        return unique_vals if unique_vals else []

    def all_list(x):
        """Get all non-null values as a list."""
        all_vals = [str(v) for v in x if pd.notna(v) and str(v).strip()]
        return all_vals if all_vals else []

    # Aggregate news by date, preserving all information
    news_agg = news_df.groupby('date_for_price').agg({
        'text': combine_text,  # Combined text for sentiment analysis
        'title': lambda x: list(x) if len(x) > 0 else [],  # All titles
        'source': lambda x: unique_list(x),  # Unique sources
        'publication': lambda x: unique_list(x),  # Unique publications
        'link': lambda x: all_list(x),  # All links (may have duplicates)
        'article_url': lambda x: all_list(x),  # All article URLs
        'original_link': lambda x: all_list(x),  # All original Google News links
        'domain': lambda x: unique_list(x),  # Unique domains
        'query': lambda x: unique_list(x),  # Unique queries used
    })

    # Add news count
    news_agg['news_count'] = news_df.groupby('date_for_price').size()

    # Rename columns for clarity
    news_agg.columns = [
        'combined_text',
        'titles',
        'sources',
        'publications',
        'links',
        'article_urls',
        'original_links',
        'domains',
        'queries',
        'news_count'
    ]

    # Reorder columns: news_count first, then others
    cols = ['news_count'] + [c for c in news_agg.columns if c != 'news_count']
    news_agg = news_agg[cols]

    # Merge price and news data
    # Use left join to keep all price dates, even if there's no news
    aligned_df = pd.DataFrame(price_series)
    
    # Add stock if available
    if stock_series is not None:
        stock_df = pd.DataFrame({stock_series.name: stock_series})
        aligned_df = aligned_df.join(stock_df, how='left')
    
    aligned_df = aligned_df.merge(
        news_agg,
        left_index=True,
        right_index=True,
        how='left'
    )

    # Fill missing values for days without news
    aligned_df['news_count'] = aligned_df['news_count'].fillna(0).astype(int)
    aligned_df['combined_text'] = aligned_df['combined_text'].fillna('')

    # Fill list columns with empty lists
    list_columns = ['titles', 'sources', 'publications', 'links', 'article_urls',
                    'original_links', 'domains', 'queries']
    for col in list_columns:
        if col in aligned_df.columns:
            aligned_df[col] = aligned_df[col].apply(
                lambda x: x if isinstance(x, list) else []
            )

    # Reset index to have date as a column
    aligned_df = aligned_df.reset_index()
    aligned_df = aligned_df.rename(columns={'index': 'date'})

    # Reorder columns: date, price, lme_copper_stock (if available), then news-related columns
    base_cols = ['date', 'price']
    if 'lme_copper_stock' in aligned_df.columns:
        base_cols.append('lme_copper_stock')
    cols = base_cols + [c for c in aligned_df.columns if c not in base_cols]
    aligned_df = aligned_df[cols]

    # Sort by date
    aligned_df = aligned_df.sort_values('date').reset_index(drop=True)

    return aligned_df


def get_aligned_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Get aligned price and news data with optional date filtering.

    This is a convenience function that wraps align_price_and_news
    and adds date filtering.

    Args:
        start_date: Start date (inclusive). Format: 'YYYY-MM-DD' or None
        end_date: End date (inclusive). Format: 'YYYY-MM-DD' or None
        **kwargs: Additional arguments passed to align_price_and_news

    Returns:
        Filtered DataFrame with aligned price and news data
    """
    df = align_price_and_news(**kwargs)

    # Filter by date if specified
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]

    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]

    return df.reset_index(drop=True)


def get_price_column(df: pd.DataFrame, column: str = 'lme_copper_cash') -> pd.Series:
    """
    Extract a specific price column from the price DataFrame.

    Args:
        df: Price DataFrame (output from load_copper_prices)
        column: Column name to extract. Options:
                - 'lme_copper_cash': cash price (default)
                - 'lme_copper_3m': 3-month forward price
                - 'lme_copper_stock': stock level

    Returns:
        Series with price data indexed by date
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame. "
                         f"Available columns: {list(df.columns)}")

    return df[column]


if __name__ == "__main__":
    # Example usage
    print("Loading copper prices...")
    prices = load_copper_prices()
    print(f"Loaded {len(prices)} price records")
    print(f"Date range: {prices.index.min()} to {prices.index.max()}")

    print("\n" + "="*60)
    print("Loading and aligning price and news data...")
    aligned_df = align_price_and_news()

    print(f"\nAligned dataset shape: {aligned_df.shape}")
    print(f"Date range: {aligned_df['date'].min()} to {aligned_df['date'].max()}")
    print(f"\nColumns: {list(aligned_df.columns)}")
    print(f"\nFirst few rows:")
    print(aligned_df.head(10))

    print(f"\nSummary statistics:")
    print(aligned_df[['price', 'news_count']].describe())

    print(f"\nDays with news: {(aligned_df['news_count'] > 0).sum()} / {len(aligned_df)}")

