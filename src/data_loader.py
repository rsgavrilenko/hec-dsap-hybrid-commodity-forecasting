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
from typing import Optional, List
from datetime import time, timedelta
import re


_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")


def _normalize_title(title: str) -> str:
    """Normalize title for deduplication (lowercase, strip punctuation, collapse whitespace)."""
    if title is None:
        return ""
    t = str(title).strip().lower()
    t = _NON_ALNUM_RE.sub(" ", t)
    t = _WHITESPACE_RE.sub(" ", t).strip()
    return t


def _is_price_recap_title(title: str) -> bool:
    """
    Heuristic: identify 'price recap / chart narration' titles like:
    'LME Copper Prices Surge Past $13,000 to Hit Record High'.
    """
    if title is None:
        return False
    t = str(title).lower()

    # Must talk about price/moves
    price_terms = r"(price|prices|lme|copper|metals)"
    move_terms = r"(surge|jump|soar|rise|rises|rising|gain|gains|climb|advance|extend|hit|hits|record|high|low|drop|fall|falls|slide|slip|tumble|plunge|retreat)"
    has_price_move = re.search(rf"\b{price_terms}\b.*\b{move_terms}\b", t) is not None

    # Often contains explicit numbers/levels
    has_level = re.search(r"(\$?\s?\d{1,3}(?:[,]\d{3})+(?:\.\d+)?|\b\d+(?:\.\d+)?\s?(?:%|pct|percent)\b)", t) is not None

    # Classic recap formats
    recap_phrases = [
        "to hit", "to reach", "past", "above", "below", "at record", "record high", "record low",
        "extends gains", "extends losses", "down", "up", "slips", "slides"
    ]
    has_recap_phrase = any(p in t for p in recap_phrases)

    return has_price_move and (has_level or has_recap_phrase)


def _is_causal_news(title: str, text: str) -> bool:
    """Heuristic: whether the news likely contains 'why' (supply/demand shock) rather than pure price narration."""
    blob = f"{title or ''} {text or ''}".lower()
    causal_terms = [
        "strike", "labor", "union", "mine", "mining", "smelter", "refinery", "shutdown",
        "outage", "inventory", "stocks", "warehouse", "lme stocks", "sanction", "embargo",
        "export ban", "export restriction", "disruption", "supply", "shortage", "demand",
        "china", "pmi", "stimulus", "tariff", "protest", "royalty", "tax", "permit",
        "earthquake", "flood", "fire", "accident"
    ]
    return any(term in blob for term in causal_terms)


def _annotate_news_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Add helper columns used for filtering/dedup/debugging."""
    df = df.copy()
    if 'title' in df.columns:
        df['title_normalized'] = df['title'].fillna('').astype(str).map(_normalize_title)
    else:
        df['title_normalized'] = ""

    # Basic length stats
    if 'text' in df.columns:
        df['text_len'] = df['text'].fillna('').astype(str).str.len()
    else:
        df['text_len'] = 0
    df['title_len'] = df['title'].fillna('').astype(str).str.len() if 'title' in df.columns else 0

    # Price recap flags
    df['is_price_recap'] = df['title'].fillna('').astype(str).map(_is_price_recap_title) if 'title' in df.columns else False
    if 'text' in df.columns and 'title' in df.columns:
        df['is_price_recap_only'] = df.apply(lambda r: bool(r['is_price_recap']) and not _is_causal_news(r.get('title'), r.get('text')), axis=1)
    else:
        df['is_price_recap_only'] = df['is_price_recap']

    return df


def _dedup_news(df: pd.DataFrame) -> pd.DataFrame:
    """Conservative deduplication: prefer URL-based, then (date, normalized title)."""
    df = df.copy()
    before = len(df)

    # URL-based dedup (most reliable)
    for col in ['article_url', 'original_link', 'link']:
        if col in df.columns:
            non_null = df[col].notna().sum()
            if non_null > 0:
                df = df.drop_duplicates(subset=[col], keep='first')

    # Same title on same day (common when the same item is pulled from multiple queries)
    if 'date' in df.columns and 'title_normalized' in df.columns:
        df['date_only'] = pd.to_datetime(df['date']).dt.date
        df = df.drop_duplicates(subset=['date_only', 'title_normalized'], keep='first')
        df = df.drop(columns=['date_only'], errors='ignore')

    after = len(df)
    if after != before:
        print(f"📌 Deduplicated news: {before} -> {after} articles (removed {before-after})")
    return df


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
        # Prefer Week-11 structure: data/raw/copper/...
        project_root = Path(__file__).resolve().parents[1]
        preferred = project_root / "data" / "raw" / "copper" / "data_copper_lme_all_years.csv"
        legacy = project_root / "src" / "data" / "copper" / "data_copper_lme_all_years.csv"
        file_path = preferred if preferred.exists() else legacy

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


def load_news_data(
    file_path: Optional[str] = None, 
    filter_sufficient_only: bool = False,
    allowed_sources: Optional[List[str]] = None,
    drop_price_recap_only: bool = False,
    deduplicate: bool = True
) -> pd.DataFrame:
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
        filter_sufficient_only: If True, only load news with is_likely_sufficient=True.
                               Default: False
        allowed_sources: List of allowed source names. If None, all sources are included.
                        Default: None (all sources)
                        Common values: ['Reuters', 'Mining.com', 'Bloomberg']

    Returns:
        DataFrame with news data, with date column as datetime
    """
    if file_path is None:
        # Prefer Week-11 structure: data/raw/news/...
        project_root = Path(__file__).resolve().parents[1]
        preferred = project_root / "data" / "raw" / "news" / "copper_news_all_sources.csv"
        legacy = project_root / "src" / "data" / "news" / "copper_news_all_sources.csv"
        file_path = preferred if preferred.exists() else legacy

    # Load CSV
    df = pd.read_csv(file_path)
    initial_count = len(df)

    # Add helper flags for filtering/analysis
    if len(df) > 0:
        df = _annotate_news_quality(df)

    # Deduplicate (safe): URL-based then (date, normalized title)
    if deduplicate and len(df) > 0:
        df = _dedup_news(df)
        initial_count = len(df)

    # Filter by source if specified
    if allowed_sources is not None:
        # Normalize source names for matching (case-insensitive)
        df['source_normalized'] = df['source'].str.strip().str.lower()
        allowed_normalized = [s.strip().lower() for s in allowed_sources]
        
        # Filter
        df = df[df['source_normalized'].isin(allowed_normalized)].copy()
        source_filtered_count = len(df)
        print(f"   Filtered by source: {initial_count} -> {source_filtered_count} articles "
              f"({source_filtered_count/initial_count*100:.1f}%)")
        print(f"   Allowed sources: {', '.join(allowed_sources)}")
        
        # Drop temporary column
        df = df.drop(columns=['source_normalized'], errors='ignore')
        initial_count = source_filtered_count

    # Filter by is_likely_sufficient if column exists and filter requested
    if filter_sufficient_only and 'is_likely_sufficient' in df.columns:
        df = df[df['is_likely_sufficient'] == True].copy()
        filtered_count = len(df)
        print(f"   Filtered by sufficient: {initial_count} -> {filtered_count} articles "
              f"({filtered_count/initial_count*100:.1f}%)")
        initial_count = filtered_count

    # Drop 'price recap only' items (optional, helps avoid leaking the chart into features)
    if drop_price_recap_only and 'is_price_recap_only' in df.columns:
        df = df[~df['is_price_recap_only']].copy()
        filtered_count = len(df)
        print(f"   Dropped price-recap-only: {initial_count} -> {filtered_count} articles "
              f"({filtered_count/initial_count*100:.1f}%)")

    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Sort by date
    df = df.sort_values('date')

    return df


def align_price_and_news(
    price_df: Optional[pd.DataFrame] = None,
    news_df: Optional[pd.DataFrame] = None,
    cutoff_time: time = time(17, 0),  # 5:00 PM
    price_column: str = 'lme_copper_cash',
    filter_sufficient_news: bool = False,
    allowed_sources: Optional[List[str]] = None,
    drop_price_recap_only: bool = False,
    deduplicate_news: bool = True
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
        filter_sufficient_news: If True, only use news with is_likely_sufficient=True.
                               Default: False
        allowed_sources: List of allowed source names. If None, all sources are included.
                        Default: None (all sources)
                        Common values: ['Reuters', 'Mining.com', 'Bloomberg']
        drop_price_recap_only: If True, drop 'price recap only' items (titles that just narrate price moves).
                              Default: False
        deduplicate_news: If True, deduplicate news items by URL and (date, normalized title).
                          Default: True

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
        news_df = load_news_data(
            filter_sufficient_only=filter_sufficient_news,
            allowed_sources=allowed_sources,
            drop_price_recap_only=drop_price_recap_only,
            deduplicate=deduplicate_news
        )
    elif filter_sufficient_news and 'is_likely_sufficient' in news_df.columns:
        # Filter existing dataframe
        initial_count = len(news_df)
        news_df = news_df[news_df['is_likely_sufficient'] == True].copy()
        print(f"   Filtered news: {initial_count} -> {len(news_df)} articles "
              f"({len(news_df)/initial_count*100:.1f}%)")
    
    # Filter by source if specified and news_df was provided
    if allowed_sources is not None and news_df is not None:
        initial_count = len(news_df)
        news_df['source_normalized'] = news_df['source'].str.strip().str.lower()
        allowed_normalized = [s.strip().lower() for s in allowed_sources]
        news_df = news_df[news_df['source_normalized'].isin(allowed_normalized)].copy()
        news_df = news_df.drop(columns=['source_normalized'], errors='ignore')
        print(f"   Filtered by source: {initial_count} -> {len(news_df)} articles "
              f"({len(news_df)/initial_count*100:.1f}%)")
        print(f"   Allowed sources: {', '.join(allowed_sources)}")

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

    # Use both title + text for combined_text (titles often carry most of the signal in RSS)
    # This also makes the downstream NLP less sensitive to short/templated snippets.
    if 'title' in news_df.columns and 'text' in news_df.columns:
        news_df['text_plus_title'] = (
            news_df['title'].fillna('').astype(str).str.strip()
            + ". "
            + news_df['text'].fillna('').astype(str).str.strip()
        ).str.strip()
    elif 'text' in news_df.columns:
        news_df['text_plus_title'] = news_df['text'].fillna('').astype(str)
    elif 'title' in news_df.columns:
        news_df['text_plus_title'] = news_df['title'].fillna('').astype(str)
    else:
        news_df['text_plus_title'] = ''

    # Assign news to appropriate date based on cutoff_time
    # News before cutoff_time affects same day, after cutoff_time affects next day
    news_df['date_for_price'] = news_df['date'].apply(
        lambda dt: dt.date() if dt.time() <= cutoff_time else (dt.date() + timedelta(days=1))
    )

    # Get available price dates for filtering
    available_price_dates = set(price_series.index.date)
    last_price_date = max(available_price_dates)
    first_price_date = min(available_price_dates)
    
    # Count articles before filtering
    total_articles = len(news_df)
    
    # Filter news to only dates that have prices
    # (This is correct behavior - we can't use news for dates without prices)
    news_df_filtered = news_df[news_df['date_for_price'].isin(available_price_dates)].copy()
    articles_used = len(news_df_filtered)
    articles_dropped = total_articles - articles_used
    
    if articles_dropped > 0:
        # Note: articles can also be dropped *within* the calendar range if there is no price for that date
        # (weekends/holidays/missing LME observations), or if cutoff_time shifts an article onto such a date.
        dropped_before = int((news_df['date_for_price'] < first_price_date).sum())
        dropped_after = int((news_df['date_for_price'] > last_price_date).sum())
        dropped_no_price_on_date = int(articles_dropped - dropped_before - dropped_after)
        print(f"📊 News alignment summary:")
        print(f"   Total articles: {total_articles}")
        print(f"   Articles in price date range ({first_price_date} to {last_price_date}): {articles_used}")
        print(f"   Articles dropped: {articles_dropped}")
        if dropped_before > 0:
            print(f"      - Before {first_price_date}: {dropped_before}")
        if dropped_after > 0:
            print(f"      - After {last_price_date}: {dropped_after}")
        if dropped_no_price_on_date > 0:
            print(f"      - No price for date (weekends/holidays/missing): {dropped_no_price_on_date}")
    
    # Use filtered news for alignment
    news_df = news_df_filtered

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
        'text_plus_title': combine_text,  # Combined text for sentiment analysis (title + snippet)
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

