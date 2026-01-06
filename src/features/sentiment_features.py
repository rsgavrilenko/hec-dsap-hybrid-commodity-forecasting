"""
Module for creating news-based sentiment features.

This module provides functions to create features from news text, including:
- FinBERT embeddings (financial sentiment analysis)
- TF-IDF embeddings with PCA dimensionality reduction (fallback)
- Heuristic features based on keyword matching
- Aggregate sentiment scores
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from typing import List, Dict, Optional

# Try to import FinBERT
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    print("⚠️  FinBERT not available. Using TF-IDF as fallback.")


def create_tfidf_embeddings(
    texts: List[str],
    max_features: int = 500,
    n_components: int = 50,
    random_state: int = 42
) -> np.ndarray:
    """
    Create TF-IDF embeddings with PCA dimensionality reduction.
    
    Args:
        texts: List of text strings to vectorize
        max_features: Maximum number of TF-IDF features. Default: 500
        n_components: Number of PCA components. Default: 50
        random_state: Random state for PCA. Default: 42
    
    Returns:
        Array of shape (n_samples, n_components) with PCA-reduced embeddings
    """
    # Prepare text data (replace NaN with empty strings)
    texts_clean = [str(text) if pd.notna(text) else '' for text in texts]
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,            # Minimum document frequency
        max_df=0.95          # Maximum document frequency (ignore very common words)
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts_clean)
    tfidf_dense = tfidf_matrix.toarray()
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_components, random_state=random_state)
    embeddings = pca.fit_transform(tfidf_dense)
    
    return embeddings


def create_finbert_embeddings(
    texts: List[str],
    batch_size: int = 32,
    max_length: int = 512
) -> np.ndarray:
    """
    Create FinBERT embeddings for financial sentiment analysis.
    
    Uses the FinBERT model (ProsusAI/finbert) which is trained on financial texts.
    
    Args:
        texts: List of text strings to encode
        batch_size: Batch size for processing. Default: 32
        max_length: Maximum sequence length. Default: 512
        
    Returns:
        Array of shape (n_samples, 768) with FinBERT embeddings
    """
    if not FINBERT_AVAILABLE:
        raise ImportError("FinBERT not available. Install transformers and torch.")
    
    # Load FinBERT model and tokenizer
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    # Prepare text data
    texts_clean = [str(text) if pd.notna(text) else '' for text in texts]
    
    embeddings = []
    
    # Process in batches
    for i in range(0, len(texts_clean), batch_size):
        batch_texts = texts_clean[i:i + batch_size]
        
        # Tokenize
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**encoded)
            # Use [CLS] token embedding (first token)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        
        embeddings.append(batch_embeddings)
    
    # Concatenate all batches
    all_embeddings = np.vstack(embeddings)
    
    return all_embeddings


def get_heuristic_keywords() -> Dict[str, Dict]:
    """
    Get dictionary of heuristic features with their keywords.
    
    Returns:
        Dictionary mapping feature names to keyword lists and expected effects
    """
    return {
        'mine_closure': {
            'keywords': ['mine closure', 'mine closed', 'mine shutdown', 'mine shut down', 
                         'closure of mine', 'shutting down mine', 'ceases operation', 
                         'suspends operation', 'halts production'],
            'expected_effect': 'positive'
        },
        'strike_labor': {
            'keywords': ['strike', 'striking', 'labor dispute', 'workers strike', 
                         'union strike', 'walkout', 'industrial action'],
            'expected_effect': 'positive'
        },
        'production_cut': {
            'keywords': ['production cut', 'cuts production', 'reduces production', 
                         'production reduction', 'output cut', 'reduces output',
                         'lower production', 'production decline', 'output decline'],
            'expected_effect': 'positive'
        },
        'export_ban': {
            'keywords': ['export ban', 'export restriction', 'bans export', 
                         'export embargo', 'export limit'],
            'expected_effect': 'positive'
        },
        'sanctions': {
            'keywords': ['sanction', 'sanctions', 'embargo', 'trade restriction'],
            'expected_effect': 'positive'
        },
        'mine_opening': {
            'keywords': ['new mine', 'mine opening', 'opens mine', 'new mine opens',
                         'mine starts', 'begins mining', 'new mining project'],
            'expected_effect': 'negative'
        },
        'production_increase': {
            'keywords': ['production increase', 'increases production', 'raises production',
                         'production rise', 'output increase', 'increases output',
                         'higher production', 'production growth', 'output growth',
                         'production expansion', 'expands production', 'ramps up production'],
            'expected_effect': 'negative'
        },
        'capacity_expansion': {
            'keywords': ['capacity expansion', 'expands capacity', 'capacity increase',
                         'expands production capacity', 'increases capacity'],
            'expected_effect': 'negative'
        },
        'demand_surge': {
            'keywords': ['demand surge', 'surge in demand', 'strong demand', 
                         'robust demand', 'demand growth', 'growing demand',
                         'demand increase', 'increasing demand'],
            'expected_effect': 'positive'
        },
        'china_demand': {
            'keywords': ['china demand', 'chinese demand', 'china\'s demand',
                         'demand from china', 'chinese consumption'],
            'expected_effect': 'positive'
        },
        'infrastructure_spending': {
            'keywords': ['infrastructure', 'infrastructure spending', 'infrastructure investment',
                         'infrastructure project', 'infrastructure development'],
            'expected_effect': 'positive'
        },
        'economic_growth': {
            'keywords': ['economic growth', 'gdp growth', 'economic expansion',
                         'economic recovery', 'strong economy'],
            'expected_effect': 'positive'
        },
        'demand_weakness': {
            'keywords': ['weak demand', 'demand weakness', 'slowing demand',
                         'declining demand', 'demand decline', 'soft demand'],
            'expected_effect': 'negative'
        },
        'recession': {
            'keywords': ['recession', 'economic downturn', 'economic crisis',
                         'economic slowdown', 'economic contraction'],
            'expected_effect': 'negative'
        },
        'inventory_drop': {
            'keywords': ['inventory drop', 'stock decline', 'inventory decline',
                         'stock fall', 'inventory fall', 'lower inventory',
                         'reduced inventory', 'stock decrease', 'inventory decrease',
                         'warehouse stock down', 'stocks down'],
            'expected_effect': 'positive'
        },
        'inventory_build': {
            'keywords': ['inventory build', 'stock build', 'inventory increase',
                         'stock increase', 'higher inventory', 'inventory rise',
                         'stock rise', 'warehouse stock up', 'stocks up'],
            'expected_effect': 'negative'
        },
        'price_rally': {
            'keywords': ['price rally', 'price surge', 'price jump', 'price spike',
                         'price gains', 'price rise', 'prices rise', 'prices gain',
                         'record high', 'hits record', 'all-time high'],
            'expected_effect': 'positive'
        },
        'price_crash': {
            'keywords': ['price crash', 'price slump', 'price drop', 'price fall',
                         'price decline', 'prices fall', 'prices drop', 'prices decline'],
            'expected_effect': 'negative'
        },
        'environmental_issue': {
            'keywords': ['environmental', 'pollution', 'environmental concern',
                         'environmental impact', 'environmental damage'],
            'expected_effect': 'positive'
        },
        'regulation': {
            'keywords': ['regulation', 'regulatory', 'new regulation', 'regulatory change'],
            'expected_effect': 'ambiguous'
        }
    }


def create_heuristic_features(
    df: pd.DataFrame,
    text_column: str = 'combined_text',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create heuristic features based on keyword matching in news text.
    
    Args:
        df: DataFrame with text_column containing news text
        text_column: Name of column containing text. Default: 'combined_text'
        verbose: Whether to print progress. Default: True
    
    Returns:
        DataFrame with new heuristic feature columns added
    """
    df = df.copy()
    
    # Get texts
    texts = df[text_column].fillna('').astype(str).tolist()
    texts_lower = [text.lower() for text in texts]
    
    # Get heuristics
    heuristics = get_heuristic_keywords()
    
    # Create heuristic features
    for feature_name, heuristic in heuristics.items():
        keywords = heuristic['keywords']
        # Check if any keyword appears in the text
        matches = []
        for text in texts_lower:
            found = any(keyword in text for keyword in keywords)
            matches.append(1 if found else 0)
        
        df[f'news_heuristic_{feature_name}'] = matches
        
        if verbose:
            count = sum(matches)
            if count > 0:
                print(f"  ✅ {feature_name:25s}: {count:5d} days ({count/len(df)*100:.1f}%)")
    
    # Calculate aggregate scores
    bullish_features = [f'news_heuristic_{name}' for name, h in heuristics.items() 
                        if h['expected_effect'] == 'positive']
    bearish_features = [f'news_heuristic_{name}' for name, h in heuristics.items() 
                        if h['expected_effect'] == 'negative']
    
    if bullish_features and bearish_features:
        df['news_heuristic_bullish_score'] = df[bullish_features].sum(axis=1)
        df['news_heuristic_bearish_score'] = df[bearish_features].sum(axis=1)
        df['news_heuristic_net_score'] = (df['news_heuristic_bullish_score'] - 
                                           df['news_heuristic_bearish_score'])
        
        if verbose:
            print(f"\n✅ Created aggregate scores:")
            print(f"   Bullish score range: {df['news_heuristic_bullish_score'].min()} - {df['news_heuristic_bullish_score'].max()}")
            print(f"   Bearish score range: {df['news_heuristic_bearish_score'].min()} - {df['news_heuristic_bearish_score'].max()}")
            print(f"   Net score range: {df['news_heuristic_net_score'].min()} - {df['news_heuristic_net_score'].max()}")
    
    if verbose:
        total_with_heuristic = (df[[f'news_heuristic_{name}' for name in heuristics.keys()]].sum(axis=1) > 0).sum()
        print(f"\n✅ Created {len(heuristics)} heuristic features")
        print(f"   Total days with at least one heuristic: {total_with_heuristic}")
    
    return df


def create_news_features(
    df: pd.DataFrame,
    text_column: str = 'combined_text',
    use_finbert: bool = True,
    max_features: int = 500,
    n_components: int = 50,
    create_heuristics: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create all news-based features: FinBERT or TF-IDF embeddings and heuristic features.
    
    Args:
        df: DataFrame with text_column containing news text
        text_column: Name of column containing text. Default: 'combined_text'
        use_finbert: Whether to use FinBERT (if available). Default: True
                     Falls back to TF-IDF if FinBERT not available
        max_features: Maximum number of TF-IDF features (if using TF-IDF). Default: 500
        n_components: Number of PCA components (if using TF-IDF). Default: 50
        create_heuristics: Whether to create heuristic features. Default: True
        verbose: Whether to print progress. Default: True
    
    Returns:
        DataFrame with new feature columns added:
        - news_embed_0 to news_embed_{n}: FinBERT or TF-IDF embeddings
        - news_heuristic_*: Heuristic binary features
        - news_heuristic_bullish_score: Sum of bullish heuristics
        - news_heuristic_bearish_score: Sum of bearish heuristics
        - news_heuristic_net_score: Bullish - Bearish
    """
    df = df.copy()
    
    # Prepare text data
    texts = df[text_column].fillna('').astype(str).tolist()
    
    # Try to use FinBERT if requested and available
    if use_finbert and FINBERT_AVAILABLE:
        if verbose:
            print(f"📝 Computing FinBERT embeddings for {len(texts)} documents...")
            print(f"   Using ProsusAI/finbert model")
        
        try:
            # Create FinBERT embeddings
            embeddings = create_finbert_embeddings(texts)
            
            # Apply PCA to reduce dimensionality (FinBERT gives 768-dim vectors)
            if n_components < embeddings.shape[1]:
                pca = PCA(n_components=n_components, random_state=42)
                embeddings = pca.fit_transform(embeddings)
                if verbose:
                    print(f"   Applied PCA: {embeddings.shape[1]} components")
            
            if verbose:
                print(f"✅ FinBERT embeddings shape: {embeddings.shape}")
            
        except Exception as e:
            if verbose:
                print(f"⚠️  FinBERT failed: {e}")
                print(f"   Falling back to TF-IDF...")
            use_finbert = False
    
    # Fallback to TF-IDF if FinBERT not used or failed
    if not use_finbert or not FINBERT_AVAILABLE:
        if verbose:
            print(f"📝 Computing TF-IDF vectors for {len(texts)} documents...")
            print(f"   Max features: {max_features}, PCA components: {n_components}")
        
        # Create TF-IDF embeddings
        embeddings = create_tfidf_embeddings(
            texts,
            max_features=max_features,
            n_components=n_components
        )
        
        if verbose:
            print(f"✅ TF-IDF embeddings shape: {embeddings.shape}")
    
    # Add embeddings as columns
    n_embed_features = embeddings.shape[1]
    for i in range(n_embed_features):
        df[f'news_embed_{i}'] = embeddings[:, i]
    
    if verbose:
        method = "FinBERT" if (use_finbert and FINBERT_AVAILABLE) else "TF-IDF"
        print(f"✅ Added {n_embed_features} news embedding columns ({method}) to dataframe")
    
    # Create heuristic features
    if create_heuristics:
        if verbose:
            print("\n" + "="*80)
            print("🔍 Creating Heuristic Features from News Keywords")
            print("="*80)
        
        df = create_heuristic_features(df, text_column=text_column, verbose=verbose)
    
    return df

