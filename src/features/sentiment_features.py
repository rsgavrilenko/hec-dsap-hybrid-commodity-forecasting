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

try:
    from transformers import AutoModelForSequenceClassification
    FINBERT_CLASSIFIER_AVAILABLE = True
except ImportError:
    FINBERT_CLASSIFIER_AVAILABLE = False


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


def create_finbert_sentiment_scores(
    texts: List[str],
    batch_size: int = 32,
    max_length: int = 256
) -> np.ndarray:
    """
    Compute FinBERT sentiment scores (pos/neg/neu probabilities).

    Returns array of shape (n_samples, 3): [neg, neu, pos] for ProsusAI/finbert.
    """
    if not FINBERT_CLASSIFIER_AVAILABLE:
        raise ImportError("FinBERT classifier not available. Install transformers and torch.")

    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    texts_clean = [str(text) if pd.notna(text) else '' for text in texts]

    all_probs: List[np.ndarray] = []
    for i in range(0, len(texts_clean), batch_size):
        batch_texts = texts_clean[i:i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        with torch.no_grad():
            out = model(**encoded)
            logits = out.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

    return np.vstack(all_probs)


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
        },
        'war_conflict': {
            'keywords': ['war', 'conflict', 'military', 'invasion', 'attack', 'bombing',
                         'armed conflict', 'hostilities', 'combat', 'warfare', 'battle',
                         'military action', 'military operation', 'defense', 'offensive'],
            'expected_effect': 'positive'  # Wars disrupt supply chains
        },
        'geopolitical_tension': {
            'keywords': ['geopolitical', 'tension', 'crisis', 'diplomatic', 'sanctions',
                         'embargo', 'trade war', 'tariff war', 'trade dispute', 'trade conflict',
                         'political crisis', 'international crisis', 'standoff', 'confrontation'],
            'expected_effect': 'positive'  # Geopolitical tensions disrupt trade
        },
        'supply_chain_disruption': {
            'keywords': ['supply chain disruption', 'supply chain crisis', 'logistics crisis',
                         'shipping disruption', 'port closure', 'transport disruption',
                         'shipping delay', 'logistics delay', 'supply chain breakdown',
                         'transportation crisis', 'shipping crisis', 'port congestion',
                         'freight disruption', 'cargo delay'],
            'expected_effect': 'positive'  # Disruptions reduce supply
        },
        'export_restriction': {
            'keywords': ['export restriction', 'export limit', 'export control', 'export quota',
                         'export ban', 'export embargo', 'restricts export', 'limits export',
                         'export policy', 'export regulation', 'trade restriction'],
            'expected_effect': 'positive'  # Export restrictions reduce supply
        },
        'sanctions_embargo': {
            'keywords': ['sanctions', 'sanction', 'embargo', 'trade embargo', 'economic sanctions',
                         'financial sanctions', 'trade ban', 'trade restriction', 'blockade',
                         'trade blockade', 'economic blockade'],
            'expected_effect': 'positive'  # Sanctions disrupt supply
        },
        'russia_ukraine': {
            'keywords': ['russia', 'russian', 'ukraine', 'ukrainian', 'putin', 'zelensky',
                         'moscow', 'kyiv', 'kiev', 'donbas', 'donbass', 'crimea',
                         'russian invasion', 'ukraine war', 'russia-ukraine', 'russian-ukrainian'],
            'expected_effect': 'positive'  # Major geopolitical event affecting commodities
        },
        'china_trade': {
            'keywords': ['china trade', 'chinese trade', 'china tariff', 'chinese tariff',
                         'china export', 'chinese export', 'china import', 'chinese import',
                         'trade war china', 'china trade war', 'us-china trade', 'china-us trade',
                         'trade dispute china', 'china trade dispute'],
            'expected_effect': 'positive'  # Trade tensions affect demand/supply
        },
        'middle_east_conflict': {
            'keywords': ['middle east', 'middle eastern', 'gulf', 'persian gulf', 'red sea',
                         'suez canal', 'strait of hormuz', 'iran', 'iraq', 'syria', 'yemen',
                         'saudi arabia', 'uae', 'qatar', 'israel', 'palestine', 'gaza',
                         'houthi', 'hezbollah', 'hamas'],
            'expected_effect': 'positive'  # Middle East conflicts affect shipping/energy
        },
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
        
        # Category-based aggregations (group related heuristics for better signal)
        # Supply-side shocks (mine closures, strikes, production cuts)
        supply_shock_heuristics = ['mine_closure', 'strike_labor', 'production_cut', 'export_ban']
        supply_cols = [f'news_heuristic_{h}' for h in supply_shock_heuristics if f'news_heuristic_{h}' in df.columns]
        if supply_cols:
            df['news_heuristic_supply_shock'] = df[supply_cols].sum(axis=1)
        
        # Geopolitical shocks (wars, sanctions, embargoes)
        geo_shock_heuristics = ['war_conflict', 'geopolitical_tension', 'sanctions', 'sanctions_embargo', 
                                'russia_ukraine', 'middle_east_conflict', 'export_restriction']
        geo_cols = [f'news_heuristic_{h}' for h in geo_shock_heuristics if f'news_heuristic_{h}' in df.columns]
        if geo_cols:
            df['news_heuristic_geo_shock'] = df[geo_cols].sum(axis=1)
        
        # Demand-side signals (demand surge, China demand, infrastructure)
        demand_positive_heuristics = ['demand_surge', 'china_demand', 'infrastructure_spending', 'economic_growth']
        demand_pos_cols = [f'news_heuristic_{h}' for h in demand_positive_heuristics if f'news_heuristic_{h}' in df.columns]
        if demand_pos_cols:
            df['news_heuristic_demand_positive'] = df[demand_pos_cols].sum(axis=1)
        
        # Demand-side negative (weakness, recession)
        demand_negative_heuristics = ['demand_weakness', 'recession']
        demand_neg_cols = [f'news_heuristic_{h}' for h in demand_negative_heuristics if f'news_heuristic_{h}' in df.columns]
        if demand_neg_cols:
            df['news_heuristic_demand_negative'] = df[demand_neg_cols].sum(axis=1)
        
        # Production expansion (increases supply, bearish)
        production_expansion_heuristics = ['mine_opening', 'production_increase', 'capacity_expansion']
        prod_exp_cols = [f'news_heuristic_{h}' for h in production_expansion_heuristics if f'news_heuristic_{h}' in df.columns]
        if prod_exp_cols:
            df['news_heuristic_production_expansion'] = df[prod_exp_cols].sum(axis=1)
        
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
    finbert_mode: str = 'sentiment',  # 'sentiment' (compact) or 'embeddings' (high-dim)
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
    
    # Filter out empty texts for processing (only process non-empty texts)
    # This significantly speeds up FinBERT processing
    non_empty_mask = pd.Series([len(text.strip()) > 0 for text in texts])
    non_empty_indices = non_empty_mask[non_empty_mask].index.tolist()
    texts_to_process = [texts[i] for i in non_empty_indices]
    
    if verbose:
        total_texts = len(texts)
        non_empty_count = len(texts_to_process)
        empty_count = total_texts - non_empty_count
        print(f"📊 Text statistics:")
        print(f"   Total rows: {total_texts}")
        print(f"   Non-empty texts: {non_empty_count} ({non_empty_count/total_texts*100:.1f}%)")
        print(f"   Empty texts: {empty_count} ({empty_count/total_texts*100:.1f}%)")
        print(f"   Processing only {non_empty_count} non-empty texts with FinBERT...")
    
    # Edge case: no non-empty texts at all → skip FinBERT/TF-IDF, keep heuristics/availability features
    non_empty_count = len(texts_to_process)
    if non_empty_count == 0:
        if verbose:
            print("⚠️  No non-empty news texts found. Skipping FinBERT/TF-IDF features.")
        # Provide neutral FinBERT-like columns so downstream code remains stable
        df['news_finbert_neg'] = 0.0
        df['news_finbert_neu'] = 1.0
        df['news_finbert_pos'] = 0.0
        df['news_finbert_net'] = 0.0
        # Still create heuristics and rolling features below
        # Jump to heuristics/rolling section
        if create_heuristics:
            if verbose:
                print("\n" + "="*80)
                print("🔍 Creating Heuristic Features from News Keywords")
                print("="*80)
            df = create_heuristic_features(df, text_column=text_column, verbose=verbose)
        if verbose:
            print("\n" + "="*80)
            print("📊 Creating Rolling Aggregations and Lags for News Features")
            print("="*80)
        df = add_rolling_news_features(df, verbose=verbose)
        return df
    
    # Add basic news availability features
    if 'news_count' in df.columns:
        df['no_news'] = (df['news_count'] == 0).astype(int)
    
    # Add temporal patterns for news (day of week, month effects)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['month'] = df['date'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # News intensity by day of week (some days may have more impactful news)
        # Create interaction: news_count * day_of_week (captures weekly patterns)
        if 'news_count' in df.columns:
            df['news_count_weekday'] = df['news_count'] * (df['day_of_week'] + 1)  # Weight by day
            df['news_count_weekend'] = df['news_count'] * df['is_weekend']  # Weekend effect
    else:
        df['no_news'] = 1

    # Try to use FinBERT if requested and available
    if use_finbert and FINBERT_AVAILABLE and finbert_mode == 'embeddings':
        if verbose:
            print(f"📝 Computing FinBERT embeddings for {len(texts_to_process)} documents...")
            print(f"   Using ProsusAI/finbert model")
            print(f"   (Skipping {len(texts) - len(texts_to_process)} empty texts)")
        
        try:
            # Create FinBERT embeddings only for non-empty texts
            embeddings_non_empty = create_finbert_embeddings(texts_to_process)
            
            # Create full embeddings array (zeros for empty texts)
            embeddings = np.zeros((len(texts), embeddings_non_empty.shape[1]))
            for idx, orig_idx in enumerate(non_empty_indices):
                embeddings[orig_idx] = embeddings_non_empty[idx]
            
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
    
    # Use FinBERT sentiment classifier (preferred: low-dim, more robust)
    if use_finbert and finbert_mode == 'sentiment':
        if verbose:
            print(f"📝 Computing FinBERT sentiment scores for {len(texts_to_process)} documents...")
            print(f"   Using ProsusAI/finbert (sequence classification)")
            print(f"   (Skipping {len(texts) - len(texts_to_process)} empty texts)")
        try:
            # Process only non-empty texts with FinBERT
            probs_non_empty = create_finbert_sentiment_scores(texts_to_process)
            
            # Initialize columns with default values (neutral for empty texts)
            df['news_finbert_neg'] = 0.0
            df['news_finbert_neu'] = 1.0  # Default to neutral for empty texts
            df['news_finbert_pos'] = 0.0
            df['news_finbert_net'] = 0.0
            
            # Fill in scores only for non-empty texts
            # ProsusAI/finbert label order is typically [negative, neutral, positive]
            # probs_non_empty is numpy array of shape (n_non_empty, 3)
            for idx, orig_idx in enumerate(non_empty_indices):
                df.loc[orig_idx, 'news_finbert_neg'] = float(probs_non_empty[idx, 0])
                df.loc[orig_idx, 'news_finbert_neu'] = float(probs_non_empty[idx, 1])
                df.loc[orig_idx, 'news_finbert_pos'] = float(probs_non_empty[idx, 2])
                df.loc[orig_idx, 'news_finbert_net'] = float(probs_non_empty[idx, 2] - probs_non_empty[idx, 0])

            # Force neutral on no-news days (prevents model from learning noise from empty text)
            if 'news_count' in df.columns:
                mask_no = df['news_count'] == 0
                df.loc[mask_no, ['news_finbert_neg', 'news_finbert_pos', 'news_finbert_net']] = 0.0
                df.loc[mask_no, 'news_finbert_neu'] = 1.0

            if verbose:
                print("✅ Added FinBERT sentiment score columns: news_finbert_{neg,neu,pos,net}")
        except Exception as e:
            if verbose:
                print(f"⚠️  FinBERT sentiment failed: {e}")
                print("   Falling back to TF-IDF embeddings...")
            use_finbert = False
            finbert_mode = 'embeddings'

    # Fallback to TF-IDF if FinBERT not used or failed
    if (not use_finbert or not FINBERT_AVAILABLE) and finbert_mode == 'embeddings':
        if verbose:
            print(f"📝 Computing TF-IDF vectors for {len(texts_to_process)} documents...")
            print(f"   Max features: {max_features}, PCA components: {n_components}")
            print(f"   (Skipping {len(texts) - len(texts_to_process)} empty texts)")
        
        # Create TF-IDF embeddings only for non-empty texts
        embeddings_non_empty = create_tfidf_embeddings(
            texts_to_process,
            max_features=max_features,
            n_components=n_components
        )
        
        # Create full embeddings array (zeros for empty texts)
        embeddings = np.zeros((len(texts), embeddings_non_empty.shape[1]))
        for idx, orig_idx in enumerate(non_empty_indices):
            embeddings[orig_idx] = embeddings_non_empty[idx]
        
        if verbose:
            print(f"✅ TF-IDF embeddings shape: {embeddings.shape}")
    
    # Add embeddings as columns (only if embeddings were created)
    if finbert_mode == 'embeddings':
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
    
    # Add weighted sentiment features (based on source reliability)
    if 'sources' in df.columns:
        if verbose:
            print("\n" + "="*80)
            print("Creating Weighted Sentiment Features (Source Reliability)")
            print("="*80)
        df = add_weighted_sentiment_features(df, verbose=verbose)
    
    # Add news intensity and scarcity features
    if verbose:
        print("\n" + "="*80)
        print("Creating News Intensity & Scarcity Features")
        print("="*80)
    df = add_news_intensity_features(df, verbose=verbose)
    
    # Add rolling aggregations and lags for news features
    if verbose:
        print("\n" + "="*80)
        print("📊 Creating Rolling Aggregations and Lags for News Features")
        print("="*80)
    
    df = add_rolling_news_features(df, verbose=verbose)
    
    return df


def add_rolling_news_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Add rolling aggregations and lags for news sentiment features.
    
    Creates:
    - Rolling means/sums for sentiment scores over 1/3/5 day windows
    - Lags (1/2/5 days) for sentiment scores
    - Rolling aggregations for embedding features (mean over windows)
    
    Args:
        df: DataFrame with news features already created
        verbose: Whether to print progress. Default: True
    
    Returns:
        DataFrame with additional rolling and lagged features
    """
    df = df.copy()
    
    # Ensure date column is sorted
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)
    
    # Base numeric news signals (low-dim, recommended)
    sentiment_score_cols = [
        'news_count',
        'no_news',
        'news_finbert_neg',
        'news_finbert_neu',
        'news_finbert_pos',
        'news_finbert_net',
        'news_heuristic_bullish_score',
        'news_heuristic_bearish_score',
        'news_heuristic_net_score',
        # New weighted and intensity features
        'source_weight',
        'news_count_high_impact',
        'news_intensity_score',
        'news_scarcity',
        # Temporal patterns (if they exist)
        'news_count_weekday',
        'news_count_weekend',
        # Category-based aggregations
        'news_heuristic_supply_shock',
        'news_heuristic_geo_shock',
        'news_heuristic_demand_positive',
        'news_heuristic_demand_negative',
        'news_heuristic_production_expansion',
    ]
    # Add weighted sentiment features if they exist
    weighted_cols = [col for col in df.columns if col.endswith('_weighted')]
    sentiment_score_cols.extend(weighted_cols)
    
    # Filter to only existing columns
    sentiment_score_cols = [col for col in sentiment_score_cols if col in df.columns]
    
    # Rolling aggregations for sentiment scores (extended windows for better temporal coverage)
    windows = [1, 3, 5, 7, 10, 14]  # Added 14-day window for longer-term patterns
    for window in windows:
        for col in sentiment_score_cols:
            # Rolling mean
            df[f'{col}_rolling_mean_{window}d'] = df[col].rolling(window=window, min_periods=1).mean().shift(1)
            # Rolling sum
            df[f'{col}_rolling_sum_{window}d'] = df[col].rolling(window=window, min_periods=1).sum().shift(1)
            # Rolling max (captures peak sentiment)
            df[f'{col}_rolling_max_{window}d'] = df[col].rolling(window=window, min_periods=1).max().shift(1)
            # Rolling std (captures sentiment volatility)
            df[f'{col}_rolling_std_{window}d'] = df[col].rolling(window=window, min_periods=1).std().shift(1)
    
    # Lags for sentiment scores (extended set)
    lags = [1, 2, 5, 7]  # Added 7-day lag (weekly pattern)
    for lag in lags:
        for col in sentiment_score_cols:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    # Rolling aggregations for embedding features (mean over windows)
    embed_cols = [col for col in df.columns if col.startswith('news_embed_')]
    for window in windows:
        for col in embed_cols:
            df[f'{col}_rolling_mean_{window}d'] = df[col].rolling(window=window, min_periods=1).mean().shift(1)
    
    if verbose:
        n_new_features = len(windows) * len(sentiment_score_cols) * 2  # mean + sum
        n_new_features += len(lags) * len(sentiment_score_cols)  # lags
        n_new_features += len(windows) * len(embed_cols)  # embedding rolling means
        print(f"✅ Created {n_new_features} rolling/lag features")
        print(f"   - Rolling aggregations: {len(windows)} windows × {len(sentiment_score_cols)} scores × 2 (mean/sum)")
        print(f"   - Lags: {len(lags)} lags × {len(sentiment_score_cols)} scores")
        print(f"   - Embedding rolling means: {len(windows)} windows × {len(embed_cols)} embeddings")
    
    return df


def add_weighted_sentiment_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Add weighted sentiment features based on source reliability.
    
    Source weights:
    - Reuters: 3.0 (most reliable)
    - Bloomberg: 2.5
    - Mining.com: 2.0
    - Others: 1.0
    
    Args:
        df: DataFrame with 'sources' column (list of sources per day) and sentiment features
        verbose: Whether to print progress. Default: True
    
    Returns:
        DataFrame with weighted sentiment features added
    """
    df = df.copy()
    
    # Source reliability weights
    source_weights = {
        'reuters': 3.0,
        'bloomberg': 2.5,
        'mining.com': 2.0,
        'default': 1.0
    }
    
    # Initialize weighted sentiment columns
    sentiment_cols = ['news_finbert_neg', 'news_finbert_neu', 'news_finbert_pos', 'news_finbert_net']
    sentiment_cols = [col for col in sentiment_cols if col in df.columns]
    
    if 'sources' not in df.columns or len(sentiment_cols) == 0:
        if verbose:
            print("⚠️  Cannot create weighted sentiment: missing 'sources' column or sentiment features")
        return df
    
    # Calculate source weights per day
    def get_source_weight(sources_list):
        """Calculate average source weight for a day."""
        if not isinstance(sources_list, list) or len(sources_list) == 0:
            return source_weights['default']
        
        weights = []
        for source in sources_list:
            source_lower = str(source).lower()
            weight = source_weights['default']
            for key, w in source_weights.items():
                if key in source_lower and key != 'default':
                    weight = w
                    break
            weights.append(weight)
        
        return np.mean(weights) if weights else source_weights['default']
    
    df['source_weight'] = df['sources'].apply(get_source_weight)
    
    # Create weighted sentiment features
    for col in sentiment_cols:
        weighted_col = f'{col}_weighted'
        df[weighted_col] = df[col] * df['source_weight']
    
    # Also create weighted heuristic scores if available
    if 'news_heuristic_net_score' in df.columns:
        df['news_heuristic_net_score_weighted'] = df['news_heuristic_net_score'] * df['source_weight']
    
    if verbose:
        print(f"✅ Created weighted sentiment features (based on source reliability)")
        print(f"   Average source weight: {df['source_weight'].mean():.2f}")
    
    return df


def add_news_intensity_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Add news intensity and scarcity features.
    
    Features:
    - news_count_high_impact: Count of news with high-impact keywords (strike, closure, ban, sanctions, war)
    - news_intensity_score: Weighted sum of heuristics (important events have higher weights)
    - news_scarcity: Inverse frequency of occurrence in historical data
    
    Args:
        df: DataFrame with heuristic features and news_count
        verbose: Whether to print progress. Default: True
    
    Returns:
        DataFrame with intensity and scarcity features added
    """
    df = df.copy()
    
    # High-impact heuristic features (events that strongly affect prices)
    high_impact_features = [
        'news_heuristic_strike_labor',
        'news_heuristic_mine_closure',
        'news_heuristic_export_ban',
        'news_heuristic_sanctions',
        'news_heuristic_sanctions_embargo',
        'news_heuristic_war_conflict',
        'news_heuristic_geopolitical_tension',
        'news_heuristic_supply_chain_disruption',
        'news_heuristic_production_cut'
    ]
    
    # Count high-impact news per day
    high_impact_cols = [col for col in high_impact_features if col in df.columns]
    if len(high_impact_cols) > 0:
        df['news_count_high_impact'] = df[high_impact_cols].sum(axis=1).astype(int)
    else:
        df['news_count_high_impact'] = 0
    
    # Intensity score: weighted sum of heuristics (high-impact events weighted more)
    intensity_weights = {
        'strike_labor': 3.0,
        'mine_closure': 3.5,
        'export_ban': 4.0,
        'sanctions': 3.5,
        'sanctions_embargo': 3.5,
        'war_conflict': 4.0,
        'geopolitical_tension': 2.5,
        'supply_chain_disruption': 2.0,
        'production_cut': 3.0,
        'default': 1.0
    }
    
    # Calculate intensity score
    df['news_intensity_score'] = 0.0
    heuristic_cols = [col for col in df.columns if col.startswith('news_heuristic_') and 
                     not col.endswith('_score') and not col.endswith('_weighted')]
    
    for col in heuristic_cols:
        feature_name = col.replace('news_heuristic_', '')
        weight = intensity_weights.get(feature_name, intensity_weights['default'])
        df['news_intensity_score'] += df[col].astype(int) * weight
    
    # Scarcity: inverse frequency (rare events are more valuable signals)
    # Calculate historical frequency of each heuristic
    if 'date' in df.columns and len(df) > 100:
        # Use rolling window to avoid lookahead
        window_size = min(365, len(df) // 2)  # 1 year or half of data
        
        df['news_scarcity'] = 0.0
        for col in heuristic_cols:
            if df[col].sum() > 0:
                # Calculate rolling frequency (avoiding future data)
                rolling_freq = df[col].rolling(window=window_size, min_periods=30).mean().shift(1)
                # Inverse frequency (rare = higher value)
                # Avoid division by zero
                inverse_freq = 1.0 / (rolling_freq + 1e-6)
                # Weight by actual occurrence
                df['news_scarcity'] += df[col].astype(int) * inverse_freq.fillna(0)
        
        # Normalize scarcity by number of heuristics
        if len(heuristic_cols) > 0:
            df['news_scarcity'] = df['news_scarcity'] / len(heuristic_cols)
    else:
        df['news_scarcity'] = df['news_count_high_impact'].astype(float)
    
    if verbose:
        print(f"✅ Created news intensity and scarcity features")
        print(f"   Average high-impact news per day: {df['news_count_high_impact'].mean():.2f}")
        print(f"   Average intensity score: {df['news_intensity_score'].mean():.2f}")
        print(f"   Average scarcity: {df['news_scarcity'].mean():.2f}")
    
    return df


def validate_sentiment_features(
    df: pd.DataFrame,
    price_column: str = 'price',
    verbose: bool = True
) -> dict:
    """
    Validate sentiment features by checking correlation with future returns
    and conditional means.
    
    Args:
        df: DataFrame with price and news features
        price_column: Name of price column. Default: 'price'
        verbose: Whether to print results. Default: True
    
    Returns:
        Dictionary with validation metrics:
        - correlations: dict of correlations between sentiment features and return[t+1]
        - conditional_means: dict of mean returns conditional on sentiment signals
    """
    df = df.copy()
    
    # Calculate returns
    df['return'] = df[price_column].pct_change()
    df['return_next'] = df['return'].shift(-1)  # return[t+1]
    
    # Remove NaN rows
    df_valid = df.dropna(subset=['return_next'])
    
    if len(df_valid) == 0:
        if verbose:
            print("⚠️  No valid data for sentiment validation")
        return {'correlations': {}, 'conditional_means': {}}
    
    results = {
        'correlations': {},
        'conditional_means': {}
    }
    
    # Base news/sentiment signals (low-dim, most useful for feature selection)
    base_cols = [
        'news_count',
        'no_news',
        'news_finbert_neg',
        'news_finbert_neu',
        'news_finbert_pos',
        'news_finbert_net',
        'news_heuristic_bullish_score',
        'news_heuristic_bearish_score',
        'news_heuristic_net_score',
    ]

    # Also check rolling aggregations and lags for these signals
    rolling_cols = [
        col for col in df_valid.columns
        if any(x in col for x in ['rolling_mean_', 'rolling_sum_', '_lag'])
        and (
            col.startswith('news_count')
            or col.startswith('no_news')
            or 'news_finbert' in col
            or 'news_heuristic' in col
        )
    ]

    all_sentiment_cols = [col for col in (base_cols + rolling_cols) if col in df_valid.columns]
    
    if verbose:
        print("\n" + "="*80)
        print("🔍 Validating Sentiment Features")
        print("="*80)
        print(f"\nChecking {len(all_sentiment_cols)} sentiment features...")
    
    # Calculate correlations
    for col in all_sentiment_cols:
        if col in df_valid.columns:
            corr = df_valid[col].corr(df_valid['return_next'])
            if not np.isnan(corr):
                results['correlations'][col] = corr
    
    # Conditional means (simple sanity checks)
    # - heuristic/net scores: >0 / <0
    # - finbert_net: >0 / <0
    cond_cols = [
        'news_heuristic_net_score',
        'news_finbert_net',
    ]
    for col in cond_cols:
        if col in df_valid.columns:
            positive_mask = df_valid[col] > 0
            negative_mask = df_valid[col] < 0

            if positive_mask.sum() > 0:
                results['conditional_means'][f'{col}_positive'] = df_valid.loc[positive_mask, 'return_next'].mean()
            if negative_mask.sum() > 0:
                results['conditional_means'][f'{col}_negative'] = df_valid.loc[negative_mask, 'return_next'].mean()
    
    if verbose:
        print("\n📊 Top Correlations with return[t+1]:")
        sorted_corrs = sorted(results['correlations'].items(), key=lambda x: abs(x[1]), reverse=True)
        for col, corr in sorted_corrs[:10]:
            print(f"   {col:40s}: {corr:7.4f}")
        
        print("\n📈 Conditional Mean Returns:")
        for key, mean_ret in results['conditional_means'].items():
            print(f"   {key:40s}: {mean_ret:7.4f} ({mean_ret*100:.2f}%)")
    
    return results

