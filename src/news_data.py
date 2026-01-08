import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import time
from typing import Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from urllib.parse import urlparse

# Try to import Selenium (optional dependency)
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️  Selenium not available. Install with: pip install selenium")


# Extended list of news sources
RSS_SOURCES = {
    # Direct RSS feeds (Commodity-focused)
    "InvestingNewsNetwork_Copper": (
        "https://investingnews.com/category/daily/copper-news/feed/"
    ),
    "CommodityHQ": (
        "https://commodityhq.com/feed/"
    ),
    "OilPrice": (
        "https://oilprice.com/rss/main"
    ),
    "WorldOil": (
        "https://www.worldoil.com/rss?feed=news"
    ),
    "EconomicTimes_Commodities": (
        "https://economictimes.indiatimes.com/markets/commodities/rssfeeds/1977022931.cms"
    ),
    "Moneycontrol_Commodities": (
        "https://www.moneycontrol.com/rss/commodities.xml"
    ),
    "FinancialExpress_Commodities": (
        "https://www.financialexpress.com/market/commodities/feed/"
    ),
    "BusinessStandard_Markets": (
        "https://www.business-standard.com/rss/markets-106.rss"
    ),
    # Google News RSS (works reliably)
    # Reuters - expanded queries for better coverage
    "Reuters": (
        "https://news.google.com/rss/search?"
        "q=copper+source:reuters&hl=en-US&gl=US&ceid=US:en"
    ),
    "Reuters_Copper_Price": (
        "https://news.google.com/rss/search?"
        "q=(copper+price+OR+LME+copper+OR+copper+futures)+source:reuters&hl=en-US&gl=US&ceid=US:en"
    ),
    "Reuters_Copper_Mining": (
        "https://news.google.com/rss/search?"
        "q=(copper+mining+OR+copper+mine+OR+copper+production)+source:reuters&hl=en-US&gl=US&ceid=US:en"
    ),
    "Reuters_Copper_Supply": (
        "https://news.google.com/rss/search?"
        "q=(copper+supply+OR+copper+demand+OR+copper+inventory)+source:reuters&hl=en-US&gl=US&ceid=US:en"
    ),
    "Reuters_Commodities": (
        "https://news.google.com/rss/search?"
        "q=(copper+OR+metals+OR+commodities)+source:reuters&hl=en-US&gl=US&ceid=US:en"
    ),
    "Reuters_Chile_Copper": (
        "https://news.google.com/rss/search?"
        "q=(copper+Chile+OR+Codelco+OR+Escondida+OR+Collahuasi)+source:reuters&hl=en-US&gl=US&ceid=US:en"
    ),
    "Reuters_China_Copper": (
        "https://news.google.com/rss/search?"
        "q=(copper+China+OR+Chinese+copper+import+OR+Chinese+copper+demand)+source:reuters&hl=en-US&gl=US&ceid=US:en"
    ),
    "Bloomberg": (
        "https://news.google.com/rss/search?"
        "q=copper+source:bloomberg&hl=en-US&gl=US&ceid=US:en"
    ),
    "FinancialTimes": (
        "https://news.google.com/rss/search?"
        "q=copper+source:ft.com&hl=en-US&gl=US&ceid=US:en"
    ),
    "Mining.com": (
        "https://news.google.com/rss/search?"
        "q=copper+source:mining.com&hl=en-US&gl=US&ceid=US:en"
    ),
    # Mining.com - expanded queries for better coverage
    "Mining.com_Copper_News": (
        "https://news.google.com/rss/search?"
        "q=(copper+news+OR+copper+price+OR+copper+mining)+source:mining.com&hl=en-US&gl=US&ceid=US:en"
    ),
    "Mining.com_Copper_Mines": (
        "https://news.google.com/rss/search?"
        "q=(copper+mine+OR+copper+deposit+OR+copper+project)+source:mining.com&hl=en-US&gl=US&ceid=US:en"
    ),
    "Mining.com_Copper_Production": (
        "https://news.google.com/rss/search?"
        "q=(copper+production+OR+copper+output+OR+copper+capacity)+source:mining.com&hl=en-US&gl=US&ceid=US:en"
    ),
    "Mining.com_Major_Mines": (
        "https://news.google.com/rss/search?"
        "q=(Escondida+OR+Collahuasi+OR+Cerro+Verde+OR+Grasberg+OR+Antamina+OR+Buenavista)+source:mining.com&hl=en-US&gl=US&ceid=US:en"
    ),
    "Mining.com_Miners": (
        "https://news.google.com/rss/search?"
        "q=(BHP+OR+Codelco+OR+Freeport+OR+Grupo+Mexico+OR+Glencore+OR+Rio+Tinto)+copper+source:mining.com&hl=en-US&gl=US&ceid=US:en"
    ),
    # Additional sources
    "WSJ": (
        "https://news.google.com/rss/search?"
        "q=copper+source:wsj.com&hl=en-US&gl=US&ceid=US:en"
    ),
    "MarketWatch": (
        "https://news.google.com/rss/search?"
        "q=copper+source:marketwatch.com&hl=en-US&gl=US&ceid=US:en"
    ),
    "Investing.com": (
        "https://news.google.com/rss/search?"
        "q=copper+source:investing.com&hl=en-US&gl=US&ceid=US:en"
    ),
    "Kitco": (
        "https://news.google.com/rss/search?"
        "q=copper+source:kitco.com&hl=en-US&gl=US&ceid=US:en"
    ),
    "MetalBulletin": (
        "https://news.google.com/rss/search?"
        "q=copper+source:metalbulletin.com&hl=en-US&gl=US&ceid=US:en"
    ),
    # General keyword search (may yield more results)
    "CopperGeneral": (
        "https://news.google.com/rss/search?"
        "q=copper+price+OR+copper+mining+OR+copper+supply+OR+LME+copper&hl=en-US&gl=US&ceid=US:en"
    ),
    # Geopolitical and war-related sources
    "Copper_Geopolitics": (
        "https://news.google.com/rss/search?"
        "q=(copper+OR+commodity+OR+mining)+AND+(war+OR+conflict+OR+geopolitical+OR+sanctions+OR+embargo+OR+trade+war)&hl=en-US&gl=US&ceid=US:en"
    ),
    "Copper_Ukraine": (
        "https://news.google.com/rss/search?"
        "q=(copper+OR+commodity)+AND+(Ukraine+OR+Russia+OR+Russian)&hl=en-US&gl=US&ceid=US:en"
    ),
    "Copper_China_Trade": (
        "https://news.google.com/rss/search?"
        "q=(copper+OR+commodity)+AND+(China+OR+Chinese)+AND+(trade+OR+tariff+OR+export+OR+import)&hl=en-US&gl=US&ceid=US:en"
    ),
    "Copper_Supply_Chain": (
        "https://news.google.com/rss/search?"
        "q=(copper+OR+mining)+AND+(supply+chain+OR+logistics+OR+transport+OR+shipping+OR+port)&hl=en-US&gl=US&ceid=US:en"
    ),
    # Major copper mines (mine-specific Google News queries)
    "Copper_Escondida": (
        "https://news.google.com/rss/search?"
        "q=Escondida+mine+copper+Chile+BHP&hl=en-US&gl=US&ceid=US:en"
    ),
    "Copper_Collahuasi": (
        "https://news.google.com/rss/search?"
        "q=Collahuasi+mine+copper+Chile&hl=en-US&gl=US&ceid=US:en"
    ),
    "Copper_CerroVerde": (
        "https://news.google.com/rss/search?"
        "q=Cerro+Verde+mine+copper+Peru+Freeport&hl=en-US&gl=US&ceid=US:en"
    ),
    "Copper_Buenavista": (
        "https://news.google.com/rss/search?"
        "q=Buenavista+del+Cobre+mine+copper+Mexico+Grupo+Mexico&hl=en-US&gl=US&ceid=US:en"
    ),
    "Copper_KamoaKakula": (
        "https://news.google.com/rss/search?"
        "q=Kamoa+Kakula+mine+copper+DRC+Congo&hl=en-US&gl=US&ceid=US:en"
    ),
    "Copper_Grasberg": (
        "https://news.google.com/rss/search?"
        "q=Grasberg+mine+copper+Indonesia+Freeport&hl=en-US&gl=US&ceid=US:en"
    ),
    "Copper_Antamina": (
        "https://news.google.com/rss/search?"
        "q=Antamina+mine+copper+Peru&hl=en-US&gl=US&ceid=US:en"
    ),
    "Copper_Morenci": (
        "https://news.google.com/rss/search?"
        "q=Morenci+mine+copper+Arizona+Freeport&hl=en-US&gl=US&ceid=US:en"
    ),
    "Copper_ElTeniente": (
        "https://news.google.com/rss/search?"
        "q=El+Teniente+mine+copper+Chile+Codelco&hl=en-US&gl=US&ceid=US:en"
    ),
    "Copper_Chuquicamata": (
        "https://news.google.com/rss/search?"
        "q=Chuquicamata+mine+copper+Chile+Codelco&hl=en-US&gl=US&ceid=US:en"
    ),
    # Major producers/operators
    "Copper_BHP": (
        "https://news.google.com/rss/search?"
        "q=BHP+copper+production+Escondida&hl=en-US&gl=US&ceid=US:en"
    ),
    "Copper_Codelco": (
        "https://news.google.com/rss/search?"
        "q=Codelco+copper+Chile+(El+Teniente+OR+Chuquicamata)&hl=en-US&gl=US&ceid=US:en"
    ),
    "Copper_Freeport": (
        "https://news.google.com/rss/search?"
        "q=Freeport+McMoRan+copper+(Grasberg+OR+Morenci)&hl=en-US&gl=US&ceid=US:en"
    ),
    "Copper_GrupoMexico": (
        "https://news.google.com/rss/search?"
        "q=Grupo+Mexico+copper+(Buenavista)&hl=en-US&gl=US&ceid=US:en"
    ),
    # Additional commodity news sources
    "CNBC_Commodities": (
        "https://news.google.com/rss/search?"
        "q=copper+source:cnbc.com&hl=en-US&gl=US&ceid=US:en"
    ),
    "YahooFinance_Commodities": (
        "https://news.google.com/rss/search?"
        "q=copper+source:yahoo.com&hl=en-US&gl=US&ceid=US:en"
    ),
    "SeekingAlpha_Commodities": (
        "https://news.google.com/rss/search?"
        "q=copper+source:seekingalpha.com&hl=en-US&gl=US&ceid=US:en"
    ),
    "ZeroHedge_Commodities": (
        "https://news.google.com/rss/search?"
        "q=copper+source:zerohedge.com&hl=en-US&gl=US&ceid=US:en"
    ),
    "S&P_Global_Commodities": (
        "https://news.google.com/rss/search?"
        "q=copper+source:spglobal.com&hl=en-US&gl=US&ceid=US:en"
    ),
    "Argus_Media": (
        "https://news.google.com/rss/search?"
        "q=copper+source:argusmedia.com&hl=en-US&gl=US&ceid=US:en"
    ),
    "FastMarkets": (
        "https://news.google.com/rss/search?"
        "q=copper+source:fastmarkets.com&hl=en-US&gl=US&ceid=US:en"
    ),
    # Mining industry sources
    "MiningWeekly": (
        "https://news.google.com/rss/search?"
        "q=copper+source:miningweekly.com&hl=en-US&gl=US&ceid=US:en"
    ),
    "MiningJournal": (
        "https://news.google.com/rss/search?"
        "q=copper+source:mining-journal.com&hl=en-US&gl=US&ceid=US:en"
    ),
    "NorthernMiner": (
        "https://news.google.com/rss/search?"
        "q=copper+source:northernminer.com&hl=en-US&gl=US&ceid=US:en"
    ),
}


def extract_url_from_google_news_rss(google_news_url: str, timeout: int = 10) -> Optional[str]:
    """
    Extract the real article URL from Google News RSS feed by finding the item
    with matching link and extracting URL from its description.
    
    Args:
        google_news_url: Google News article URL (e.g., https://news.google.com/rss/articles/...)
        timeout: Request timeout in seconds
    
    Returns:
        Real article URL if found in RSS description, None otherwise
    """
    if not google_news_url or 'news.google.com' not in google_news_url:
        return None
    
    try:
        # Extract article ID from Google News URL
        # Format: https://news.google.com/rss/articles/ARTICLE_ID?oc=5
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(google_news_url)
        article_id = parsed.path.split('/')[-1] if parsed.path else None
        
        if not article_id:
            return None
        
        # Try to construct RSS feed URL directly from the article URL
        # Google News article URLs can sometimes be accessed via RSS by constructing the feed URL
        # Format: https://news.google.com/rss/articles/ARTICLE_ID
        # We can try to get the article's RSS feed directly
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        
        # Method 1: Try to access the article's RSS feed directly
        # Some Google News articles have RSS feeds at: https://news.google.com/rss/articles/ARTICLE_ID
        rss_article_url = f"https://news.google.com/rss/articles/{article_id}"
        try:
            response = requests.get(rss_article_url, timeout=timeout, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "xml")
                items = soup.find_all("item")
                for item in items:
                    link = item.link.text if item.link else None
                    if link and article_id in link:
                        desc_raw = item.description.text if item.description else ""
                        if desc_raw:
                            url_pattern = r'https?://[^\s<>"]+'
                            urls = re.findall(url_pattern, desc_raw)
                            for found_url in urls:
                                if 'news.google.com' not in found_url and 'google.com' not in found_url:
                                    return found_url
        except Exception:
            pass
        
        # Method 2: Try to find in common Google News RSS feeds (for recent articles)
        rss_feeds = [
            "https://news.google.com/rss/search?q=copper&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=copper+price&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=copper+mining&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=copper+supply&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=LME+copper&hl=en-US&gl=US&ceid=US:en",
        ]
        
        for rss_url in rss_feeds:
            try:
                response = requests.get(rss_url, timeout=timeout, headers=headers)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, "xml")
                items = soup.find_all("item")
                
                # Look for item with matching link
                for item in items:
                    link = item.link.text if item.link else None
                    if link and article_id in link:
                        # Found matching item, extract URL from description
                        desc_raw = item.description.text if item.description else ""
                        if desc_raw:
                            url_pattern = r'https?://[^\s<>"]+'
                            urls = re.findall(url_pattern, desc_raw)
                            for found_url in urls:
                                if 'news.google.com' not in found_url and 'google.com' not in found_url:
                                    return found_url
            except Exception:
                continue
        
        return None
    except Exception:
        return None


def get_final_url(url: str, timeout: int = 10) -> tuple[str, Optional[str]]:
    """
    Follow redirects and get the final URL and domain.
    Handles Google News redirects that may lead to consent pages.
    
    Args:
        url: Initial URL (may be a Google News redirect)
        timeout: Request timeout in seconds
    
    Returns:
        Tuple of (final_url, domain) or (original_url, None) if failed
    """
    if not url:
        return url, None
    
    # Enhanced headers to mimic a real browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    
    try:
        # First try HEAD request
        response = requests.head(url, allow_redirects=True, timeout=timeout, headers=headers)
        final_url = response.url
        
        # Check if we hit a consent page
        if 'consent.google.com' in final_url or 'accounts.google.com' in final_url:
            # First, try to extract URL from RSS description (most reliable for Google News)
            if 'news.google.com' in url:
                rss_url = extract_url_from_google_news_rss(url, timeout)
                if rss_url:
                    domain = extract_domain_from_url(rss_url)
                    return rss_url, domain
            
            # Try Selenium first if available (better at handling consent pages)
            if SELENIUM_AVAILABLE:
                try:
                    selenium_result = _extract_url_with_selenium(url, timeout)
                    if selenium_result[0] != url and 'consent' not in selenium_result[0]:
                        return selenium_result
                except Exception:
                    pass  # Fall back to regular method
            
            # Try to extract URL from consent page's continue parameter
            parsed_consent = urlparse(final_url)
            from urllib.parse import parse_qs
            params = parse_qs(parsed_consent.query)
            if 'continue' in params:
                continue_url = params['continue'][0]
                # Decode URL-encoded characters
                from urllib.parse import unquote
                continue_url = unquote(continue_url)
                # If continue URL is still a Google News URL, try to extract from it
                if 'news.google.com' in continue_url:
                    return _extract_url_from_google_news_page(continue_url, timeout, headers)
                else:
                    # This might be the real URL
                    domain = extract_domain_from_url(continue_url)
                    return continue_url, domain
            # If we hit consent page, try to get the page and extract real URL from HTML
            return _extract_url_from_google_news_page(url, timeout, headers)
        
        # Extract domain from final URL
        parsed = urlparse(final_url)
        domain = parsed.netloc
        
        # Clean domain (remove www.)
        if domain.startswith('www.'):
            domain = domain[4:]
        
        return final_url, domain
    except Exception:
        # If HEAD fails, try GET
        try:
            response = requests.get(url, allow_redirects=True, timeout=timeout, 
                                  headers=headers, stream=True)
            final_url = response.url
            
            # Check if we hit a consent page
            if 'consent.google.com' in final_url or 'accounts.google.com' in final_url:
                # First, try to extract URL from RSS description (most reliable for Google News)
                if 'news.google.com' in url:
                    rss_url = extract_url_from_google_news_rss(url, timeout)
                    if rss_url:
                        domain = extract_domain_from_url(rss_url)
                        return rss_url, domain
                
                # Try Selenium first if available (better at handling consent pages)
                if SELENIUM_AVAILABLE:
                    try:
                        selenium_result = _extract_url_with_selenium(url, timeout)
                        if selenium_result[0] != url and 'consent' not in selenium_result[0]:
                            return selenium_result
                    except Exception:
                        pass  # Fall back to regular method
                
                # Try to extract URL from consent page's continue parameter
                parsed_consent = urlparse(final_url)
                from urllib.parse import parse_qs, unquote
                params = parse_qs(parsed_consent.query)
                if 'continue' in params:
                    continue_url = params['continue'][0]
                    continue_url = unquote(continue_url)
                    # If continue URL is still a Google News URL, try to extract from it
                    if 'news.google.com' in continue_url:
                        return _extract_url_from_google_news_page(continue_url, timeout, headers)
                    else:
                        # This might be the real URL
                        domain = extract_domain_from_url(continue_url)
                        return continue_url, domain
                # Try to extract from HTML
                return _extract_url_from_google_news_page(url, timeout, headers)
            
            parsed = urlparse(final_url)
            domain = parsed.netloc
            if domain.startswith('www.'):
                domain = domain[4:]
            return final_url, domain
        except Exception:
            # If both fail, return original URL
            return url, None


def _extract_url_with_selenium(url: str, timeout: int = 15) -> tuple[str, Optional[str]]:
    """
    Extract the real article URL from Google News using Selenium.
    Selenium can handle JavaScript and consent pages better than requests.
    
    Args:
        url: Google News URL
        timeout: Maximum wait time in seconds
    
    Returns:
        Tuple of (final_url, domain)
    """
    if not SELENIUM_AVAILABLE:
        return url, extract_domain_from_url(url)
    
    driver = None
    try:
        # Setup Chrome options
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Run in background
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # Create driver
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(timeout)
        
        # Navigate to URL
        driver.get(url)
        
        # Wait a bit for page to load
        time.sleep(2)
        
        # Get current URL (after redirects)
        current_url = driver.current_url
        
        # Check if we got redirected to a real article
        if 'news.google.com' not in current_url and 'consent.google.com' not in current_url:
            domain = extract_domain_from_url(current_url)
            return current_url, domain
        
        # Try to find article link in the page
        try:
            # Look for article links
            article_links = driver.find_elements(By.TAG_NAME, "a")
            for link in article_links:
                href = link.get_attribute('href')
                if href and href.startswith('http'):
                    if ('news.google.com' not in href and 
                        'consent' not in href and 
                        'google.com' not in href):
                        # Check if it looks like a news article URL
                        if any(pattern in href for pattern in ['/article', '/news/', '/story', '/business/', '/markets/']):
                            domain = extract_domain_from_url(href)
                            if domain:
                                return href, domain
        except Exception:
            pass
        
        # Try to get canonical URL from meta tags
        try:
            canonical = driver.find_element(By.XPATH, "//link[@rel='canonical']")
            canonical_url = canonical.get_attribute('href')
            if canonical_url and 'news.google.com' not in canonical_url:
                domain = extract_domain_from_url(canonical_url)
                return canonical_url, domain
        except Exception:
            pass
        
        # Fallback: return current URL
        domain = extract_domain_from_url(current_url)
        return current_url, domain
        
    except TimeoutException:
        return url, extract_domain_from_url(url)
    except Exception as e:
        print(f"⚠️  Selenium error: {e}")
        return url, extract_domain_from_url(url)
    finally:
        if driver:
            driver.quit()


def _extract_url_from_google_news_page(url: str, timeout: int, headers: dict) -> tuple[str, Optional[str]]:
    """
    Extract the real article URL from Google News page HTML.
    Google News pages contain the real URL in various places.
    
    Args:
        url: Google News URL
        timeout: Request timeout
        headers: Request headers
    
    Returns:
        Tuple of (final_url, domain)
    """
    try:
        # Get the page content
        response = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        
        # If we still hit consent page, try to find URL in the page
        if 'consent.google.com' in response.url or 'accounts.google.com' in response.url:
            # Try to extract URL from consent page's continue parameter first
            parsed_consent = urlparse(response.url)
            from urllib.parse import parse_qs, unquote
            params = parse_qs(parsed_consent.query)
            if 'continue' in params:
                continue_url = params['continue'][0]
                continue_url = unquote(continue_url)
                # If continue URL is still a Google News URL, we need to parse the HTML
                if 'news.google.com' not in continue_url:
                    # This might be the real URL
                    domain = extract_domain_from_url(continue_url)
                    return continue_url, domain
                else:
                    # Continue URL is still Google News, try to get that page
                    # But this will likely also hit consent, so we'll try to parse HTML
                    pass
            
            # Try to extract URL from the page content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for canonical link
            canonical = soup.find('link', rel='canonical')
            if canonical and canonical.get('href'):
                final_url = canonical['href']
                domain = extract_domain_from_url(final_url)
                if domain and 'news.google.com' not in domain and 'consent' not in domain:
                    return final_url, domain
            
            # Look for og:url meta tag
            og_url = soup.find('meta', property='og:url')
            if og_url and og_url.get('content'):
                final_url = og_url['content']
                domain = extract_domain_from_url(final_url)
                if domain and 'news.google.com' not in domain and 'consent' not in domain:
                    return final_url, domain
            
            # Look for article URL in script tags or data attributes
            # Google News sometimes embeds the URL in JavaScript
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string:
                    # Look for URL patterns in JavaScript
                    url_pattern = r'https?://[^\s"\'<>)]+'
                    urls = re.findall(url_pattern, script.string)
                    for found_url in urls:
                        # Clean URL (remove trailing characters)
                        found_url = found_url.rstrip('.,;:!?)\'"')
                        if ('news.google.com' not in found_url and 
                            'consent' not in found_url and 
                            'google.com' not in found_url and
                            len(found_url) > 20):  # Filter out very short URLs
                            domain = extract_domain_from_url(found_url)
                            if domain and domain != 'news.google.com' and 'consent' not in domain:
                                return found_url, domain
            
            # Look for links in the page that might be the article
            links = soup.find_all('a', href=True)
            for link in links:
                href = link.get('href', '')
                if href.startswith('http') and 'news.google.com' not in href and 'consent' not in href:
                    domain = extract_domain_from_url(href)
                    if domain and 'news.google.com' not in domain and 'consent' not in domain:
                        return href, domain
        
        # If we got redirected to a real page, use that
        if 'news.google.com' not in response.url and 'consent' not in response.url:
            final_url = response.url
            domain = extract_domain_from_url(final_url)
            return final_url, domain
        
        # Try to parse the HTML for article links even if we didn't hit consent
        # Sometimes Google News shows the article URL in the page
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for article links - Google News often has them in specific divs
        article_links = soup.find_all('a', href=True)
        for link in article_links:
            href = link.get('href', '')
            # Look for external article links
            if (href.startswith('http') and 
                'news.google.com' not in href and 
                'consent' not in href and
                'google.com' not in href):
                domain = extract_domain_from_url(href)
                if domain and 'news.google.com' not in domain and 'consent' not in domain:
                    # Check if this looks like a news article URL (has common news patterns)
                    if any(pattern in href for pattern in ['/article', '/news/', '/story', '/business/', '/markets/']):
                        return href, domain
        
        # Fallback: return original URL
        return url, extract_domain_from_url(url)
    except Exception:
        return url, extract_domain_from_url(url)


def extract_domain_from_url(url: str) -> Optional[str]:
    """
    Extract domain name from URL.
    
    Args:
        url: URL string
    
    Returns:
        Domain name or None
    """
    if not url:
        return None
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except Exception:
        return None


def clean_text(text: str) -> str:
    """
    Remove Google News artifacts, URLs, tracking tokens.
    """
    if not text:
        return None

    # remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # remove source labels
    text = re.sub(
        r"\b(Google News|Reuters|Bloomberg|Financial Times|FT|Mining\.com)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # remove long tracking tokens
    text = re.sub(r"\b[A-Za-z0-9_-]{20,}\b", "", text)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_copper_relevant(title: str, text: str) -> bool:
    """
    Check if an article is relevant to copper/commodity news.
    
    Args:
        title: Article title
        text: Article text/description
    
    Returns:
        True if article is relevant to copper, False otherwise
    """
    if not title and not text:
        return False
    
    # Combine title and text for checking
    combined_text = f"{title} {text}".lower()
    
    # Key terms that indicate copper relevance
    copper_keywords = [
        'copper',
        'lme',  # London Metal Exchange
        'codelco',  # Major copper producer
        'bhp',  # Mining company
        'rio tinto',  # Mining company
        'freeport',  # Freeport-McMoRan
        'glencore',  # Mining/trading company
        'antofagasta',  # Copper mining company
        'anglo american',  # Mining company
        # Major copper mines and operators
        'escondida',
        'collahuasi',
        'cerro verde',
        'buenavista',
        'kamoa',
        'kakula',
        'kamoa-kakula',
        'grasberg',
        'antamina',
        'morenci',
        'el teniente',
        'chuquicamata',
        'grupo mexico',
        'freeport-mcmoran',
        'copper mine',
        'copper price',
        'copper production',
        'copper supply',
        'copper demand',
        'copper futures',
        'copper market',
        'copper deficit',
        'copper surplus',
        'copper inventory',
        'copper warehouse',
        'copper strike',
        'chile copper',  # Major producer
        'peru copper',  # Major producer
        'drc copper',  # Democratic Republic of Congo
        'zambia copper',
        'copper cathode',
        'copper concentrate',
        # Geopolitical and war-related terms
        'copper sanctions',
        'copper embargo',
        'copper export ban',
        'copper trade war',
        'copper supply chain',
        'copper logistics',
        'copper shipping',
        'copper port',
        'copper transport',
        'copper disruption',
        'copper conflict',
        'copper war',
        'copper geopolitical',
        # Major producing regions (geopolitical hotspots)
        'chile mining',
        'peru mining',
        'congo mining',
        'zambia mining',
        'russia mining',
        'ukraine mining',
        'china mining',
    ]
    
    # Check if any keyword appears in the text
    for keyword in copper_keywords:
        if keyword in combined_text:
            return True
    
    # If no copper keywords found, it's likely not relevant
    return False


def _resolve_link_redirect(link: str) -> tuple[str, Optional[str]]:
    """
    Resolve a single link redirect to get final URL and domain.
    Helper function for parallel execution.
    
    Args:
        link: Initial URL (may be Google News redirect)
    
    Returns:
        Tuple of (final_url, domain)
    """
    try:
        final_url, domain = get_final_url(link)
        return final_url, domain
    except Exception:
        return link, extract_domain_from_url(link)


def fetch_direct_rss(
    source_name: str,
    url: str,
    retry_count: int = 3,
    delay: float = 1.0,
    filter_copper: bool = True,  # Filter for copper-relevant articles
    query: Optional[str] = None
) -> list[dict]:
    """
    Fetch news from direct RSS feed (not Google News) with retry logic.
    Uses feedparser if available, otherwise falls back to BeautifulSoup.
    
    Args:
        source_name: Name of the news source
        url: RSS feed URL
        retry_count: Number of retry attempts
        delay: Delay between retries in seconds
        filter_copper: If True, only return articles relevant to copper
        query: Query/search term used to fetch this data (for tracking)
    
    Returns:
        List of news records
    """
    # Try to use feedparser (cleaner API)
    try:
        import feedparser
        USE_FEEDPARSER = True
    except ImportError:
        USE_FEEDPARSER = False
    
    for attempt in range(retry_count):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
            
            if USE_FEEDPARSER:
                # Use requests to fetch RSS (handles SSL better than feedparser's urllib)
                try:
                    response = requests.get(url, timeout=30, headers=headers, verify=True)
                    response.raise_for_status()
                except requests.exceptions.SSLError:
                    # If SSL verification fails, try without verification (less secure but works)
                    import urllib3
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                    response = requests.get(url, timeout=30, headers=headers, verify=False)
                    response.raise_for_status()
                
                # Parse with feedparser from response content
                feed = feedparser.parse(response.content)
                if feed.bozo and feed.bozo_exception:
                    # Try to continue anyway if it's just a minor parsing issue
                    if 'SSL' in str(feed.bozo_exception) or 'certificate' in str(feed.bozo_exception).lower():
                        # SSL error already handled by requests, ignore feedparser warning
                        pass
                    elif not feed.entries:
                        # Only raise if we have no entries AND it's not SSL-related
                        raise Exception(f"RSS parse error: {feed.bozo_exception}")
                
                records = []
                for entry in feed.entries:
                    title = entry.get('title', '')
                    link = entry.get('link', '')
                    # Parse publication date
                    pub_date_raw = None
                    if hasattr(entry, 'published'):
                        pub_date_raw = entry.published
                    elif hasattr(entry, 'updated'):
                        pub_date_raw = entry.updated
                    elif hasattr(entry, 'published_parsed') and entry.published_parsed:
                        # feedparser provides parsed time struct
                        import time as time_module
                        pub_date_raw = time_module.strftime('%a, %d %b %Y %H:%M:%S %z', entry.published_parsed)
                    
                    # Get description/summary
                    desc_raw = ''
                    if hasattr(entry, 'summary'):
                        desc_raw = entry.summary
                    elif hasattr(entry, 'description'):
                        desc_raw = entry.description
                    
                    # Filter for copper relevance if requested
                    if filter_copper and not is_copper_relevant(title, desc_raw):
                        continue
                    
                    # Extract domain from link
                    domain = extract_domain_from_url(link)
                    
                    # Parse date string to datetime
                    date_obj = None
                    if pub_date_raw:
                        try:
                            # feedparser provides parsed time struct in entry.published_parsed
                            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                                import time as time_module
                                date_obj = datetime(*entry.published_parsed[:6])
                            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                                import time as time_module
                                date_obj = datetime(*entry.updated_parsed[:6])
                            else:
                                # Fallback: try dateutil parser
                                from dateutil import parser
                                date_obj = parser.parse(pub_date_raw)
                        except Exception:
                            # If parsing fails, try manual parsing
                            try:
                                date_obj = datetime.strptime(pub_date_raw[:25], '%a, %d %b %Y %H:%M:%S')
                            except Exception:
                                pass
                    
                    record = {
                        'title': clean_text(title),
                        'text': clean_text(desc_raw),
                        'source': source_name,
                        'publication': domain or source_name,
                        'link': link,
                        'article_url': link,
                        'original_link': link,
                        'domain': domain or 'unknown',
                        'query': query or source_name,
                        'date': date_obj if date_obj else datetime.now(),
                    }
                    records.append(record)
                
                return records
            else:
                # Fallback to BeautifulSoup
                try:
                    response = requests.get(url, timeout=30, headers=headers, verify=True)
                    response.raise_for_status()
                except requests.exceptions.SSLError:
                    # If SSL verification fails, try without verification (less secure but works)
                    import urllib3
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                    response = requests.get(url, timeout=30, headers=headers, verify=False)
                    response.raise_for_status()
                
                soup = BeautifulSoup(response.text, "xml")
                
                records = []
                items = soup.find_all("item")
                
                for item in items:
                    title_raw = item.title.text if item.title else ""
                    link = item.link.text if item.link else ""
                    pub_date_raw = item.pubDate.text if item.pubDate else None
                    desc_raw = item.description.text if item.description else ""
                    
                    # Filter for copper relevance if requested
                    if filter_copper and not is_copper_relevant(title_raw, desc_raw):
                        continue
                    
                    domain = extract_domain_from_url(link)
                    
                    # Parse date string to datetime
                    date_obj = None
                    if pub_date_raw:
                        try:
                            from dateutil import parser
                            date_obj = parser.parse(pub_date_raw)
                        except Exception:
                            try:
                                date_obj = datetime.strptime(pub_date_raw[:25], '%a, %d %b %Y %H:%M:%S')
                            except Exception:
                                pass
                    
                    record = {
                        'title': clean_text(title_raw),
                        'text': clean_text(desc_raw),
                        'source': source_name,
                        'publication': domain or source_name,
                        'link': link,
                        'article_url': link,
                        'original_link': link,
                        'domain': domain or 'unknown',
                        'query': query or source_name,
                        'date': date_obj if date_obj else datetime.now(),
                    }
                    records.append(record)
                
                return records
                
        except Exception as e:
            if attempt < retry_count - 1:
                time.sleep(delay * (attempt + 1))
                continue
            else:
                print(f"  ⚠️  Failed to fetch {source_name} after {retry_count} attempts: {e}")
                return []
    
    return []


def fetch_google_news_rss(
    source_name: str, 
    url: str, 
    retry_count: int = 3,
    delay: float = 1.0,
    resolve_redirects: bool = False,  # Disabled by default - will be done in post-processing
    max_redirect_workers: int = 10,
    query: Optional[str] = None  # Query/search term used to fetch this data
) -> list[dict]:
    """
    Fetch news from Google News RSS feed with retry logic.
    
    Args:
        source_name: Name of the news source
        url: RSS feed URL
        retry_count: Number of retry attempts
        delay: Delay between retries in seconds
        query: Query/search term used to fetch this data (for tracking)
    
    Returns:
        List of news records
    """
    for attempt in range(retry_count):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
            response = requests.get(url, timeout=30, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "xml")

            records = []
            items = list(soup.find_all("item"))

            # First pass: collect all items
            items_data = []
            for item in items:
                title_raw = item.title.text if item.title else ""
                link = item.link.text if item.link else None
                pub_date_raw = item.pubDate.text if item.pubDate else None
                desc_raw = item.description.text if item.description else ""
                
                # Try to extract real URL from description (Google News sometimes includes it)
                extracted_url = None
                if desc_raw:
                    # Look for URLs in description
                    url_pattern = r'https?://[^\s<>"]+'
                    urls = re.findall(url_pattern, desc_raw)
                    for found_url in urls:
                        if 'news.google.com' not in found_url and 'google.com' not in found_url:
                            extracted_url = found_url
                            break
                
                items_data.append({
                    'title_raw': title_raw,
                    'link': link,
                    'pub_date_raw': pub_date_raw,
                    'desc_raw': desc_raw,
                    'extracted_url': extracted_url  # URL found in description
                })
            
            # Resolve redirects in parallel if enabled
            link_to_final = {}
            link_to_domain = {}
            if resolve_redirects and items_data:
                links_to_resolve = [item['link'] for item in items_data if item['link']]
                if links_to_resolve:
                    print(f"  🔗 Resolving {len(links_to_resolve)} link redirects...")
                    with ThreadPoolExecutor(max_workers=max_redirect_workers) as executor:
                        future_to_link = {
                            executor.submit(_resolve_link_redirect, link): link
                            for link in links_to_resolve
                        }
                        for future in as_completed(future_to_link):
                            original_link = future_to_link[future]
                            try:
                                final_url, domain = future.result()
                                link_to_final[original_link] = final_url
                                if domain:
                                    link_to_domain[original_link] = domain
                            except Exception:
                                # If redirect resolution fails, use original link
                                link_to_final[original_link] = original_link
                                domain = extract_domain_from_url(original_link)
                                if domain:
                                    link_to_domain[original_link] = domain

            # Second pass: process items with resolved links
            for item_data in items_data:
                title_raw = item_data['title_raw']
                link = item_data['link']
                pub_date_raw = item_data['pub_date_raw']
                desc_raw = item_data['desc_raw']
                extracted_url = item_data['extracted_url']  # Get extracted URL from item_data
                
                # Try to extract actual source from description first
                actual_source = source_name  # Default to provided source_name
                publication_name = None  # Real publication name (e.g., "The Economic Times")
                final_url = link  # Default to original link
                real_domain = None
                
                if desc_raw:
                    # Try to extract source from description (often contains "Source Name" or domain)
                    desc_soup = BeautifulSoup(desc_raw, "html.parser")
                    desc_text = desc_soup.get_text(" ")
                    
                    # Helper function to check if a potential name is actually the article title
                    def is_likely_title(potential_name, article_title):
                        """Check if potential_name is likely the article title, not publication name"""
                        if not potential_name or not article_title:
                            return False
                        # Normalize for comparison
                        potential_lower = potential_name.lower().strip()
                        title_lower = article_title.lower().strip()
                        # If they're very similar, it's likely the title
                        if potential_lower in title_lower or title_lower in potential_lower:
                            return True
                        # If potential name is very long (like a title), it's probably the title
                        if len(potential_name) > 60:
                            return True
                        return False
                    
                    # First, try to extract from HTML structure
                    # Google News RSS often has publication name in specific HTML tags or at the end
                    # Look for font tags with small text (often publication names)
                    source_tags = desc_soup.find_all(['font', 'span', 'div', 'b', 'strong', 'a'])
                    for tag in source_tags:
                        tag_text = tag.get_text(strip=True)
                        if tag_text and len(tag_text) > 2 and len(tag_text) < 100:
                            # Skip if it looks like the article title
                            if is_likely_title(tag_text, title_raw):
                                continue
                            
                            # Check if it looks like a publication name
                            if tag_text[0].isupper() and not tag_text.lower().startswith(('by ', 'from ', 'via ', 'posted ', 'updated ', 'source: ')):
                                # Check for separators
                                if '|' in tag_text:
                                    # Handle "Source Name | Additional Info" format
                                    parts = [p.strip() for p in tag_text.split('|')]
                                    if len(parts) >= 2 and all(len(p) > 1 for p in parts[:2]):
                                        potential_name = ' | '.join(parts[:2])
                                        if not is_likely_title(potential_name, title_raw) and len(potential_name) > 3:
                                            publication_name = potential_name
                                            break
                                elif len(tag_text) > 2 and len(tag_text) < 80:
                                    # Short text that's not the title - likely publication name
                                    if not is_likely_title(tag_text, title_raw):
                                        publication_name = tag_text
                                        break
                    
                    # Try to extract from description text
                    # Google News RSS formats:
                    # 1. "Title - Publication Name" (title first, then publication)
                    # 2. "Publication Name - Title" (publication first, then title)
                    # 3. "Title: Publication Name" or "Publication Name: Title"
                    if not publication_name:
                        # Try format: "Title - Publication Name" (extract from end)
                        # Look for pattern where publication name comes after the title
                        end_patterns = [
                            r'[–—\-:•]\s*([^–—\-:•]+?)\s*$',  # " - Publication Name" at the end
                            r'[–—\-:•]\s*([A-Z][^–—\-:•]{2,50}?)\s*$',  # Publication name at end (capitalized, reasonable length)
                        ]
                        
                        for pattern in end_patterns:
                            match = re.search(pattern, desc_text.strip())
                            if match:
                                potential_name = match.group(1).strip()
                                potential_name = re.sub(r'\s+', ' ', potential_name).strip()
                                
                                # Check if it's not the title and is reasonable length
                                if (len(potential_name) > 2 and len(potential_name) < 80 and 
                                    not is_likely_title(potential_name, title_raw) and
                                    potential_name not in ['The', 'A', 'An', 'By', 'From', 'Via']):
                                    publication_name = potential_name
                                    break
                        
                        # If not found at end, try beginning: "Publication Name - Title"
                        if not publication_name:
                            start_patterns = [
                                r'^([^–—\-:•|]+(?:\s*\|\s*[^–—\-:•|]+)?)\s*[–—\-:•]\s*',  # "Publication Name | Info -"
                                r'^([A-Z][^–—\-:•]{2,50}?)\s*[–—\-:•]\s*',  # "Publication Name -" (reasonable length)
                            ]
                            
                            for pattern in start_patterns:
                                match = re.match(pattern, desc_text.strip())
                                if match:
                                    potential_name = match.group(1).strip()
                                    potential_name = re.sub(r'\s+', ' ', potential_name).strip()
                                    
                                    # Check if it's not the title
                                    if (len(potential_name) > 2 and len(potential_name) < 80 and
                                        not is_likely_title(potential_name, title_raw)):
                                        # Additional validation
                                        if potential_name.startswith('The ') and len(potential_name) > 7:
                                            publication_name = potential_name
                                            break
                                        elif potential_name not in ['The', 'A', 'An', 'By', 'From', 'Via']:
                                            if '|' in potential_name:
                                                parts = [p.strip() for p in potential_name.split('|')]
                                                if all(len(p) > 1 for p in parts):
                                                    publication_name = potential_name
                                                    break
                                            else:
                                                publication_name = potential_name
                                                break
                    
                    # Check for common source patterns in description
                    source_patterns = {
                        'Reuters': r'\bReuters\b',
                        'Bloomberg': r'\bBloomberg\b',
                        'Financial Times': r'\bFinancial Times\b|\bFT\b',
                        'Wall Street Journal': r'\bWall Street Journal\b|\bWSJ\b',
                        'Mining.com': r'\bMining\.com\b',
                        'MarketWatch': r'\bMarketWatch\b',
                        'Investing.com': r'\bInvesting\.com\b',
                        'The Economic Times': r'\bThe Economic Times\b|\bET\b',
                        'CNBC': r'\bCNBC\b',
                        'BBC': r'\bBBC\b',
                        'The Guardian': r'\bThe Guardian\b',
                        'New York Times': r'\bNew York Times\b|\bNYT\b',
                        'Agencia Peruana de Noticias | ANDINA': r'\bAgencia Peruana de Noticias\b|\bANDINA\b',
                    }
                    
                    for source_key, pattern in source_patterns.items():
                        if re.search(pattern, desc_text, re.IGNORECASE):
                            actual_source = source_key
                            if not publication_name:
                                publication_name = source_key
                            break
                
                # Get final URL and domain (from parallel resolution or extract now)
                # Prefer extracted URL from description if available
                if extracted_url:
                    final_url = extracted_url
                    real_domain = extract_domain_from_url(extracted_url)
                elif resolve_redirects and link:
                    final_url = link_to_final.get(link, link)
                    real_domain = link_to_domain.get(link)
                else:
                    final_url = link
                    real_domain = None
                
                # If we don't have domain yet, try to extract it
                if not real_domain:
                    if final_url and 'news.google.com' not in final_url and 'consent.google.com' not in final_url:
                        real_domain = extract_domain_from_url(final_url)
                    elif link:
                        real_domain = extract_domain_from_url(link)
                
                # Map domain to readable source name
                if real_domain and ('news.google.com' not in real_domain):
                    domain_to_source = {
                        'reuters.com': 'Reuters',
                        'bloomberg.com': 'Bloomberg',
                        'ft.com': 'Financial Times',
                        'wsj.com': 'Wall Street Journal',
                        'marketwatch.com': 'MarketWatch',
                        'investing.com': 'Investing.com',
                        'mining.com': 'Mining.com',
                        'economictimes.indiatimes.com': 'The Economic Times',
                        'cnbc.com': 'CNBC',
                        'financialpost.com': 'Financial Post',
                        'theguardian.com': 'The Guardian',
                        'bbc.com': 'BBC',
                        'nytimes.com': 'New York Times',
                        'andina.pe': 'Agencia Peruana de Noticias | ANDINA',
                        'portal.andina.pe': 'Agencia Peruana de Noticias | ANDINA',
                        'farmonaut.com': 'Farmonaut',
                    }
                    
                    # Use mapped name or clean domain name
                    if real_domain in domain_to_source:
                        mapped_name = domain_to_source[real_domain]
                        if actual_source == source_name:  # Only override if we haven't found a better source
                            actual_source = mapped_name
                        if not publication_name:
                            publication_name = mapped_name
                    elif actual_source == source_name:  # Only override if we haven't found a better source
                        # Use domain as source name (cleaned)
                        cleaned_domain_name = real_domain.replace('.com', '').replace('.co.', '.').replace('.', ' ').title()
                        actual_source = cleaned_domain_name
                        if not publication_name:
                            publication_name = cleaned_domain_name

        # parse publication date
                pub_date = None
                if pub_date_raw:
                    try:
                        # Try different date formats
                        for fmt in [
                            "%a, %d %b %Y %H:%M:%S %Z",
                            "%a, %d %b %Y %H:%M:%S %z",
                            "%Y-%m-%d %H:%M:%S",
                        ]:
                            try:
                                pub_date = datetime.strptime(pub_date_raw, fmt)
                                break
                            except ValueError:
                                continue
                    except Exception as e:
                        print(f"⚠️ Could not parse date '{pub_date_raw}': {e}")
                        continue

        # description contains HTML
                desc_text = BeautifulSoup(desc_raw, "html.parser").get_text(" ")

                # Skip if no title or date
                if not title_raw or not pub_date:
                    continue
                
                # Clean the text
                cleaned_title = clean_text(title_raw)
                cleaned_text = clean_text(desc_text)
                
                # Filter out non-copper-relevant articles
                if not is_copper_relevant(cleaned_title, cleaned_text):
                    continue

                # Determine the original article URL (not Google News redirect)
                article_url = None
                if extracted_url:
                    article_url = extracted_url
                elif final_url and 'news.google.com' not in final_url and 'consent.google.com' not in final_url:
                    article_url = final_url
                
                records.append(
                    {
                        "date": pub_date,
                        "title": cleaned_title,
                        "text": cleaned_text,
                        "source": actual_source,  # Use extracted source if found
                        "publication": publication_name if publication_name else actual_source,  # Real publication name (e.g., "The Economic Times")
                        "link": final_url if final_url else link,  # Use final URL after redirect
                        "article_url": article_url,  # Original article URL (not Google News)
                        "original_link": link,  # Keep original Google News link for reference
                        "domain": real_domain,  # Real domain from final URL
                        "query": query if query else source_name,  # Query/search term used to fetch this article
                    }
                )

            return records

        except requests.exceptions.RequestException as e:
            if attempt < retry_count - 1:
                print(f"⚠️ Attempt {attempt + 1} failed for {source_name}: {e}. Retrying...")
                time.sleep(delay * (attempt + 1))  # Exponential backoff
            else:
                print(f"❌ Failed to fetch {source_name} after {retry_count} attempts: {e}")
                return []
        except Exception as e:
            print(f"❌ Unexpected error fetching {source_name}: {e}")
            return []

    return []


def process_redirects_in_dataframe(
    df: pd.DataFrame,
    max_workers: int = 20,
    update_source: bool = True
) -> pd.DataFrame:
    """
    Post-process DataFrame to resolve redirects and update links and sources.
    This function should be called after parsing to update all links with final URLs.
    
    Args:
        df: DataFrame with news articles (must have 'link' column)
        max_workers: Maximum number of parallel threads for redirect resolution
        update_source: If True, update source names based on final domain
    
    Returns:
        DataFrame with updated links and sources
    """
    if df.empty or 'link' not in df.columns:
        return df
    
    print(f"\n🔗 Processing redirects for {len(df)} articles (parallel, {max_workers} workers)...")
    
    # Create a copy to avoid modifying original
    df_updated = df.copy()
    
    # Get unique links to resolve (avoid resolving same link multiple times)
    unique_links = df_updated['link'].dropna().unique()
    print(f"  Found {len(unique_links)} unique links to check")
    
    # Resolve redirects in parallel
    link_to_final = {}
    link_to_domain = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_link = {
            executor.submit(_resolve_link_redirect, link): link
            for link in unique_links
        }
        
        completed = 0
        for future in as_completed(future_to_link):
            original_link = future_to_link[future]
            completed += 1
            try:
                final_url, domain = future.result()
                link_to_final[original_link] = final_url
                if domain:
                    link_to_domain[original_link] = domain
                
                if completed % 100 == 0:
                    print(f"  [{completed}/{len(unique_links)}] Processed...")
            except Exception:
                # If redirect resolution fails, use original link
                link_to_final[original_link] = original_link
                domain = extract_domain_from_url(original_link)
                if domain:
                    link_to_domain[original_link] = domain
    
    print(f"  ✓ Completed processing {len(unique_links)} links")
    
    # Check for consent pages and filter them out
    consent_count = sum(1 for url in link_to_final.values() 
                       if url and ('consent.google.com' in url or 'accounts.google.com' in url))
    if consent_count > 0:
        print(f"  ⚠️  Warning: {consent_count} links redirected to Google consent page (will keep original)")
    
    # Update links in DataFrame
    df_updated['original_link'] = df_updated['link']  # Keep original for reference
    
    def update_link(original):
        final = link_to_final.get(original, original)
        # If we got a consent page, keep original link
        if final and ('consent.google.com' in final or 'accounts.google.com' in final):
            return original
        return final
    
    df_updated['link'] = df_updated['link'].map(update_link)
    
    # Update domain column (skip consent pages)
    def get_domain(original):
        domain = link_to_domain.get(original)
        # If domain is from consent page, try to extract from original
        if not domain or 'consent.google.com' in str(domain) or 'accounts.google.com' in str(domain):
            return extract_domain_from_url(original)
        return domain
    
    df_updated['domain'] = df_updated['original_link'].map(get_domain)
    
    # Update source names based on domain if requested
    if update_source:
        domain_to_source = {
            'reuters.com': 'Reuters',
            'bloomberg.com': 'Bloomberg',
            'ft.com': 'Financial Times',
            'wsj.com': 'Wall Street Journal',
            'marketwatch.com': 'MarketWatch',
            'investing.com': 'Investing.com',
            'mining.com': 'Mining.com',
            'economictimes.indiatimes.com': 'Economic Times',
            'cnbc.com': 'CNBC',
            'financialpost.com': 'Financial Post',
            'theguardian.com': 'The Guardian',
            'bbc.com': 'BBC',
            'nytimes.com': 'New York Times',
        }
        
        def update_source_from_domain(row):
            domain = row.get('domain')
            if domain and domain in domain_to_source:
                return domain_to_source[domain]
            elif domain and 'news.google.com' not in domain:
                # Use domain as source name (cleaned)
                return domain.replace('.com', '').replace('.co.', '.').replace('.', ' ').title()
            return row.get('source', 'Unknown')
        
        df_updated['source'] = df_updated.apply(update_source_from_domain, axis=1)
        print(f"  ✓ Updated source names based on domains")
    
    return df_updated


def process_redirects_in_csv(
    csv_file: str,
    output_file: Optional[str] = None,
    max_workers: int = 20,
    update_source: bool = True
) -> pd.DataFrame:
    """
    Load CSV file, process redirects, and save updated version.
    
    Args:
        csv_file: Path to input CSV file
        output_file: Path to output CSV file (if None, overwrites input)
        max_workers: Maximum number of parallel threads
        update_source: If True, update source names based on final domain
    
    Returns:
        Updated DataFrame
    """
    print(f"📂 Loading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Convert date column if it exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    print(f"  Loaded {len(df)} articles")
    
    # Process redirects
    df_updated = process_redirects_in_dataframe(df, max_workers=max_workers, update_source=update_source)
    
    # Save updated CSV
    if output_file is None:
        output_file = csv_file
    
    df_updated.to_csv(output_file, index=False)
    print(f"💾 Saved updated data to {output_file}")
    
    return df_updated


def fetch_google_news_by_date_range(
    query: str,
    start_date: datetime,
    end_date: datetime,
    source_filter: Optional[str] = None
) -> list[dict]:
    """
    Fetch Google News for a specific date range.
    Note: Google News RSS has limitations, but we can try different queries.
    
    Args:
        query: Search query (e.g., "copper")
        start_date: Start date for search
        end_date: End date for search
        source_filter: Optional source filter (e.g., "reuters")
    
    Returns:
        List of news records
    """
    records = []
    
    # Google News RSS doesn't directly support date ranges in URL,
    # but we can try different search strategies
    base_query = query
    if source_filter:
        base_query = f"{query} source:{source_filter}"
    
    url = (
        f"https://news.google.com/rss/search?"
        f"q={base_query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
    )
    
    try:
        fetched = fetch_google_news_rss("DateRange", url, query=base_query)
        # Filter by date range after fetching
        for record in fetched:
            if record["date"] and start_date <= record["date"] <= end_date:
                records.append(record)
    except Exception as e:
        print(f"⚠️ Error fetching date range {start_date} to {end_date}: {e}")
    
    return records


def _fetch_single_query(query: str) -> tuple[str, list[dict]]:
    """
    Helper function to fetch a single query (for parallel execution).
    
    Args:
        query: Search query string
    
    Returns:
        Tuple of (query_name, records)
    """
    query_name = f"Query_{query.replace(' ', '_')}"
    url = (
        f"https://news.google.com/rss/search?"
        f"q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
    )
    records = fetch_google_news_rss(query_name, url, query=query)
    return query_name, records


def fetch_copper_news_with_variations(max_workers: int = 4) -> pd.DataFrame:
    """
    Fetch copper news using multiple query variations to get more results.
    Google News RSS typically returns ~100 results per query, so we use
    different search terms to get more coverage.
    
    Args:
        max_workers: Maximum number of parallel threads (default: 4)
    
    Returns:
        DataFrame with news articles from multiple query variations
    """
    query_variations = [
        # Basic queries
        "copper price",
        "copper mining",
        "copper supply",
        "LME copper",
        "copper market",
        "copper demand",
        "copper futures",
        "copper production",
        # Major mine names to increase recall on operational news
        "Escondida copper mine",
        "Collahuasi copper mine",
        "Cerro Verde copper mine",
        "Buenavista del Cobre copper mine",
        "Kamoa Kakula copper mine",
        "Grasberg copper mine",
        "Antamina copper mine",
        "Morenci copper mine",
        "El Teniente copper mine",
        "Chuquicamata copper mine",
        # Geopolitical and war-related queries
        "copper war",
        "copper conflict",
        "copper sanctions",
        "copper embargo",
        "copper trade war",
        "copper supply chain",
        "copper logistics",
        "copper shipping",
        "copper disruption",
        "copper geopolitical",
        # Region-specific (geopolitical hotspots)
        "copper Russia",
        "copper Ukraine",
        "copper China trade",
        "copper Chile",
        "copper Peru",
        "copper Congo",
        "copper Zambia",
        # Reuters-specific queries (high-priority source)
        "copper price source:reuters",
        "copper mining source:reuters",
        "LME copper source:reuters",
        "copper supply source:reuters",
        "copper demand source:reuters",
        "copper futures source:reuters",
        "copper Chile source:reuters",
        "copper China source:reuters",
        "Codelco source:reuters",
        "BHP copper source:reuters",
        # Mining.com-specific queries (high-priority source)
        "copper source:mining.com",
        "copper mine source:mining.com",
        "copper production source:mining.com",
        "copper project source:mining.com",
        "copper deposit source:mining.com",
        "Escondida source:mining.com",
        "Grasberg source:mining.com",
        "Freeport copper source:mining.com",
    ]
    
    all_records = []

    print(f"📰 Fetching news with {len(query_variations)} query variations (parallel, {max_workers} workers)...")
    
    # Parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_query = {
            executor.submit(_fetch_single_query, query): query 
            for query in query_variations
        }
        
        completed = 0
        for future in as_completed(future_to_query):
            query = future_to_query[future]
            completed += 1
            try:
                query_name, records = future.result()
                all_records.extend(records)
                print(f"  [{completed}/{len(query_variations)}] ✓ '{query}': {len(records)} articles")
            except Exception as e:
                print(f"  [{completed}/{len(query_variations)}] ❌ '{query}' failed: {e}")
    
    if not all_records:
        return pd.DataFrame(columns=["date", "title", "text", "source", "publication", "link", "article_url", "original_link", "domain", "query"])

    df = pd.DataFrame(all_records)

    # Remove duplicates
    if not df.empty:
        df = df.dropna(subset=["date", "title"])
        df["date_only"] = df["date"].dt.date
        df = df.drop_duplicates(subset=["title", "date_only"], keep="first")
        df = df.drop(columns=["date_only"])
        df = df.sort_values("date", ascending=False).reset_index(drop=True)

    return df


def _fetch_single_year_query(year: int, query: str, query_index: int) -> tuple[int, list[dict]]:
    """
    Helper function to fetch a single query for a year (for parallel execution).
    
    Args:
        year: Target year
        query: Search query string
        query_index: Index of query in the list
    
    Returns:
        Tuple of (query_index, filtered_records)
    """
    url = (
        f"https://news.google.com/rss/search?"
        f"q={query.replace(' ', '+').replace('OR', 'OR')}&hl=en-US&gl=US&ceid=US:en"
    )
    records = fetch_google_news_rss(f"Historical_{year}", url, query=query)
    
    # Filter to only include articles from the target year
    filtered_records = [
        r for r in records 
        if r["date"] and r["date"].year == year
    ]
    return query_index, filtered_records


def fetch_copper_news_by_year(year: int, base_queries: Optional[list] = None, max_workers: int = 6) -> pd.DataFrame:
    """
    Fetch copper news for a specific year using multiple query strategies.
    Uses year-specific queries to try to get historical data.
    
    Args:
        year: Year to fetch news for
        base_queries: Optional list of base queries, otherwise uses defaults
    
    Returns:
        DataFrame with news articles for that year
    """
    if base_queries is None:
        base_queries = [
            "copper price",
            "copper mining",
            "LME copper",
            "copper market",
            "copper supply",
        ]
    
    all_records = []

    # Strategy 1: Query with year explicitly
    year_queries = [f"{q} {year}" for q in base_queries]
    
    # Strategy 2: Query with year range (year-1 to year+1 to catch edge cases)
    year_range_queries = [
        f"{q} {year-1} OR {q} {year} OR {q} {year+1}"
        for q in base_queries[:3]  # Use fewer for range queries
    ]
    
    # Strategy 3: Company/event specific queries that might have historical data
    company_queries = [
        f"BHP copper {year}",
        f"Rio Tinto copper {year}",
        f"Freeport copper {year}",
        f"Glencore copper {year}",
        f"Codelco copper {year}",
        f"Anglo American copper {year}",
        f"Antofagasta copper {year}",
    ]
    
    # Strategy 4: Event/keyword specific queries for better historical coverage
    event_queries = [
        f"copper strike {year}",
        f"copper mine {year}",
        f"copper production {year}",
        f"copper deficit {year}",
        f"copper surplus {year}",
        f"copper inventory {year}",
        f"copper warehouse {year}",
    ]
    
    # Strategy 5: Reuters and Mining.com specific queries (high-priority sources)
    reuters_queries = [
        f"copper {year} source:reuters",
        f"copper price {year} source:reuters",
        f"copper mining {year} source:reuters",
        f"LME copper {year} source:reuters",
        f"copper Chile {year} source:reuters",
        f"copper China {year} source:reuters",
    ]
    
    mining_com_queries = [
        f"copper {year} source:mining.com",
        f"copper mine {year} source:mining.com",
        f"copper production {year} source:mining.com",
        f"copper project {year} source:mining.com",
    ]
    
    # Strategy 6: Location-based queries (major copper producing regions)
    location_queries = [
        f"Chile copper {year}",
        f"Peru copper {year}",
        f"China copper {year}",
        f"DRC copper {year}",  # Democratic Republic of Congo
        f"Zambia copper {year}",
        f"Russia copper {year}",
        f"Ukraine copper {year}",
    ]
    
    # Strategy 6: Geopolitical and war-related queries
    geopolitical_queries = [
        f"copper war {year}",
        f"copper conflict {year}",
        f"copper sanctions {year}",
        f"copper embargo {year}",
        f"copper trade war {year}",
        f"copper supply chain {year}",
        f"copper disruption {year}",
        f"copper geopolitical {year}",
        f"Russia Ukraine copper {year}",
        f"China trade copper {year}",
    ]
    
    all_query_variations = (
        year_queries + 
        year_range_queries +
        reuters_queries +
        mining_com_queries + 
        company_queries + 
        event_queries + 
        location_queries +
        geopolitical_queries
    )
    
    print(f"📅 Fetching news for year {year} using {len(all_query_variations)} query variations (parallel, {max_workers} workers)...")
    
    # Parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_query = {
            executor.submit(_fetch_single_year_query, year, query, i): (i, query)
            for i, query in enumerate(all_query_variations, 1)
        }
        
        completed = 0
        for future in as_completed(future_to_query):
            i, query = future_to_query[future]
            completed += 1
            try:
                query_index, filtered_records = future.result()
                all_records.extend(filtered_records)
                print(f"  [{completed}/{len(all_query_variations)}] ✓ '{query[:50]}...': {len(filtered_records)} articles")
            except Exception as e:
                print(f"  [{completed}/{len(all_query_variations)}] ❌ Query {i} failed: {e}")
            # Small delay to avoid overwhelming the server
            time.sleep(0.1)
    
    if not all_records:
        return pd.DataFrame(columns=["date", "title", "text", "source", "publication", "link", "article_url", "original_link", "domain", "query"])

    df = pd.DataFrame(all_records)

    # Remove duplicates
    if not df.empty:
        df = df.dropna(subset=["date", "title"])
        df["date_only"] = df["date"].dt.date
        df = df.drop_duplicates(subset=["title", "date_only"], keep="first")
        df = df.drop(columns=["date_only"])
        df = df.sort_values("date", ascending=False).reset_index(drop=True)

    return df


def fetch_historical_copper_news(
    start_year: int = 2008,
    end_year: Optional[int] = None,
    years_to_fetch: Optional[list] = None,
    max_workers: int = 3
) -> pd.DataFrame:
    """
    Fetch historical copper news by iterating through years.
    This is more effective for getting older news than general queries.
    
    Args:
        start_year: Starting year (default 2008, matching price data)
        end_year: Ending year (default: current year - 1)
        years_to_fetch: Optional specific list of years to fetch
    
    Returns:
        DataFrame with historical news articles
    """
    if end_year is None:
        end_year = datetime.now().year - 1
    
    if years_to_fetch is None:
        years_to_fetch = list(range(start_year, end_year + 1))
    
    # Focus on years that likely have less coverage
    # Recent years (last 2) usually have good coverage from general queries
    # But we can still fetch them if needed
    recent_years = [datetime.now().year - i for i in range(2)]
    
    # Optionally skip recent years to save time
    # Uncomment the next line to skip recent years
    # years_to_fetch = [y for y in years_to_fetch if y not in recent_years]
    
    if years_to_fetch:
        print(f"📚 Fetching historical news for {len(years_to_fetch)} years: {min(years_to_fetch)}-{max(years_to_fetch)}")
        if recent_years and any(y in years_to_fetch for y in recent_years):
            print(f"   (Note: Recent years {recent_years} may already be covered by general queries)\n")
        else:
            print()
    
    all_records = []

    # Process years in parallel (but with limited workers to avoid overwhelming)
    print(f"📚 Processing {len(years_to_fetch)} years in parallel (max {max_workers} concurrent)...\n")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_year = {
            executor.submit(fetch_copper_news_by_year, year, None, max_workers): year
            for year in years_to_fetch
        }
        
        completed = 0
        for future in as_completed(future_to_year):
            year = future_to_year[future]
            completed += 1
            try:
                df_year = future.result()
                if not df_year.empty:
                    year_records = df_year.to_dict("records")
                    all_records.extend(year_records)
                    print(f"[{completed}/{len(years_to_fetch)}] ✓ Year {year}: {len(year_records)} articles\n")
                else:
                    print(f"[{completed}/{len(years_to_fetch)}] ⚠️ Year {year}: No articles found\n")
            except Exception as e:
                print(f"[{completed}/{len(years_to_fetch)}] ❌ Year {year} failed: {e}\n")
    
    if not all_records:
        return pd.DataFrame(columns=["date", "title", "text", "source", "publication", "link", "article_url", "original_link", "domain", "query"])

    df = pd.DataFrame(all_records)

    # Remove duplicates
    if not df.empty:
        df = df.dropna(subset=["date", "title"])
        df["date_only"] = df["date"].dt.date
        df = df.drop_duplicates(subset=["title", "date_only"], keep="first")
        df = df.drop(columns=["date_only"])
        df = df.sort_values("date", ascending=False).reset_index(drop=True)

    return df


def _fetch_single_source(source_name: str, url: str) -> tuple[str, list[dict]]:
    """
    Helper function to fetch from a single source (for parallel execution).
    Automatically detects if URL is a direct RSS feed or Google News RSS.
    
    Args:
        source_name: Name of the source
        url: RSS feed URL
    
    Returns:
        Tuple of (source_name, records)
    """
    # Check if this is a direct RSS feed (not Google News)
    is_direct_rss = 'news.google.com' not in url.lower()
    
    # Extract query from URL if possible, otherwise use source_name
    query = source_name  # Default to source_name
    if 'q=' in url:
        try:
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            if 'q' in params:
                query = unquote(params['q'][0])
        except Exception:
            pass
    
    # Use appropriate fetch function
    if is_direct_rss:
        records = fetch_direct_rss(source_name, url, filter_copper=True, query=query)
    else:
        records = fetch_google_news_rss(source_name, url, query=query)
    
    return source_name, records


def fetch_all_copper_news(
    use_all_sources: bool = True,
    min_date: Optional[datetime] = None,
    use_query_variations: bool = False,
    fetch_historical: bool = False,
    historical_start_year: int = 2008,
    historical_end_year: Optional[int] = None,
    max_workers: int = 5
) -> pd.DataFrame:
    """
    Fetch all available copper news from multiple sources.
    
    Args:
        use_all_sources: If True, use all sources including general search
        min_date: Optional minimum date to filter results
        use_query_variations: If True, also fetch using query variations (gets more data)
        fetch_historical: If True, also fetch historical news by year (slower but gets older articles)
        historical_start_year: Starting year for historical fetch (default 2008)
        historical_end_year: Ending year for historical fetch (default: current year - 1)
    
    Returns:
        DataFrame with news articles
    """
    all_records = []
    sources_to_use = RSS_SOURCES if use_all_sources else {
        k: v for k, v in RSS_SOURCES.items() 
        if k not in ["CopperGeneral"]
    }

    print(f"📰 Fetching news from {len(sources_to_use)} sources (parallel, {max_workers} workers)...")
    
    # Parallel execution for sources
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_source = {
            executor.submit(_fetch_single_source, source, url): source
            for source, url in sources_to_use.items()
        }
        
        completed = 0
        for future in as_completed(future_to_source):
            source = future_to_source[future]
            completed += 1
            try:
                source_name, records = future.result()
                all_records.extend(records)
                print(f"  [{completed}/{len(sources_to_use)}] ✓ {source}: {len(records)} articles")
            except Exception as e:
                print(f"  [{completed}/{len(sources_to_use)}] ❌ {source} failed: {e}")
    
    # Also fetch using query variations if requested
    if use_query_variations:
        print("\n📰 Fetching additional news using query variations...")
        df_variations = fetch_copper_news_with_variations(max_workers=max_workers)
        if not df_variations.empty:
            # Convert to list of dicts and add to all_records
            variation_records = df_variations.to_dict("records")
            all_records.extend(variation_records)
            print(f"  ✓ Got {len(variation_records)} additional articles from query variations")
    
    # Fetch historical news by year if requested (this takes longer but gets older articles)
    if fetch_historical:
        print("\n📚 Fetching historical news by year (this may take a while)...")
        df_historical = fetch_historical_copper_news(
            start_year=historical_start_year,
            end_year=historical_end_year,
            max_workers=max_workers
        )
        if not df_historical.empty:
            historical_records = df_historical.to_dict("records")
            all_records.extend(historical_records)
            print(f"  ✓ Got {len(historical_records)} additional historical articles")

    if not all_records:
        print("⚠️ No news records fetched!")
        return pd.DataFrame(columns=["date", "title", "text", "source", "publication", "link", "article_url", "original_link", "domain", "query"])

    df = pd.DataFrame(all_records)

    # Remove duplicates based on title and date
    print(f"\n📊 Total articles before deduplication: {len(df)}")
    
    if not df.empty:
        df = df.dropna(subset=["date", "title"])
        
        # Remove duplicates (same title and same day)
        df["date_only"] = df["date"].dt.date
        df = df.drop_duplicates(subset=["title", "date_only"], keep="first")
        df = df.drop(columns=["date_only"])
        
        # Filter by minimum date if provided
        if min_date:
            df = df[df["date"] >= min_date]
        
        df = df.sort_values("date", ascending=False).reset_index(drop=True)

    print(f"📊 Total unique articles after deduplication: {len(df)}")
    if not df.empty:
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")

    return df


if __name__ == "__main__":
    # Fetch all available news with query variations and historical data
    # 
    # Options:
    # - use_all_sources: Fetch from all RSS sources (recommended)
    # - use_query_variations: Use different query phrases for more coverage (recommended)
    # - fetch_historical: Fetch by year for better historical coverage (slower but gets older articles)
    # - historical_start_year: Start year for historical fetch (default 2008, matches price data)
    # - historical_end_year: End year (None = current year - 1)
    # - max_workers: Number of parallel threads (default: 5, increase for faster parsing but be careful with rate limits)
    #
    # For testing, you can set fetch_historical=False or use a small year range:
    #   historical_start_year=2020, historical_end_year=2021
    
    df_news = fetch_all_copper_news(
        use_all_sources=True,
        use_query_variations=True,  # Gets more recent articles
        fetch_historical=True,  # Gets historical articles by year (slower but more complete)
        historical_start_year=2008,  # Match price data start year
        historical_end_year=None,  # Will default to current year - 1
        max_workers=5  # Parallel threads (5-8 is usually safe, increase for faster parsing)
    )

    print("\n" + "="*60)
    print("📰 NEWS DATA SUMMARY")
    print("="*60)
    
    if not df_news.empty:
        print("\nPreview (first 5 articles):")
        print(df_news[["date", "title", "source"]].head().to_string())
        
        print("\n📊 News count by source:")
        print(df_news["source"].value_counts().to_string())
        
        print(f"\n📅 Date range: {df_news['date'].min()} to {df_news['date'].max()}")
        print(f"📈 Total unique articles: {len(df_news)}")
        
        # Articles per year
        df_news["year"] = df_news["date"].dt.year
        print("\n📅 Articles per year:")
        print(df_news["year"].value_counts().sort_index().to_string())
    else:
        print("⚠️ No news data fetched!")

    # Save to CSV (prefer Week-11 structure: data/raw/news)
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "raw" / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "copper_news_all_sources.csv"
    
    # Remove 'year' column if it exists (it's not needed in the saved file)
    if 'year' in df_news.columns:
        df_news = df_news.drop(columns=['year'])
    
    df_news.to_csv(output_file, index=False)
    print(f"\n💾 Saved to {output_file}")
