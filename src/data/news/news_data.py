import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import re


RSS_SOURCES = {
    "Reuters": (
        "https://news.google.com/rss/search?"
        "q=copper+source:reuters&hl=en-US&gl=US&ceid=US:en"
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
}


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


def fetch_google_news_rss(source_name: str, url: str) -> list[dict]:
    response = requests.get(url, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "xml")

    records = []

    for item in soup.find_all("item"):
        title_raw = item.title.text if item.title else ""
        link = item.link.text if item.link else None
        pub_date_raw = item.pubDate.text if item.pubDate else None
        desc_raw = item.description.text if item.description else ""

        # parse publication date
        pub_date = None
        if pub_date_raw:
            pub_date = datetime.strptime(
                pub_date_raw, "%a, %d %b %Y %H:%M:%S %Z"
            )

        # description contains HTML
        desc_text = BeautifulSoup(desc_raw, "html.parser").get_text(" ")

        records.append(
            {
                "date": pub_date,
                "title": clean_text(title_raw),
                "text": clean_text(desc_text),
                "source": source_name,
                "link": link,
            }
        )

    return records


def fetch_all_copper_news() -> pd.DataFrame:
    all_records = []

    for source, url in RSS_SOURCES.items():
        print(f"Fetching {source}...")
        try:
            all_records.extend(fetch_google_news_rss(source, url))
        except Exception as e:
            print(f"⚠️ Failed to fetch {source}: {e}")

    df = pd.DataFrame(all_records)

    if not df.empty:
        df = df.dropna(subset=["date"])
        df = df.sort_values("date", ascending=False).reset_index(drop=True)

    return df


if __name__ == "__main__":
    df_news = fetch_all_copper_news()

    print("\nPreview:")
    print(df_news.head())

    print("\nNews count by source:")
    print(df_news["source"].value_counts())

    # Optional: save to CSV
    df_news.to_csv("copper_news_all_sources.csv", index=False)
