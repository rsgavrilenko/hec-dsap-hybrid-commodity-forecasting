import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime


URL = "https://www.westmetall.com/en/markdaten.php?action=table&field=LME_Cu_cash"


def parse_number(x):
    """
    Convert strings like:
    - '5,880.50'
    - '313,300'
    - '-'
    - '' / None
    into float or NaN.
    """
    if x is None:
        return None

    x = x.strip()

    # Westmetall missing values
    if x == "-" or x == "":
        return None

    # just in case: remove non-breaking spaces
    x = x.replace("\xa0", "")

    return float(x.replace(",", ""))



def parse_date(x: str) -> pd.Timestamp:
    return pd.to_datetime(datetime.strptime(x, "%d. %B %Y"))


def fetch_copper_data_all_years(url: str = URL) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    all_rows = []

    # ⬇️ ВАЖНО: на странице несколько таблиц — по одной на каждый год
    tables = soup.find_all("table")

    if not tables:
        raise RuntimeError("No tables found on Westmetall page")

    for table in tables:
        tbody = table.find("tbody")
        if not tbody:
            continue

        for row in tbody.find_all("tr"):
            cols = [c.get_text(strip=True) for c in row.find_all("td")]
            if len(cols) != 4:
                continue

            date, cash, three_month, stock = cols

            all_rows.append({
                "date": parse_date(date),
                "lme_copper_cash": parse_number(cash),
                "lme_copper_3m": parse_number(three_month),
                "lme_copper_stock": parse_number(stock),
            })

    df = pd.DataFrame(all_rows)

    # удалить возможные дубликаты и отсортировать
    df = (
        df.drop_duplicates(subset=["date"])
          .sort_values("date")
          .reset_index(drop=True)
    )

    return df


if __name__ == "__main__":
    df = fetch_copper_data_all_years()

    print("Rows:", len(df))
    print(df.head())
    print(df.tail())

    df.to_csv("data_copper_lme_all_years.csv", index=False)
    print("Saved data_copper_lme_all_years.csv")
