import requests
from bs4 import BeautifulSoup

def scrape(ticker: str):
    url = f"https://finance.yahoo.com/quote/{ticker}/history/?period1=1710008248&period2=1741540639&filter=history&frequency=1d"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    r = requests.get(url, headers=headers)
    
    soup = BeautifulSoup(r.text, 'html.parser')
    table = soup.find('table')
    historical_data = []
    if table:
        for row in table.find('tbody').find_all('tr')[:30]:
            columns = row.find_all('td')
            if len(columns) < 6:
                continue
            date = columns[0].text.strip()
            open_price = columns[1].text.strip()
            high_price = columns[2].text.strip()
            low_price = columns[3].text.strip()
            close_price = columns[4].text.strip()
            volume = columns[6].text.strip()

            historical_data.append({
                "date": date,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume
            })

    return historical_data