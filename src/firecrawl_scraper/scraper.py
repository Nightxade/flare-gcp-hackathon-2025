import csv
import os
import time

from dotenv import load_dotenv
from firecrawl import FirecrawlApp

load_dotenv()

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

base_url = ["https://docs.flare.network/", "https://github.com/flare-foundation", "https://flare.network/news"]
output_file = "src/firecrawl_scraper/crawl_output.csv"
lastrequest = None


def find_crawl(url: str):
    try:
        crawl_result = app.crawl_url(
            url=url,
            params={
                #"limit": 2,
                "allowBackwardLinks": False,
                "allowExternalLinks": False,
                "scrapeOptions": {
                    "formats": ["html"],
                    "excludeTags": ["button"],
                },
            },
        )
        return crawl_result
    except Exception as e:
        print(f"Error fetching crawl data {e}")
        return None


def save_crawl_results(url: list):
    global lastrequest
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["HTML Content"])
    for i in url:
        if lastrequest is not None:
            elapsed = time.time() - lastrequest
            if elapsed < 60:
                time.sleep(60 - elapsed)
        crawl_result = find_crawl(i)
        if not crawl_result or "data" not in crawl_result:
            print("No crawl data")
            return
        try:
            with open(output_file, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                for page in crawl_result["data"]:
                    html_content = page.get("html", "No content")
                    writer.writerow([html_content])
            lastrequest = time.time()
        except Exception as e:
            print(f"Error saving data to CSV: {e}")
    
    print(f"Crawl data saved to {output_file}")

save_crawl_results(base_url)
