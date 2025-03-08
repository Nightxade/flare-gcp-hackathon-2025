from firecrawl import FirecrawlApp
from dotenv import load_dotenv
import csv
import os

load_dotenv()

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

base_url = "https://docs.flare.network/"
output_file = "src/firecrawl_scraper/crawl_output.csv"


def find_crawl(url: str):
    try:
        crawl_result = app.crawl_url(
            url=url,
            params={
                "limit": 3,
                "allowBackwardLinks": False,
                "allowExternalLinks": False,
                "scrapeOptions": {
                    "formats": ["links", "html"],
                    "excludeTags": ["button"],
                },
            },
        )
        return crawl_result
    except Exception as e:
        print(f"Error fetching crawl data {e}")
        return None


def save_crawl_results(url: str):
    crawl_result = find_crawl(url)
    if not crawl_result or "data" not in crawl_result:
        print("No crawl data")
        return
    try:
        with open(output_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Links", "HTML Content"])
            for page in crawl_result["data"]:
                links = ", ".join(page.get("links", []))
                html_content = page.get("html", "No content")
                writer.writerow([links, html_content])
        print(f"Crawl data saved to {output_file}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")


save_crawl_results(base_url)
