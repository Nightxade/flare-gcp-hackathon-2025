import html2text
import pandas as pd

df = pd.read_csv("src/firecrawl_scraper/crawl_output.csv", encoding="utf-8")


converter = html2text.HTML2Text()
converter.ignore_links = False  # maybe this will make us go down rabbitholes?
converter.ignore_images = True
converter.ignore_emphasis = True

# Function to convert HTML to Markdown
def convert_html(html_content):
    return converter.handle(html_content)

# Apply conversion to the correct column
markdown_articles = df["HTML Content"].apply(convert_html)


markdownedData = pd.DataFrame({"Markdown": markdown_articles})
print("Shape of the new DataFrame:", markdownedData.shape)

markdownedData.to_csv("markdownedData.csv", index=False)
