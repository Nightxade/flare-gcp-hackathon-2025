import html2text
import pandas as pd

df = pd.read_csv("src/firecrawl_scraper/crawl_output.csv")
print(df.columns)


converter = html2text.HTML2Text()
converter.ignore_links = False  # this keeps any links
converter.ignore_images = True  # no images. Maybe look into analyzing graphs
converter.ignore_emphasis = True  # who wants bolds and italics


def convert_html(html_content):
    return converter.handle(html_content)
    # convert html to markdown


df.iloc[:, 1] = df.iloc[:, 1].apply(convert_html)
df.rename(columns={df.columns[1]: "Markdown"}, inplace=True)
print(df.head())
df.to_csv("markdownedData.csv")
# maybe we just want to send ONLY the markdowned data. So I replaced the html file with markdown
