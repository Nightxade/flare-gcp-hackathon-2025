from google.oauth2 import service_account
from google.cloud import bigquery
import time
import pandas as pd 
# hey so figure out how to get this in a .env tyvm 

credentials = service_account.Credentials.from_service_account_file("FlareAPIKey.json")
client = bigquery.Client()

job_config = bigquery.LoadJobConfig(
    source_format = bigquery.SourceFormat.CSV,
    skip_leading_rows = 1,
    autodetect = True,
)

with open(r'crawl_output.csv', "rb") as source_file:
# Todo maybe?






