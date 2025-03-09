import json
import os

import pandas as pd
from dotenv import load_dotenv  # this helps protect our api key
from google import genai

load_dotenv()

# Load API Key from Environment
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# model = genai.GenerativeModel(model_name="gemini-1.5-flash")
df = pd.read_csv("markdownedData.csv")


def extract_metadata(title, text):
    """Uses Gemini to extract keywords & summary."""
    prompt = f"""
    Title: {title}
    Text: {text}
    
    Extract metadata:
    - Generate a summary (max 50 words)  
    - Identify relevant keywords (max 5) Place a special emphasis on [blockchain, Flare, oracle, crypto, smart contract, staking, consensus,
   gas, node]
    - Identify the category (e.g., DeFi, Layer 1, Regulations). 
    """
    # generating summary may not be relevant
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)

    return json.loads(response.text)
    # or maybe return response.txt

# arguments for 
df["metadata"] = df.apply(lambda row: extract_metadata(row.iloc[0], row.iloc[1]), axis = 1)
 
 # Expand JSON fields into separate columns (gemini returns json)
df["summary"] = df["metadata"].apply(lambda x: x.get("summary", ""))
df["keywords"] = df["metadata"].apply(
    lambda x: ", ".join(x.get("keywords", []))
)  # Convert list to string
df["category"] = df["metadata"].apply(lambda x: x.get("category", ""))
df.drop(columns=["metadata"], inplace=True)

df.to_csv("processed_metadata.csv", index=False)
print("Markdown processed and metadata saved in 'processed_metadata.csv'.")

# model = genai.GenerativeModel(model_name="gemini-1.5-pro")
# response = model.generate_content(prompt)
# return response.text
