import pandas as pd
import sys
import os
from src.flare_ai_rag.ai import gemini
from dotenv import load_dotenv

load_dotenv()

MAX_SIZE = 1024 * 1024 * 5
df = pd.read_csv('docs.csv')
model = "gemini-2.0-flash"
GEMINI_API_KEY = str(os.getenv("GEMINI_API_KEY"))
responder = gemini.GeminiSplitter(GEMINI_API_KEY, model)

def getsize(doc : pd.Series):
    return sys.getsizeof(str(doc["Contents"]))

def split(content : str):
    prompt = f"""
    Split the following document into meaningful sections based on its contents, ensuring that each part remains coherent, and that the size of each document is under 5mb.
    Return a JSON list of the text segments:
    {content}
    """
    response = responder.generate(prompt)
    return response

new_rows = []
for _, row in df.iterrows():
    content_size = getsize(row)
    
    if content_size > MAX_SIZE:
        chunks = split(row["Contents"])
        
        for i, chunk in enumerate(chunks):
            new_row = row.copy()
            new_row["Contents"] = chunk
            new_row["Filename"] = f"{row['Filename']}_part{i+1}"
            new_rows.append(new_row)
    else:
        new_rows.append(row)

new_df = pd.DataFrame(new_rows)

new_df.to_csv("split_docs.csv", index=False)