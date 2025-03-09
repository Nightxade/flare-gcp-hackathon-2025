import pandas as pd
import sys
import os
from typing import Any, override

import numpy.typing as npt
import structlog
from fastembed import LateInteractionTextEmbedding, SparseEmbedding, SparseTextEmbedding
from google.generativeai.client import configure
from google.generativeai.embedding import (
    EmbeddingTaskType,
)
from google.generativeai.embedding import (
    embed_content as _embed_content,
)
from google.generativeai.generative_models import ChatSession, GenerativeModel
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv

class GeminiSplitter:
    def __init__(self, api_key: str, model: str) -> None:
        configure(api_key=api_key)
        self.model = GenerativeModel(
            model_name=model
        )

    def generate(
        self,
        prompt: str,
        response_mime_type: str | None = None,
        response_schema: Any | None = None,
    ) -> str:
            
        response = self.model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                response_mime_type=response_mime_type, response_schema=response_schema
            ),
        )

        return response.text

load_dotenv()

MAX_SIZE = 1024 * 1024 * 5
df = pd.read_csv('src/data/docs.csv')
model = "gemini-2.0-flash"
GEMINI_API_KEY = str(os.getenv("GEMINI_API_KEY"))
responder = GeminiSplitter(GEMINI_API_KEY, model)

def getsize(doc : pd.Series):
    return sys.getsizeof(str(doc["Contents"])) / 1000000

def split(content : str):
    prompt = f"""
    Split the following document into meaningful sections based on its contents, ensuring that each part remains as coherent as possible.
    Return a JSON list of the text segments:
    {content}

    Each "CONTENT" in the json MUST be STRICTLY UNDER 5 megabytes.
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

new_df.to_csv("src/data/split_docs.csv", index=False)