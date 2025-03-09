import concurrent.futures
import json
from typing import Any

import pandas as pd

from flare_ai_rag.ai.gemini import GeminiGeneric
from flare_ai_rag.settings import settings
from flare_ai_rag.utils import load_json

IN_PATH = "data/md_out.csv"
OUT_PATH = "data/mdocs.csv"
PROMPT=f"""
Please parse the following file from Markdown to JSON, using these rules:
- Your output should be formatted in JSON.
- The JSON for each section should be formatted as {'filename', 'metadata', 'contents'}. 'metadata' should be in JSON format as well.
- The 'metadata' section should include the title, slug, description, and a list of keywords
- MAKE SURE YOUR FORMATTING IS CORRECT TOO, ESPECIALLY DELIMITERS, JSON FORMAT, STRING TERMINATORS, AND ESCAPE SEQUENCES.
- Infer the filename based on the contents

Here is an example `metadata` section:
sidebar_position: 1
slug: overview
title: Scaling
description: Scaling is an advanced framework designed to optimize the functionality and efficiency of FTSOv2.
keywords: [ftso, oracle, flare-time-series-oracle, flare-network]

Now, here is the file to convert:
$MARKDOWN
"""

input_config = load_json(settings.input_path / "input_parameters.json")
model = GeminiGeneric(settings.gemini_api_key, input_config["router_model"]["id"])
assert settings.gemini_api_key != ""

df = pd.read_csv(IN_PATH)
ndf = []
print(len(df))

def format_meta(m: dict) -> str:
    if "keywords" in m:
        m["keywords"] = f"  [\n{'\n'.join(["    " + k for k in m["keywords"]])}\n  ]"
    return "\n".join([f"{i[0]}: {i[1]}" for i in m.items()])

fnames = set()
def task(row: pd.Series) -> list[Any]:
    try:
        markdown = row["Markdown"]

        prompt = PROMPT.replace("$MARKDOWN", markdown)
        response = model.generate(prompt=prompt, response_mime_type="application/json")
        response = json.loads(response.text)
        for file in response:
            x = 1
            while file["filename"] in fnames:
                s: str = file["filename"]
                file["filename"] = s[:s.rfind(".")] + f"-{x}" + s[s.rfind("."):]
                x += 1
            fnames.add(file["filename"])

            print(f"size: {len(file["contents"])}")
            return [file["filename"], format_meta(file["metadata"]), file["contents"], "2025-03-09"]
    except Exception as e:  # noqa: BLE001
        print(e)  # noqa: T201

    return []

with concurrent.futures.ProcessPoolExecutor(max_workers=64) as executor:
    futures = [executor.submit(task, i[1]) for i in df.iterrows()]

ndf = [f.result() for f in futures if f.result()]
ndf = pd.DataFrame(ndf, columns=["Filename", "Metadata", "Contents", "LastUpdated"])
print(len(ndf))
ndf.to_csv(OUT_PATH, index=False)
