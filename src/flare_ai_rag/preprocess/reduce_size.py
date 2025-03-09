import concurrent.futures
import json
from typing import Any

import pandas as pd

from flare_ai_rag.ai.gemini import GeminiGeneric
from flare_ai_rag.settings import settings
from flare_ai_rag.utils import load_json

IN_PATH = "data/docs.csv"
OUT_PATH = "data/ndocs.csv"
LIMIT = 9000
PROMPT=f"""
Please split the following file into $NUM or more different sections.
Each section should be STRICTLY LESS than {LIMIT} characters long, including whitespace.
The file should be split by relevance, i.e. each section should function as a standalone file.
If the section is too long, summarize it. (i.e. longer than the limit of {LIMIT} characters).
Each section should also be assigned a new filename and new metadata, relevant to its content.
Your output should be formatted in JSON.
The JSON for each section should be formatted as {'filename', 'metadata', 'contents'}. 'metadata' should be in JSON format as well.
Remember, each section should be LESS THAN THE HARD LIMIT OF {LIMIT} CHARACTERS LONG, INCLUDING WHITESPACE. THIS IS VERY IMPORTANT. PLEASE MAKE SURE TO FOLLOW THIS LIMIT.
DO NOT HAVE MORE THAN {LIMIT} CHARACTERS IN EACH SECTION. MAKE SURE OF THIS. DO NOT IGNORE THIS.
MAKE SURE YOUR FORMATTING IS CORRECT TOO, ESPECIALLY DELIMITERS, JSON FORMAT, STRING TERMINATORS, AND ESCAPE SEQUENCES.

Here is the file to split:
Filename: $FILENAME
Metadata: $METADATA
Contents:
$CONTENTS
"""

input_config = load_json(settings.input_path / "input_parameters.json")
model = GeminiGeneric(settings.gemini_api_key, input_config["router_model"]["id"])

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
        num = (len(row["Contents"]) // LIMIT) + 1
        fname = row["Filename"]
        contents = row["Contents"]
        meta = row["Metadata"]

        if len(contents) < LIMIT:
            return [fname, meta, contents, row["LastUpdated"]]
        else:
            prompt = PROMPT.replace("$NUM", str(num)).replace("$FILENAME", fname)
            prompt = prompt.replace("$CONTENTS", contents).replace("$METADATA", meta)
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
                return [file["filename"], format_meta(file["metadata"]), file["contents"], row["LastUpdated"]]
    except Exception as e:  # noqa: BLE001
        print(row["Filename"].ljust(50, " "), e)  # noqa: T201

    return []

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(task, i[1]) for i in df.iterrows()]

ndf = [f.result() for f in futures if f.result()]
ndf = pd.DataFrame(ndf, columns=["Filename", "Metadata", "Contents", "LastUpdated"])
print(len(ndf))
ndf.to_csv(OUT_PATH, index=False)
