
import pandas as pd
from datetime import datetime
from google import genai
import os
from dotenv import load_dotenv
from tqdm import tqdm

# ok this is just gonna be based off the gemini.py example 
load_dotenv()

class MetadataGenerator:
    """
    A class to generate metadata for each row in a CSV file using Google's Gemini AI.
    This class uses the Gemini API to generate metadata like title, description, keywords, etc.
    based on the content of each row.
    """

    def __init__(self, input_csv: str, output_csv: str) -> None:
        """
        Initialize the MetadataGenerator with the input and output CSV file paths.

        Args:
            input_csv (str): Path to the input CSV file containing data
            output_csv (str): Path to the output CSV file to store metadata
        """
        self.input_csv = input_csv
        self.output_csv = output_csv

        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)

    def read_csv(self) -> pd.DataFrame:
        """
        Reads the input CSV file into a pandas DataFrame, specifying UTF-8 encoding to avoid decoding errors.

        Returns:
            pd.DataFrame: The loaded CSV file as a DataFrame
        """
        return pd.read_csv(self.input_csv, encoding='utf-8')

    def generate_metadata_and_write_to_csv(self, row: str) -> None:
        """
        Generate metadata for a single row using Gemini AI and write it directly to the output CSV.
        Let Gemini generate the title, description, keywords, slug, and content.

        Args:
            row (str): The content of the row to generate metadata for
        """
        # Define the prompt for Gemini (asking it to generate all the metadata). Finetune this a lot 
        prompt = (
            "Please read the following article and generate the following metadata:\n\n"
            "1. Slug: A URL-friendly version of the title (lowercase with hyphens instead of spaces). If none found make some up\n"
            "2. Title: The title of the article.\n"
            "3. Description: A brief description (<=20 words) summarizing the article.\n"
            "4. Keywords: A list of relevant keywords (comma-separated, <=10 total). These should just be one word each\n"
            "5. Content: The rest of the article\n\n"
            f"Article Content:\n{row}\n\n"
            "Please return the metadata in the following format:\n"
            "<slug>\n <title>\n <keywords>\n<description>\n <content>"
            "An example output for a site Introducing Flare, the blockchain for data is: " 
            '''slug: intro 
        title: Introduction  
        description: Introduction to Flare, the blockchain for data. 
        keywords: # come up with keywords. This is the keywords for the file 
        [
            flare-network,
            blockchain,
            data,
            smart contracts,
            flare-time-series-oracle,
            flare-data-connector,
        ]","import ThemedImage from ""@theme/ThemedImage"";
        import useBaseUrl from ""@docusaurus/useBaseUrl"";

        **Flare is the blockchain for data ☀️**, offering developers and users secure, decentralized access to high-integrity data from other chains and the internet. Flare's Layer-1 network uniquely supports enshrined data protocols at the network layer, making it the only EVM-compatible smart contract platform optimized for decentralized data acquisition, including price and time-series data, blockchain event and state data, and Web2 API data.

        By providing broad data access at scale and minimal cost, Flare delivers a full-stack solution for building the next generation of secure, interoperable, and data-driven decentralized applications.

        ## Getting started

        - [Hello World in your browser](/network/getting-started) — Build and deploy your first smart contract on Flare, using only your browser.

        - [Learn how to use FTSOv2](/ftso/overview) — Leverage the latest upgrades to the Flare Time Series Oracle (FTSO), with feeds now updating every ≈1.8 seconds.

        - Start building on Flare with programming languages you may already know

        - [JavaScript](/network/guides/flare-for-javascript-developers)
        - [Python](/network/guides/flare-for-python-developers)
        - [Rust](/network/guides/flare-for-rust-developers)
        - [Go](/network/guides/flare-for-go-developers)

        ## Understand the architecture

        Build a strong understanding of the core concepts that set Flare apart from other blockchains. Flare's data protocols, [Flare Time Series Oracle (FTSO)](/ftso/overview) and [Flare Data Connector (FDC)](/fdc/overview), are enshrined into the core protocol of Flare, and inherit the economic security of the entire network.

        <ThemedImage
        alt=""Flare Architecture""
        sources={{
            light: useBaseUrl(""img/flare_architecture_light.svg""),
            dark: useBaseUrl(""img/flare_architecture_dark.svg""),
        }}
        />

        ## Contribute to Flare

        - [Contribute to Flare's open-source codebase](https://github.com/flare-foundation) — Help build the future of Flare.

        - [Become an FTSO data provider](/run-node/ftso-data-provider) — Support DeFi applications on Flare with high-integrity, block-latency data feeds.

        - [Run a Flare validator](/run-node/validator-node) — Secure Flare and earn rewards by running a validator node.
        VERY IMPORTANT PLEASE ALWAYS DO: Put quotes before first instance of slug, and after keywords list''' 
        )

        response = self.client.models.generate_content(
            model='gemini-2.0-flash', 
            contents=prompt
        )

        title = response.text.split("\n")[0].strip() if len(response.text.split("\n")) > 0 else "No title generated"
        description = response.text.split("\n")[1].strip() if len(response.text.split("\n")) > 1 else "No description generated"
        keywords = response.text.split("\n")[2].strip() if len(response.text.split("\n")) > 2 else "No keywords generated"
        slug = response.text.split("\n")[3].strip() if len(response.text.split("\n")) > 3 else "No slug generated"
        content = row.strip()  # Keep the original content

        # Prepare the metadata in a format to be written in the CSV
        metadata = f"slug: {slug}\ntitle: {title}\ndescription: {description}\nkeywords: {keywords}"

        # Prepare a row with filename, metadata, and content
        row_data = {
            "filename": "test",
            "metadata": metadata,
            "content": content
        }

        # Write the row to CSV
        with open(self.output_csv, 'a', newline='', encoding='utf-8') as file:
            file.write(f'"{row_data["filename"]}","{row_data["metadata"]}","{row_data["content"]}"\n')

    def generate_metadata_for_csv(self) -> None:
        """
        Read the input CSV, generate metadata for each row using Gemini, and write it to the output CSV.
        """
        df = self.read_csv()
        print(df)
        # Iterate through rows, generate metadata for each, and write directly to the output CSV
        for idx, row in df.iterrows():
            filename = f"{idx+1}-{row[0]}"  # You can customize how to generate the filename/slug
            content = row[1]  # Assuming the content is in the second column
            self.generate_metadata_and_write_to_csv(content, filename)
        print(f"Metadata written to {self.output_csv}")
        
        for row in tqdm(df.iloc[:, 0]):  # Assuming content is in the first column
            self.generate_metadata_and_write_to_csv(row)
        print(f"Metadata written to {self.output_csv}")


# Example Usage
df1 = pd.DataFrame(list())
# os.remove("output_metadata.csv")
metadata_generator = MetadataGenerator(input_csv="markdownedData.csv", output_csv="output_metadata.csv")
print("hihi this is working")
metadata_generator.generate_metadata_for_csv()


"Please read the following article and generate the following metadata:\n\n"
            "1. Title: A concise, clear title for the article.\n"
            "2. Description: A brief description (<=20 words) summarizing the article.\n"
            "3. Keywords: A list of relevant keywords (comma-separated, <=10 total). These should just be one word each\n"
            "4. Slug: A URL-friendly version of the title (lowercase with hyphens instead of spaces). If none found make some up\n"
            "5. Content: This part is the same as the rest of the article\n\n"
            f"Article Content:\n{row}\n\n"
            "Return the metadata in the following format:\n"
            "<slug>\n<title>\n<keywords>\n\<description>\n<content>"

            "An example output for a site Introducing Flare, the blockchain for data is: " 
            '''slug: intro 
title: Introduction  
description: Introduction to Flare, the blockchain for data. 
keywords: # come up with keywords. This is the keywords for the file 
  [
    flare-network,
    blockchain,
    data,
    smart contracts,
    flare-time-series-oracle,
    flare-data-connector,
  ]","import ThemedImage from ""@theme/ThemedImage"";
import useBaseUrl from ""@docusaurus/useBaseUrl"";

**Flare is the blockchain for data ☀️**, offering developers and users secure, decentralized access to high-integrity data from other chains and the internet. Flare's Layer-1 network uniquely supports enshrined data protocols at the network layer, making it the only EVM-compatible smart contract platform optimized for decentralized data acquisition, including price and time-series data, blockchain event and state data, and Web2 API data.

By providing broad data access at scale and minimal cost, Flare delivers a full-stack solution for building the next generation of secure, interoperable, and data-driven decentralized applications.

## Getting started

- [Hello World in your browser](/network/getting-started) — Build and deploy your first smart contract on Flare, using only your browser.

- [Learn how to use FTSOv2](/ftso/overview) — Leverage the latest upgrades to the Flare Time Series Oracle (FTSO), with feeds now updating every ≈1.8 seconds.

- Start building on Flare with programming languages you may already know

  - [JavaScript](/network/guides/flare-for-javascript-developers)
  - [Python](/network/guides/flare-for-python-developers)
  - [Rust](/network/guides/flare-for-rust-developers)
  - [Go](/network/guides/flare-for-go-developers)

## Understand the architecture

Build a strong understanding of the core concepts that set Flare apart from other blockchains. Flare's data protocols, [Flare Time Series Oracle (FTSO)](/ftso/overview) and [Flare Data Connector (FDC)](/fdc/overview), are enshrined into the core protocol of Flare, and inherit the economic security of the entire network.

<ThemedImage
  alt=""Flare Architecture""
  sources={{
    light: useBaseUrl(""img/flare_architecture_light.svg""),
    dark: useBaseUrl(""img/flare_architecture_dark.svg""),
  }}
/>

## Contribute to Flare

- [Contribute to Flare's open-source codebase](https://github.com/flare-foundation) — Help build the future of Flare.

- [Become an FTSO data provider](/run-node/ftso-data-provider) — Support DeFi applications on Flare with high-integrity, block-latency data feeds.

- [Run a Flare validator](/run-node/validator-node) — Secure Flare and earn rewards by running a validator node.


''' 

import pandas as pd
from datetime import datetime
from google import genai
import os
from dotenv import load_dotenv
from tqdm import tqdm

# ok this is just gonna be based off the gemini.py example 
load_dotenv()

class MetadataGenerator:
    """
    A class to generate metadata for each row in a CSV file using Google's Gemini AI.
    This class uses the Gemini API to generate metadata like title, description, keywords, etc.
    based on the content of each row.
    """

    def __init__(self, input_csv: str, output_csv: str) -> None:
        """
        Initialize the MetadataGenerator with the input and output CSV file paths.

        Args:
            input_csv (str): Path to the input CSV file containing data
            output_csv (str): Path to the output CSV file to store metadata
        """
        self.input_csv = input_csv
        self.output_csv = output_csv

        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)

    def read_csv(self) -> pd.DataFrame:
        """
        Reads the input CSV file into a pandas DataFrame, specifying UTF-8 encoding to avoid decoding errors.

        Returns:
            pd.DataFrame: The loaded CSV file as a DataFrame
        """
        return pd.read_csv(self.input_csv, encoding='utf-8')

    def generate_metadata_and_write_to_csv(self, row: str) -> None:
        """
        Generate metadata for a single row using Gemini AI and write it directly to the output CSV.
        Let Gemini generate the title, description, keywords, slug, and content.

        Args:
            row (str): The content of the row to generate metadata for
        """
        # Define the prompt for Gemini (asking it to generate all the metadata). Finetune this a lot 
        prompt = (
        '''Read the input csv file and create a new csv file with three columns: Filename, Metadata, Contents. 
        Let Filename always be "Temp"
        Metadata is formatted from the following article way: 
        1. Slug: End of URL. If none found make some up"
        2. Title: A concise, clear title for the article."
        3. Description: A brief description (<=20 words) summarizing the article."
        4. Keywords: A list of relevant keywords (comma-separated, <=10 total). These should just be one word each"
        An example metadata is 
            slug: intro 
            title: Introduction  
            description: Introduction to Flare, the blockchain for data. 
            keywords: # come up with keywords. This is the keywords for the file 
            [
                flare-network,
                blockchain,
                data,
                smart contracts,
                flare-time-series-oracle,
                flare-data-connector,
            ]",
        Contents is anything else in the article that is not the title
        '''
        )

        response = self.client.models.generate_content(
            model='gemini-2.0-flash', 
            contents=prompt
        )

        # Directly use Gemini's response (we expect the response to be properly formatted with title, description, keywords, and slug)
        title = response.text.split("\n")[0].strip() if len(response.text.split("\n")) > 0 else "No title generated"
        description = response.text.split("\n")[1].strip() if len(response.text.split("\n")) > 1 else "No description generated"
        keywords = response.text.split("\n")[2].strip() if len(response.text.split("\n")) > 2 else "No keywords generated"
        slug = response.text.split("\n")[3].strip() if len(response.text.split("\n")) > 3 else "No slug generated"
        content = row.strip()  # Keep the original content or modify as needed

        # Format the metadata for CSV
        formatted_metadata = f"""slug: {slug}
title: {title}
description: {description}
keywords: # come up with keywords. This is the keywords for the file
  [
    {', '.join(keywords.split(","))}
  ]

**{title}**: {description}

{content}
"""

        # Write to the CSV directly (append mode to add new rows)
        with open(self.output_csv, 'a', newline='', encoding='utf-8') as file:
            file.write(formatted_metadata + "\n")

    def generate_metadata_for_csv(self) -> None:
        """
        Read the input CSV, generate metadata for each row using Gemini, and write it to the output CSV.
        """
        df = self.read_csv()

        # Iterate through rows, generate metadata for each, and write directly to the output CSV
        for row in tqdm(df.iloc[:, 0]):  # Assuming content is in the first column
            self.generate_metadata_and_write_to_csv(row)
        print(f"Metadata written to {self.output_csv}")


# Example Usage
df1 = pd.DataFrame(list())
os.remove("output_metadata.csv")
metadata_generator = MetadataGenerator(input_csv="markdownedData.csv", output_csv="output_metadata.csv")
print("hihi this is working")
metadata_generator.generate_metadata_for_csv()
