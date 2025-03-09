import pandas as pd
import os
import csv
from dotenv import load_dotenv
from google import genai

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

    def generate_metadata_and_write_to_csv(self, row: str) -> dict:
        """
        Generate metadata for a single row using Gemini AI and return it as a dictionary.
        
        Args:
            row (str): The content of the row to generate metadata for
        
        Returns:
            dict: A dictionary with 'filename', 'metadata', and 'content'
        """
        prompt = (
    '''
    Given the following article, return a response as a dictionary with the following structure:
    {
        "filename": <name of the file. Put none if
        "content": "<the content of the article>",
        "metadata": {
            "filename": "<filename or 'none'>",
            "slug": "<slug or 'none'>",
            "title": "<title or 'none'>",
            "description": "<description or 'none'>",
            "keywords": ["<keyword1>", "<keyword2>", ..., "<keywordN>"]
        }
    }

    The content of the article should be the full article excluding the title.
    The metadata fields should be:
    - "filename": The name of the file. If no filename exists, return "none".
    - "slug": The end of the URL. If none is found, generate one based on the title or return "none".
    - "title": A concise, clear title for the article.
    - "description": A brief description (<=20 words) summarizing the article.
    - "keywords": A list of relevant keywords, comma-separated (<=10 keywords).

    Here's an example:

    Article:
    "Flare is a decentralized platform for building secure data-driven applications. It supports EVM-compatible smart contracts for data interoperability."

    Expected Output:
    {
        "content": "Flare is a decentralized platform for building secure data-driven applications. It supports EVM-compatible smart contracts for data interoperability.",
        "metadata": {
            "filename": "flare_intro",
            "slug": "flare",
            "title": "Introduction to Flare",
            "description": "Overview of Flare, a blockchain for data.",
            "keywords": ["flare-network", "blockchain", "data", "smart-contracts"]
        }
    }

    '''
            
        )

        response = self.client.models.generate_content(
            model='gemini-2.0-flash', 
            contents=prompt
        )
        print(response)
        
        content = response['content']
        metadata = response['metadata']
        
        # Assuming metadata is structured as follows
        filename = metadata.get('filename', 'none')
        slug = metadata.get('slug', 'none')
        title = metadata.get('title', 'none')
        description = metadata.get('description', 'none')
        keywords = metadata.get('keywords', 'none')
        
        return {
            'filename': filename,
            'metadata': {
                'slug': slug,
                'title': title,
                'description': description,
                'keywords': keywords
            },
            'content': content
        }

    def generate_and_save_metadata(self):
        """
        Reads the input CSV, generates metadata using Gemini for each row,
        and writes the results to the output CSV file.
        """
        # Read the CSV input file
        df = self.read_csv()

        # Prepare the output CSV
        with open(self.output_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=["Filename", "Metadata", "Content"])
            writer.writeheader()

            # Process each row in the input CSV
            for index, row in df.iterrows():
                content = row['content']  # Assuming the 'content' column in your CSV
                metadata = self.generate_metadata_and_write_to_csv(content)

                # Write metadata to the CSV file
                writer.writerow({
                    'Filename': metadata['filename'],
                    'Metadata': f"slug: {metadata['metadata']['slug']}, title: {metadata['metadata']['title']}, "
                                f"description: {metadata['metadata']['description']}, keywords: {metadata['metadata']['keywords']}",
                    'Content': metadata['content']
                })

# Example of usage:
input_csv = 'markdownedData.csv'  # Path to the input CSV
output_csv = 'output_metadata.csv'  # Path to the output CSV
metadata_generator = MetadataGenerator(input_csv, output_csv)
metadata_generator.generate_and_save_metadata()