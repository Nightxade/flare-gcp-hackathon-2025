import pandas as pd
import os
import csv
from dotenv import load_dotenv
from google import genai

load_dotenv()

class MetadataGenerator:
    def __init__(self, input_csv: str, output_csv: str) -> None:
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)

    def read_csv(self) -> pd.DataFrame:
        """Reads the input CSV file into a pandas DataFrame."""
        return pd.read_csv(self.input_csv, encoding='utf-8')

    def generate_metadata_and_write_to_csv(self, row_content: str) -> dict:
        """Generate metadata for a single row (string output) and return as a dictionary."""
        prompt = '''
        Given the following article, return a response as a string formatted like this:
        "Filename: <name of the file. Put none if none>
        Content: <the content of the article>
        Metadata:
            filename: <filename or 'none'>
            slug: <slug or 'none'>
            title: <title or 'none'>
            description: <description or 'none'>
            keywords: <keyword1, keyword2, ...>"

        The content should be the full article excluding the title.
        The metadata fields should be:
        - "filename": The name of the file. If no filename exists, return "none".
        - "slug": The end of the URL. If none is found, generate one based on the title or return "none".
        - "title": A concise, clear title for the article.
        - "description": A brief description (<=20 words) summarizing the article.
        - "keywords": A list of relevant keywords, comma-separated (<=10 keywords).
        '''
        
        response = self.client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        
        print(response)  # Inspect the output string structure

        # Initialize the variables for metadata and content
        content = ""
        metadata = {"filename": "none", "slug": "none", "title": "none", "description": "none", "keywords": []}
        
        # Parse the response string (assuming it's formatted correctly)
        lines = response.splitlines()

        for line in lines:
            if line.startswith("Filename:"):
                metadata['filename'] = line.split(":")[1].strip()
            elif line.startswith("Content:"):
                content = line.split(":")[1].strip()
            elif line.startswith("Metadata:"):
                # Extract metadata fields
                for i in range(1, len(lines)):
                    meta_line = lines[i].strip()
                    if meta_line.startswith("filename:"):
                        metadata['filename'] = meta_line.split(":")[1].strip()
                    elif meta_line.startswith("slug:"):
                        metadata['slug'] = meta_line.split(":")[1].strip()
                    elif meta_line.startswith("title:"):
                        metadata['title'] = meta_line.split(":")[1].strip()
                    elif meta_line.startswith("description:"):
                        metadata['description'] = meta_line.split(":")[1].strip()
                    elif meta_line.startswith("keywords:"):
                        keywords_str = meta_line.split(":")[1].strip()
                        metadata['keywords'] = [kw.strip() for kw in keywords_str.split(",")]

        return {
            'filename': metadata['filename'],
            'metadata': {
                'slug': metadata['slug'],
                'title': metadata['title'],
                'description': metadata['description'],
                'keywords': metadata['keywords']
            },
            'content': content
        }

    def generate_and_save_metadata(self):
        """Reads the input CSV, generates metadata using Gemini for each row, and writes the results to the output CSV."""
        # Read the CSV input file
        df = self.read_csv()

        # Prepare the output CSV
        with open(self.output_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=["Filename", "Metadata", "Content"])
            writer.writeheader()

            # Dynamically find the column that contains the content
            content_column = None
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, str)).all():  # Look for the first text column
                    content_column = col
                    break

            if not content_column:
                raise ValueError("No suitable content column found in the CSV.")

            # Process each row in the input CSV
            for index, row in df.iterrows():
                content = row.get(content_column, '')  # Use dynamic content column
                metadata = self.generate_metadata_and_write_to_csv(content)

                # Write metadata to the CSV file
                writer.writerow({
                    'Filename': metadata['filename'],
                    'Metadata': f"slug: {metadata['metadata']['slug']}, title: {metadata['metadata']['title']}, "
                                f"description: {metadata['metadata']['description']}, keywords: {metadata['metadata']['keywords']}",
                    'Content': metadata['content']
                })

# Example usage:
input_csv = 'markdownedData.csv'  # Path to the input CSV
output_csv = 'output_metadata.csv'  # Path to the output CSV
metadata_generator = MetadataGenerator(input_csv, output_csv)
metadata_generator.generate_and_save_metadata()
