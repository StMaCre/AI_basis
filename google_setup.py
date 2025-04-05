# Import libraries 
from google import genai
from google.genai import types
from IPython.display import Markdown
from dotenv import load_dotenv
import os

#!pip install -qU "google-genai==1.7.0" "chromadb==0.6.3"
#!pip install python-dotenv

# This function sets up the Google GenAI client with the specified model name.
# It loads the API key from environment variables and returns a function to generate text using the model.
# The function `generate_text` takes a prompt and an optional configuration object as input and returns the generated text.
def setup_genai_client(model_name="gemini-2.0-flash"):
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY not found in environment variables.")
    
    client = genai.Client(api_key=api_key)

    def generate_text(prompt, config=None):
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt],
            config=config
        )
        return response.text

    return generate_text
