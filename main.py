from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("openai.env")      # reads the .env file
client = OpenAI()  # automatically uses OPENAI_API_KEY

response = client.responses.create(
    model="gpt-5.4-nano",
    input="Say hello in one sentence."
)

print(response.output_text)