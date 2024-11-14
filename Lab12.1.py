from openai import OpenAI
import os
from dotenv import load_dotenv

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

OpenAI.api_key = openai_api_key

client = OpenAI()

response = client.chat.completions.create(
    model = 'gpt-3.5-turbo',
    messages = [
    {'role': 'user',
    'content': 'Conte um funny fact'}
    ]
    )

print(response)

message = (response.choices[0].message.content)

print(message)