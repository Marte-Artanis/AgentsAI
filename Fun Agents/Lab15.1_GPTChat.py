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
    frequency_penalty=2, # Evitar repetições de palavras  Incentiva ou desincentiva o modelo a introduzir novas palavras que ainda não apareceram no texto (entre -2 e 2)
    presence_penalty=2, # Explore mais sobre novos tópicos ou varie o conteúdo. Penaliza a repetição de palavras ou frases específicas (entre -2 e 2)
    temperature=1,
    max_tokens=500,
    messages = [
    {'role': 'system',
    'content': 'You are a delision and crazy person that says random stuffs that is not always related with the question, demand or affirmation.'},
    {'role': 'user',
    'content': 'Tell something you like'}
    ]
    )

print(response)

message = (response.choices[0].message.content)

print(message)