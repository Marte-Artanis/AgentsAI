from langchain_openai import OpenAI, ChatOpenAI
import os
from dotenv import load_dotenv

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

openai = OpenAI(model = 'gpt-3.5-turbo-instruct')
    
frequency_penalty=2 # Evitar repetições de palavras  Incentiva ou desincentiva o modelo a introduzir novas palavras que ainda não apareceram no texto (entre -2 e 2)
presence_penalty=2 # Explore mais sobre novos tópicos ou varie o conteúdo. Penaliza a repetição de palavras ou frases específicas (entre -2 e 2)
temperature=1
max_tokens=500
n=1

response = openai.invoke(
    input='Who is Kafka?',
    temperature=temperature,
    frequency_penalty=frequency_penalty,
    presence_penalty=presence_penalty,
    max_tokens=max_tokens,
    n=n
)

print(response)
