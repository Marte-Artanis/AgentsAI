from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

openai = OpenAI(
    model_name='gpt-3.5-turbo-instruct', 
    frequency_penalty=2,  # Evitar repetições de palavras
    presence_penalty=2,   # Explorar novos tópicos
    temperature=1,        # Controle de aleatoriedade nas respostas
    max_tokens=1000,       # Máximo de tokens na resposta
    n=1)                   # Número de respostas a gerar)


prompt_template = PromptTemplate.from_template('Indique brevemente os principais personagens da {era} da História de Senhor dos Anéis')
runnable_sequence = prompt_template | openai
output = runnable_sequence.invoke({'era': 'Primeira Era'})

print(output)