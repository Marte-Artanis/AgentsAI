from langchain_openai import OpenAI
from langchain.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache
import os
import json
import hashlib
from dotenv import load_dotenv

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

openai = OpenAI(model_name = 'gpt-3.5-turbo-instruct',
    model_name='gpt-3.5-turbo-instruct', 
    frequency_penalty=2,  # Evitar repetições de palavras
    presence_penalty=2,   # Explorar novos tópicos
    temperature=1,        # Controle de aleatoriedade nas respostas
    max_tokens=500,       # Máximo de tokens na resposta
    n=1)                   # Número de respostas a gerar)

frequency_penalty=2 # Evitar repetições de palavras  Incentiva ou desincentiva o modelo a introduzir novas palavras que ainda não apareceram no texto (entre -2 e 2)
presence_penalty=2 # Explore mais sobre novos tópicos ou varie o conteúdo. Penaliza a repetição de palavras ou frases específicas (entre -2 e 2)
temperature=1
max_tokens=500
n=1

class SimpleDiskCache:
    def __init__(self, cache_dir='cache_dir'):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok = True)
    
    def _get_cache_path(self, key):
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f'{hashed_key}.json')
    
    def lookup(self, key, llm_string):
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None
    
    def update(self, key, value, llm_string):
        cache_path = self._get_cache_path(key)
        with open(cache_path, 'w') as f:
            json.dump(value, f)


cache = SimpleDiskCache()

set_llm_cache(cache)

prompt = 'Quem foi Kafka?'


def invoke_with_cache(llm, prompt, cache):
    cached_response = cache.lookup(prompt, '')
    if cached_response:
        print('Usando cache: ')
    
    response = llm.invoke(prompt)
    cache.update(prompt, response, '')
    return response

first_response = invoke_with_cache(openai, prompt, cache)
first_response_text = first_response.replace('\n', ' ')
print('Primeira resposta: ', first_response_text)


second_response = invoke_with_cache(openai, prompt, cache)
second_response_text = second_response.replace('\n', ' ')
print('Segunda resposta: ', second_response_text)
