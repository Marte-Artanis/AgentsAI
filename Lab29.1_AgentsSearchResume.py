from langchain_openai import OpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Carregar a chave de API do OpenAI
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Configuração do cliente OpenAI
openai = OpenAI(
    model_name='gpt-3.5-turbo-instruct',
    frequency_penalty=2,  # Evitar repetições de palavras
    presence_penalty=2,   # Explorar novos tópicos
    temperature=1,        # Controle de aleatoriedade nas respostas
    max_tokens=1000,      # Máximo de tokens na resposta
    n=1)

# Usar DuckDuckGoSearchResults, que tem a descrição necessária
ddg_search = DuckDuckGoSearchRun()

# Criar o agente com a ferramenta de busca
agent_executor = create_python_agent(
    llm=openai,
    tool=ddg_search,
    verbose=True
)
# Já tem action, observation e thought.
# Definir o prompt para o agente
prompt_template = PromptTemplate(
    input_variables=['query'],
    template="""
    Pesquise na web sobre {query} e forneça um resumo abrangente sobre o assunto.
    """
)

# Definir a consulta
query = 'Fíodor Dostoiévski'

# Formatando o prompt
prompt = prompt_template.format(query=query)

# Executando o agente
response = agent_executor.invoke(prompt)

# Exibindo os resultados
print(f"Entrada: {response['input']}")
print(f"Saída: {response['output']}")
