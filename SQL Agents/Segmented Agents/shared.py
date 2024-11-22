

from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
import os
from langchain.memory import ConversationBufferMemory
from typing import TypedDict

# Carregar variáveis de ambiente
load_dotenv()

# Inicializar memória compartilhada
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Definindo um esquema usando TypedDict
class StateSchema(TypedDict):
    """
    Define um esquema de dicionário tipado para descrever o estado do agente.
    - `query (str)`: Representa a consulta atual do usuário.
    - `state_data (dict)`: Contém dados relacionados ao estado atual da execução.
    """
    query: str
    state_data: dict

# Função para conectar ao banco de dados
def connect_to_database():
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", 3306)
    DB_NAME = "teste_idfy"
    database_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    print(f"Connecting to database: {database_url}")
    return SQLDatabase.from_uri(database_url)
