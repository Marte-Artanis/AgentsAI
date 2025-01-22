from langchain_text_splitters import RecursiveCharacterTextSplitter  # Para dividir textos em chunks (partes menores).
from langchain.document_loaders import PyMuPDFLoader  # Para carregar documentos PDF.
from langchain_openai import OpenAIEmbeddings  # Para criar embeddings usando a API OpenAI.
from langchain_community.vectorstores import Pinecone as PineconeVectorStore  # Para gerenciar vetores com Pinecone.
from langchain.chains import RetrievalQA  # Para criar uma cadeia que combina recuperação e QA.
from langchain_openai import ChatOpenAI  # Para usar o modelo de linguagem GPT.
from pinecone import Pinecone  # Biblioteca para interagir com Pinecone.
from dotenv import load_dotenv  # Para carregar variáveis de ambiente de um arquivo .env.
import os  # Biblioteca para manipulação do sistema operacional.
import zipfile  # Para manipular arquivos ZIP.

# Carregar a chave de API do OpenAI e Pinecone a partir do arquivo .env.
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')  # Obtém a chave da API do OpenAI.
pinecone_key = os.getenv('PINECONE_API_KEY')  # Obtém a chave da API do Pinecone.

zip_file_path = 'documentos.zip'
extracted_folder_path = 'docs'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path)

documents = []
for filename in os.listdir(extracted_folder_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(extracted_folder_path, filename)
        loader = PyMuPDFLoader(file_path)
        documents.extend(loader.load())

# Configurar o divisor de texto para criar chunks de tamanho específico.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Tamanho máximo de cada chunk.
    chunk_overlap=100,  # Quantidade de sobreposição entre chunks consecutivos.
    length_function=len  # Função para calcular o comprimento do texto.
)

# Dividir o conteúdo dos documentos em chunks.
chunks = text_splitter.create_documents([doc.page_content for doc in documents])

# Criar embeddings para os chunks usando o modelo 'text-embedding-ada-002' da OpenAI.
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

# Nome do índice Pinecone onde os vetores serão armazenados.
index_name = 'llm'

# Criar o armazenamento de vetores no Pinecone a partir dos chunks.
vector_store = PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)

# Consultas para serem respondidas com base nos documentos.
query_1 = '''Responda apenas com base no input fornecido. Qual o número do processo que trata de violação de normas ambientais pela Empresa de Construção?'''
query_2 = 'Responda apenas com base no input fornecido. Qual foi a decisão no caso de fraude financeira?'
query_3 = 'Responda apenas com base no input fornecido. Quais foram as alegações no caso de dano moral in re ipsa?'
query_4 = 'Responda apenas com base no input fornecido. Quais foram as alegações no caso do Número do Processo: 822162?'

# Configurar o modelo de linguagem GPT-3.5 com baixa temperatura (mais determinístico).
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.2)

# Configurar o retriever que realiza busca de similaridade nos vetores armazenados.
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})  
# `k=3` define que serão retornados os 3 documentos mais semelhantes.

# Criar a cadeia que combina recuperação e resposta baseada no LLM.
chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

# Exibir informações sobre o objeto retriever criado.
print(chain)

# Realizar consultas usando a cadeia de recuperação e QA.
answer_1 = chain.invoke(query_1)
answer_2 = chain.invoke(query_2)
answer_3 = chain.invoke(query_3)
answer_4 = chain.invoke(query_4)

print('Pergunta: ',answer_1['query'])
print('Resultado: ',answer_1['result'],'\n')
#---
print('Pergunta: ',answer_2['query'])
print('Resultado: ',answer_2['result'],'\n')
#---
print('Pergunta: ',answer_3['query'])
print('Resultado: ',answer_3['result'],'\n')
#---
print('Pergunta: ',answer_4['query'])
print('Resultado: ',answer_4['result'])