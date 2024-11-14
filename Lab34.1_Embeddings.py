from langchain_text_splitters import RecursiveCharacterTextSplitter # Cria chunks
from langchain_openai import OpenAIEmbeddings
from numpy import dot, array # Dot =  Array = Verifica e imprime similaridade
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt # Impressão
from dotenv import load_dotenv
import os

# Carregar a chave de API do OpenAI
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

documents = [
    'Este é o primeiro documento. Ele contém informações importantes sobre o projeto',
    'Este é o segundo documento. Ele contém informações importantes sobre o projeto',
    'O terceiro documento oferece uma visão geral dos resultados esperados e métricas de sucesso'
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    length_function=len
)

chunks = text_splitter.create_documents(documents)

# Antes de aplicação os embbeddings, divide-se em chunks

print('Chunks gerados:')
for i, chunk in enumerate(chunks):
    print(f'Chunk {i+1}: {chunk.page_content}')

print(f'Número total de chunks: {len(chunks)}')

embeddings = OpenAIEmbeddings(
    model='text-embedding-ada-002'
)

embedded_chunks = embeddings.embed_documents([chunk.page_content for chunk in chunks])

# Função para calcular similaridade coseno
def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

similarities = []
for i in range(len(embedded_chunks)):
    for j in range(i + 1, len(embedded_chunks)):
        similarity = cosine_similarity(embedded_chunks[i], embedded_chunks[j])
        similarities.append((i, j, similarity))
        print(f'Similaridade entre os chunks {i + 1} e {j + 1}: {similarity:.2f}')

embedded_chunks_array = array(embedded_chunks)

# Visualização com PCA: pega um espaço com muitas dimensões, e diminui.
pca = PCA(n_components=2)
pca_results = pca.fit_transform(embedded_chunks_array)

plt.figure(figsize=(10, 7))
plt.scatter(pca_results[:, 0], pca_results[:, 1], c='blue', edgecolors='k', s=50)
for i, chunk in enumerate(chunks):
    plt.text(pca_results[i, 0], pca_results[i, 1], f'Chunk {i + 1}', fontsize=12)

for (i, j, similarity) in similarities:
    plt.plot([pca_results[i, 0], pca_results[j, 0]], [pca_results[i, 1], pca_results[j, 1]], 'k--', alpha=similarity)
    md_x = (pca_results[i, 0] + pca_results[j, 0]) / 2
    md_y = (pca_results[i, 1] + pca_results[j, 1]) / 2

plt.title('Visualização dos Embeddings com PCA')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.grid(True)
plt.show()

# Visualização com t-SNE
tsne = TSNE(n_components=2, perplexity=2, n_iter=300)
tsne_results = tsne.fit_transform(embedded_chunks_array)

plt.figure(figsize=(10, 7))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c='green', edgecolors='k', s=50)
for i, chunk in enumerate(chunks):
    plt.text(tsne_results[i, 0], tsne_results[i, 1], f'Chunk {i + 1}', fontsize=12)

plt.title('Visualização dos Embeddings com t-SNE')
plt.xlabel('Dimensão 1')
plt.ylabel('Dimensão 2')
plt.grid(True)
plt.show()
