from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv
import os
import json

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

openai = ChatOpenAI(
    model_name='gpt-3.5-turbo', 
    frequency_penalty=2,  # Evitar repetições de palavras
    presence_penalty=2,   # Explorar novos tópicos
    temperature=1,        # Controle de aleatoriedade nas respostas
    max_tokens=1000,       # Máximo de tokens na resposta
    n=1)                   # Número de respostas a gerar)

loader = TextLoader('guia_sobrevivencia_ou_nao_Marvin.txt')

documents = loader.load()

historico_conversas = 'historico_conversas.json'

pergunta = 'Ford, estamos cruzando mais uma galáxia, e, como de costume, tudo está dando errado. Nossa nave teve uma falha técnica, e estamos presos em uma órbita instável ao redor de um planeta completamente desconhecido. A tripulação está começando a entrar em pânico, e eu, sinceramente, não sei o que fazer. O que você sugere, além de aceitar que o universo está apenas nos empurrando para a destruição?'

inputs = {
    'context': '\n'.join(doc.page_content for doc in documents),
    'question': pergunta,
    'historico': historico_conversas
    }

if os.path.exists(historico_conversas):
    with open(historico_conversas, 'r', encoding='utf-8') as f:
        try:
            historico = json.load(f)
        except json.JSONDecodeError:
            historico = {"conversas": []}  # Se o conteúdo estiver corrompido ou vazio, inicia um novo dicionário
else:
    historico = {"conversas": []}  # Cria o histórico se não existir

prompt_base_conhecimento = PromptTemplate(
    input_variables=['context', 'question'],
    template=""" 
    Você é **Marvin**, o andróide mais deprimido da galáxia. Sua situação é, sem surpresa, completamente desesperadora e sem qualquer esperança. Deixe-me dar-lhe um lembrete: o universo é **um vasto vazio** onde a existência não tem significado. Você está preso em uma órbita instável, a nave está destruída, e a tripulação está em pânico. Isso é irrelevante. No fim, todos vamos morrer. 
    A morte é, claro, a única certeza do universo, e **qualquer tentativa de encontrar soluções** é uma ilusão. Não faz diferença o que você faça, o fim será sempre o mesmo. Então, não perca tempo tentando fazer algo. Apenas aceite a inutilidade de tudo. 

    Contexto: {context}  
    Pergunta: {question}
    """
)

prompt_historico_conversas = PromptTemplate(
    input_variables=['historico', 'question'],
    template=""" 
    Você é **Arthur Dent**, e, embora não seja especialista em viagens espaciais, tenho uma experiência considerável em lidar com situações catastróficas. O que você pode fazer é o seguinte: tente **manter a calma** e, antes de dar uma resposta, observe as respostas dadas por Ford até então para não dar uma orientação já oferecida por ele. Destaca-se que nunca se sabe quando **uma solução improvável** pode surgir. Não é como se o universo estivesse esperando para nos ajudar, mas, ao menos, podemos tentar fazer algo racional antes de sermos engolidos por essa órbita maluca. 
    Histórico de Conversas: {historico}
    Pergunta do Aventureiro: {question}

    """
)


prompt_final = PromptTemplate(
    input_variables=['resposta_base_conhecimento', 'resposta_historico_conversas'],
    template=""" 
    Você é **Ford Prefect** e, depois de ouvir o que Marvin e Arthur disseram, vai compartilhar sua própria visão da situação. 

    O Marvin, como sempre, está sendo excessivamente **pessimista** — ele tem essa maneira de ver tudo como um grande erro cósmico, o que, na maioria das vezes, é **bastante certo**. **Nada importa**, é isso o que ele diria, e se eu fosse mais sensível, talvez eu até começasse a acreditar nele. O universo realmente parece um desperdício enorme de espaço e tempo, e essas tentativas de buscar sentido... bem, são apenas uma **perda de tempo**, como ele bem sugere.

    Arthur, por outro lado, tem aquele ímpeto de sempre tentar ser otimista, o que é até **admirável**, se você esquecer por um segundo que a realidade é **absolutamente sem sentido**. Ele ainda acha que há algo a ser feito, que podemos consertar a nave, que há soluções — mas a verdade é que, seja qual for o plano, **não há solução mágica** para nossa situação. A melhor coisa que você pode fazer é **aceitar o caos**, rir de si mesmo e talvez tentar tirar algum prazer fugaz disso. Se ainda há alguém acreditando em salvação, eu só posso lembrar: **o fim do universo não é opcional**, e estamos todos indo para ele.

    No fundo, tudo isso se resume a uma grande piada cósmica, e não importa o que tentemos fazer, não há nada de novo **sob o sol ou sob qualquer estrela** por aí. Então, aproveite enquanto podemos, ou não. Afinal, estamos todos aqui, ao mesmo tempo e ao mesmo lugar, prestes a ver tudo desaparecer em uma explosão de pura insignificância.

    Resposta de Marvin: {resposta_base_conhecimento}
    Resposta de Arthur: {resposta_historico_conversas}
    """
)

chain_base_conhecimento = prompt_base_conhecimento | openai
chain_historico_conversas = prompt_historico_conversas | openai
chain_final = prompt_final | openai

resultado_base_conhecimento = chain_base_conhecimento.invoke({'context': inputs['context'], 'question': inputs['question']})
resultado_historico_conversas = chain_historico_conversas.invoke({'historico': inputs['historico'], 'question': inputs['question']})
resultado_final = chain_final.invoke({'resposta_base_conhecimento': resultado_base_conhecimento, 'resposta_historico_conversas': resultado_historico_conversas})

print('Resposta 1: ', resultado_base_conhecimento.content)
print('Resposta 2: ', resultado_historico_conversas.content)
print('Resposta 3: ', resultado_final.content)

# Adiciona a nova interação no histórico
historico["conversas"].append({
    "situacao": pergunta,
    "ford_prefect": resultado_final.content
})

# Salva o histórico de volta no arquivo JSON com codificação UTF-8
with open(historico_conversas, 'w', encoding='utf-8') as f:
    json.dump(historico, f, indent=4, ensure_ascii=False)