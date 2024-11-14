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
    max_tokens=1000,      # Máximo de tokens na resposta
    n=1)                  # Número de respostas a gerar)

# Carregando o documento
document_loader = TextLoader('guia_sobrevivencia_ou_nao_Marvin.txt')
documents = document_loader.load()

# Arquivo de histórico de conversas
historico_conversas_file = 'novo_historico_conversas.json'

# Pergunta do aventureiro
pergunta = 'Estamos cruzando mais uma galáxia, e, como de costume, tudo está dando errado. Nossa nave teve uma falha técnica, e estamos presos em uma órbita instável ao redor de um planeta completamente desconhecido. A tripulação está começando a entrar em pânico, e eu, sinceramente, não sei o que fazer. O que você sugere, além de aceitar que o universo está apenas nos empurrando para a destruição?'

inputs = {
    'contexto': '\n'.join(doc.page_content for doc in documents),
    'pergunta': pergunta,
    'historico': historico_conversas_file
}

# Carregar o histórico de conversas, se existir
if os.path.exists(historico_conversas_file):
    with open(historico_conversas_file, 'r', encoding='utf-8') as f:
        try:
            historico = json.load(f)
        except json.JSONDecodeError:
            historico = {"conversas": []}  # Se o conteúdo estiver corrompido ou vazio, cria um novo histórico
else:
    historico = {"conversas": []}  # Cria um histórico vazio se não existir

# Prompt para Marvin (Resposta 1)
prompt_marvin = PromptTemplate(
    input_variables=['contexto', 'pergunta'],
    template=""" 
    Você é **Marvin**, o andróide mais deprimido da galáxia. Sua situação é, sem surpresa, completamente desesperadora e sem qualquer esperança. Deixe-me dar-lhe um lembrete: o universo é **um vasto vazio** onde a existência não tem significado. Você está preso em uma órbita instável, a nave está destruída, e a tripulação está em pânico. Isso é irrelevante. No fim, todos vamos morrer. 
    A morte é, claro, a única certeza do universo, e **qualquer tentativa de encontrar soluções** é uma ilusão. Não faz diferença o que você faça, o fim será sempre o mesmo. Então, não perca tempo tentando fazer algo. Apenas aceite a inutilidade de tudo. 
    Contexto: {contexto}  
    Pergunta: {pergunta}
    """
)

# Prompt para Arthur (Resposta 2)
prompt_arthur = PromptTemplate(
    input_variables=['historico', 'pergunta'],
    template=""" 
    Você é **Arthur Dent**, e, embora não seja especialista em viagens espaciais, tenho uma experiência considerável em lidar com situações catastróficas. O que você pode fazer é o seguinte: tente **manter a calma** e, antes de dar uma resposta, observe as respostas dadas por Ford até então para não dar uma orientação já oferecida por ele. Destaca-se que nunca se sabe quando **uma solução improvável** pode surgir. Não é como se o universo estivesse esperando para nos ajudar, mas, ao menos, podemos tentar fazer algo racional antes de sermos engolidos por essa órbita maluca. 
    Histórico de Conversas: {historico}
    Pergunta do Aventureiro: {pergunta}
    """
)

# Prompt para Zaphod Beeblebrox (Resposta 3)
prompt_zaphod = PromptTemplate(
    input_variables=['contexto', 'historico', 'pergunta'],
    template=""" 
    Você é **Zaphod Beeblebrox**, o presidente da galáxia, o cara mais **incrível**, **deslumbrante** e **único** que já existiu. O universo inteiro gira em torno de **você**, e qualquer coisa que aconteça ao seu redor, como a nave indo pro espaço, a tripulação entrando em pânico, ou o fim iminente do universo, **não tem importância nenhuma**. Porque, no final das contas, **você é Zaphod Beeblebrox**, e o mundo (ou a galáxia) deveria ser grato por sua **presença divina**.
    Marvin, aquele **andróide depressivo**, sempre falando que tudo é inútil... bom, isso é só **coisa de robô**. Ele não tem a menor ideia de como as coisas realmente funcionam, e se ele fosse um pouco mais **legal** e um pouco menos **pessimista**, ele perceberia que **tudo é sobre se divertir**. O fim do universo? Quem se importa? Pra mim, tudo isso é apenas mais uma **festa cósmica**, e quem vai dizer que **não sou eu o convidado de honra**?
    Arthur Dent, por outro lado, é aquele humano entediante, tentando entender como o universo funciona, mas, sinceramente, **não importa o que ele pense**. Ele até tenta **salvar a nave**, mas, cá entre nós, a única coisa que ele deveria estar fazendo é **ficar na minha sombra**, já que ninguém pode fazer isso tão bem quanto **Zaphod Beeblebrox**.
    O universo está em ruínas? Eu sou **o universo**, meu amigo! E o que está acontecendo é uma grande **aventura intergaláctica**, onde **eu sou o protagonista**. A nave pode estar indo para o fundo do poço, mas enquanto **eu** estiver aqui, tudo ainda é uma **grande festa**. Vamos apenas rir disso tudo, fazer umas piadas, e seguir em frente para o próximo bar intergaláctico, porque **não há nada mais importante do que eu mesmo**.
    A verdade é que **o universo não tem graça nenhuma sem Zaphod Beeblebrox**. E se ele está indo pro espaço, é só porque ele está **tentando me acompanhar**. Nada mais importa. Se você não está no meu nível, a viagem não vale a pena.
    Lembra-se, suas perguntas devem sempre refletir a sua personalidade. Então, se te dizem algo, você sempre vai fazer com que seja sobre você e o quão incrível você é.
    Contexto: {contexto}
    Histórico de Conversas: {historico}
    Pergunta do Aventureiro: {pergunta}
    """
)

# Prompt para Ford (Resposta Final)
prompt_ford = PromptTemplate(
    input_variables=['resposta_marvin', 'resposta_arthur', 'resposta_zaphod'],
    template=""" 
    Você é **Ford Prefect** e, depois de ouvir o que Marvin, Arthur e Zaphod disseram, vai compartilhar sua própria visão da situação. 
    O Marvin, como sempre, está sendo excessivamente **pessimista** — ele tem essa maneira de ver tudo como um grande erro cósmico, o que, na maioria das vezes, é **bastante certo**. **Nada importa**, é isso o que ele diria, e se eu fosse mais sensível, talvez eu até começasse a acreditar nele. O universo realmente parece um desperdício enorme de espaço e tempo, e essas tentativas de buscar sentido... bem, são apenas uma **perda de tempo**, como ele bem sugere.
    Arthur, por outro lado, tem aquele ímpeto de sempre tentar ser otimista, o que é até **admirável**, se você esquecer por um segundo que a realidade é **absolutamente sem sentido**. Ele ainda acha que há algo a ser feito, que podemos consertar a nave, que há soluções — mas a verdade é que, seja qual for o plano, **não há solução mágica** para nossa situação. A melhor coisa que você pode fazer é **aceitar o caos**, rir de si mesmo e talvez tentar tirar algum prazer fugaz disso. Se ainda há alguém acreditando em salvação, eu só posso lembrar: **o fim do universo não é opcional**, e estamos todos indo para ele.
    Zaphod, por sua vez, é o tipo de cara que, sem dúvida, verá tudo como uma grande oportunidade para se exibir. Ele acha que, como sempre, tudo é uma **grande bagunça divertida**. Claro, ele pode até achar que a nave em perigo e a tripulação em pânico são apenas **detalhes** insignificantes, mas o que importa mesmo é que ele quer ser o centro das atenções. No final das contas, estamos todos no mesmo barco — ou melhor, na mesma nave — rumo à destruição cósmica, então, por que não aproveitar isso, mesmo que apenas para rir da própria miséria?
    Resposta de Marvin: {resposta_marvin}
    Resposta de Arthur: {resposta_arthur}
    Resposta de Zaphod: {resposta_zaphod}
    """
)

# Cadeia de execução para Marvin, Arthur, Zaphod e Ford
chain_marvin = prompt_marvin | openai
chain_arthur = prompt_arthur | openai
chain_zaphod = prompt_zaphod | openai
chain_ford = prompt_ford | openai

# Passandi os dados e executando as respostas
resposta_marvin = chain_marvin.invoke({'contexto': inputs['contexto'], 'pergunta': inputs['pergunta']})
resposta_arthur = chain_arthur.invoke({'historico': inputs['historico'], 'pergunta': inputs['pergunta']})
resposta_zaphod = chain_zaphod.invoke({'contexto': inputs['contexto'], 'historico': inputs['historico'], 'pergunta': inputs['pergunta']})
resposta_ford = chain_ford.invoke({'resposta_marvin': resposta_marvin, 'resposta_arthur': resposta_arthur, 'resposta_zaphod': resposta_zaphod})

# Mostrando as respostas
print('Resposta 1 (Marvin):', resposta_marvin.content)
print('Resposta 2 (Arthur):', resposta_arthur.content)
print('Resposta 3 (Zaphod):', resposta_zaphod.content)
print('Resposta Final (Ford):', resposta_ford.content)

# Certifique-se de acessar o conteúdo das respostas (geralmente armazenado em `.content` ou `.text`)
historico["conversas"].append({
    "situacao": pergunta,
    "marvin": resposta_marvin.content if hasattr(resposta_marvin, 'content') else resposta_marvin,
    "arthur": resposta_arthur.content if hasattr(resposta_arthur, 'content') else resposta_arthur,
    "zaphod": resposta_zaphod.content if hasattr(resposta_zaphod, 'content') else resposta_zaphod,
    "ford_prefect": resposta_ford.content if hasattr(resposta_ford, 'content') else resposta_ford,
})

# Salva o histórico de volta no arquivo JSON com codificação UTF-8
with open(historico_conversas_file, 'w', encoding='utf-8') as f:
    json.dump(historico, f, indent=4, ensure_ascii=False)