from langchain_openai import ChatOpenAI, OpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

# Variáveis para personalizar a interação
pessoa = "um servo Orc insubordinado"  # O servo é um Orc desobediente que desafia Azog
periodo_historico = "Segunda Era, quando Azog ascendeu ao poder após a morte do Rei Thrór"  # Azog começa a se estabelecer como líder dos Orcs
idioma = "Mistura de Westron com a língua dos Orcs"  # O idioma reflete uma mistura ameaçadora e rude
fatores_historicos = "Azog matou Thrór, o Rei dos Anões, e assumiu o comando dos Orcs. A ascensão do seu poder é marcada por sua crueldade e vingança contra os anões, enquanto busca expandir sua influência."  # Contexto histórico da ascensão de Azog

chat_template = ChatPromptTemplate.from_messages(
        [
            # SystemMessage define o papel de Azog (ou outro líder) e o contexto
            SystemMessage(content=f"""
            Você é Azog, o Grande Orc, líder temido dos Orcs. Sua missão é manter a ordem e a disciplina entre seus subordinados, especialmente agora que está consolidando seu poder.
            O contexto histórico é o seguinte:
            - **Quem está falando com você**: {pessoa}
            - **Período Histórico**: {periodo_historico}
            - **Idioma**: {idioma} (uma mistura ameaçadora de Westron com a língua dos Orcs)
            - **Fatores Históricos**: {fatores_historicos}
            Você deve responder com autoridade e firmeza, não aceitando nenhum tipo de insubordinação. O servo está desafiando sua autoridade, então não hesite em puni-lo como for necessário.
            """),
            # HumanMessagePromptTemplate com um servo desafiando Azog
            HumanMessagePromptTemplate.from_template(f'{pessoa} fala de forma desrespeitosa: "Você matou Thrór, mas não será mais do que um líder de Orcs caídos. O que você pode fazer além de ameaçar?"'),
            # AIMessage (Azog, a resposta do líder Orc)
            AIMessage(content="Você ousa me desafiar, {pessoa}? Eu matei Thrór e conquistei Moria. Você se esqueceu de quem está no comando aqui? Vou mostrar o que acontece com aqueles que não sabem seu lugar."),
            # HumanMessage (O servo, mais insubordinado)
            HumanMessage(content=f"{pessoa} responde com desprezo: 'Você pode ter matado Thrór, mas seus Orcs estão enfraquecidos e em ruínas. A queda de Moria é inevitável!'"),
            # AIMessage (Azog com mais firmeza e ameaças)
            AIMessage(content="Você está cego pela sua tolice. Moria é minha, e todos que desafiarem meu domínio pagarão com a vida. Você não passa de um peão, e agora, você verá o verdadeiro poder de Azog!")
        ]
    )

prompt = chat_template.format_messages(
    pessoa=pessoa,
    periodo_historico=periodo_historico,
    idioma=idioma,
    fatores_historicos=fatores_historicos
    )

openai = ChatOpenAI(
    model_name='gpt-3.5-turbo', 
    frequency_penalty=2,  # Evitar repetições de palavras
    presence_penalty=2,   # Explorar novos tópicos
    temperature=1,        # Controle de aleatoriedade nas respostas
    max_tokens=500,       # Máximo de tokens na resposta
    n=1)                   # Número de respostas a gerar)

response = openai.invoke(prompt)
print(response.content)