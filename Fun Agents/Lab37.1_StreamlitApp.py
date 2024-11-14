import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

# Descrições de personagens e idiomas
personagens_descricao = {
    "Azog": "Azog, o Deflorador de Anões, é o infame líder dos Orcs, temido em toda a Terra-média...",
    "Uruk-hai": "Os Uruk-hai são bestas de guerra forjadas na escuridão...",
    "Elfo": "Um elfo nobre e destemido, exilado de sua terra natal...",
    "Anão": "Este anão é uma sombra de seu antigo eu, forjado no fogo da guerra...",
    "Hobbit": "Um Hobbit aparentemente insignificante, capturado pelos Orcs...",
    "Troll": "O Troll da Montanha é uma criatura primitiva e indomável...",
    "Warg": "Um Warg é uma fera selvagem, ferozmente leal a Azog...",
    "Soldado de Gondor": "Um soldado de Gondor, capturado por Azog durante um ataque surpresa...",
    "Nazgûl": "O Nazgûl, um dos temíveis servos de Sauron...",
    "Capitão de Mordor": "Fiel a Sauron, o Capitão de Mordor observa Azog com um olhar astuto..."
}

idiomas_descricao = {
    "Westron": "A língua comum entre os povos da Terra-média. Exemplos de palavras: 'mellon' (amigo), 'loth' (flor), 'ciriath' (caminho).",
    "A língua dos Orcs": "Grosseira e áspera, cheia de malícia e crueldade. Exemplos de palavras: 'durbat' (matar), 'gûl' (mestre), 'bûb' (sangue).",
    "Sindarin": "Língua élfica, bonita e fluida, falada pelos Elfos. Exemplos de palavras: 'mellon' (amigo), 'na vedui' (finalmente), 'athrad' (coragem).",
    "Quenya": "Língua élfica antiga e formal, usada em cerimônias. Exemplos de palavras: 'yéni' (anos), 'lúme' (tempo), 'anda' (norte).",
    "Khuzdul": "A língua secreta dos anões, dura e com poucas palavras conhecidas fora de sua raça. Exemplos de palavras: 'baraz' (rocha), 'khazâd' (anões), 'zâram' (forja).",
    "Black Speech": "A língua de Mordor, falada pelos servos de Sauron. Exemplos de palavras: 'nazg' (anel), 'ghâsh' (fogo), 'gûl' (mestre).",
    "Adûnaic": "A língua dos homens de Númenor. Exemplos de palavras: 'naur' (fogo), 'ammûr' (sombra), 'gil' (estrela).",
    "A língua dos Trolls das Montanhas": "Uma linguagem bruta e primitiva. Exemplos de palavras: 'ug' (grande), 'shaz' (esmagar), 'ghash' (grunhido).",
    "O idioma dos Wargs": "A comunicação bruta entre as feras de Sauron. Exemplos de palavras: 'ug' (fogo), 'ghur' (caça), 'ragh' (matar)."
}

# Disponibilidade de personagens por período histórico
periodos_disponiveis = {
    "Azog": ["A ascensão de Azog à liderança dos Orcs", "A Batalha dos Cinco Exércitos", "A Guerra do Anel"],
    "Uruk-hai": ["A ascensão de Azog à liderança dos Orcs", "A Batalha dos Cinco Exércitos"],
    "Elfo": ["A Última Aliança entre Elfos e Homens", "A Guerra do Anel"],
    "Anão": ["A ascensão de Azog à liderança dos Orcs", "A Batalha dos Cinco Exércitos"],
    "Hobbit": ["A Batalha dos Cinco Exércitos", "A Guerra do Anel"],
    "Troll": ["A ascensão de Azog à liderança dos Orcs", "A Batalha dos Cinco Exércitos"],
    "Warg": ["A ascensão de Azog à liderança dos Orcs", "A Batalha dos Cinco Exércitos"],
    "Soldado de Gondor": ["A Guerra do Anel"],
    "Nazgûl": ["A Guerra do Anel", "A Primeira Guerra contra Sauron"],
    "Capitão de Mordor": ["A Guerra do Anel", "A Primeira Guerra contra Sauron"]
}

# Fatores históricos por personagem
fatores_historicos_por_personagem = {
    "Azog": {
        "A ascensão de Azog à liderança dos Orcs": ["A formação da legião de Azog", "A derrota dos Anões em Moria"],
        "A Batalha dos Cinco Exércitos": ["A luta pela sobrevivência na Batalha", "A morte de Thorin Escudo de Carvalho"],
        "A Guerra do Anel": ["A aliança com Sauron", "O ataque a Gondor"]
    },
    "Uruk-hai": {
        "A ascensão de Azog à liderança dos Orcs": ["A luta pela sobrevivência no exército de Azog", "A captura dos anões"],
        "A Batalha dos Cinco Exércitos": ["A luta contra os Elfos e Anões", "O massacre dos Exércitos do Oeste"]
    },
    "Elfo": {
        "A Última Aliança entre Elfos e Homens": ["A queda de Sauron e a destruição do Anel", "A batalha de Dagorlad"],
        "A Guerra do Anel": ["A ajuda a Aragorn na luta contra Sauron", "A queda de Minas Tirith"]
    },
    "Anão": {
        "A ascensão de Azog à liderança dos Orcs": ["A morte de Thrór", "A luta pela sobrevivência dos Anões"],
        "A Batalha dos Cinco Exércitos": ["A aliança com os Elfos", "A luta pelo tesouro de Erebor"]
    },
    "Hobbit": {
        "A Batalha dos Cinco Exércitos": ["A intervenção dos Hobbits na batalha", "A luta contra os Orcs em Erebor"],
        "A Guerra do Anel": ["A jornada para destruir o Um Anel", "A aliança com Aragorn"]
    },
    "Troll": {
        "A ascensão de Azog à liderança dos Orcs": ["A selvageria dos Trolls", "A brutalidade no campo de batalha"],
        "A Batalha dos Cinco Exércitos": ["A destruição causada pelos Trolls", "A guerra contra os Elfos e Anões"]
    },
    "Warg": {
        "A ascensão de Azog à liderança dos Orcs": ["A captura de Wargs para a batalha", "A lealdade feroz a Azog"],
        "A Batalha dos Cinco Exércitos": ["A luta dos Wargs contra os exércitos aliados", "A perseguição aos Anões e Elfos"]
    },
    "Soldado de Gondor": {
        "A Guerra do Anel": ["A defesa de Gondor contra os Orcs", "A luta pela sobrevivência na Torre de Cirith Ungol"]
    },
    "Nazgûl": {
        "A Guerra do Anel": ["A busca pelo Um Anel", "A perseguição a Frodo Bolsão"],
        "A Primeira Guerra contra Sauron": ["A batalha contra os homens e Elfos"]
    },
    "Capitão de Mordor": {
        "A Guerra do Anel": ["A aliança com os Orcs e Uruk-hai", "A batalha pela destruição dos Reinos Livres"],
        "A Primeira Guerra contra Sauron": ["A luta pela supremacia de Mordor", "A queda de Númenor"]
    }
}

# Inicializando a lista de mensagens se ainda não existir
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Layout da aplicação Streamlit
st.title("Interaja com o Personagem da Terra-média!")

# Seleção de "quem está falando"
personagem_select = st.selectbox("Selecione quem você é: ", list(personagens_descricao.keys()))

# Atualiza a seleção de períodos históricos com base no personagem
periodos_historicos_disponiveis = periodos_disponiveis.get(personagem_select, [])
periodo_historico_select = st.selectbox("Selecione em qual período histórico você está: ", periodos_historicos_disponiveis)

# Atualiza a seleção de fatores históricos com base no personagem e no período histórico
fatores_historicos_disponiveis = fatores_historicos_por_personagem.get(personagem_select, {}).get(periodo_historico_select, [])
fatores_historicos_select = st.selectbox("Selecione o fator histórico:", fatores_historicos_disponiveis)

# Seleção de idioma
idioma_select = st.selectbox("Selecione o idioma:", list(idiomas_descricao.keys()))
# Exibição da descrição do idioma
st.write(f"**Descrição do idioma selecionado:** {idiomas_descricao[idioma_select]}")

# Função de resposta com tradução
def gerar_resposta(user_input):
    # Obtenção da descrição do personagem
    personagem_descricao = personagens_descricao.get(personagem_select, "Você é um personagem desconhecido, sem uma descrição definida.")
    
    # Adiciona o contexto histórico e a descrição do personagem
    system_message_content = f"""
    Você é {personagem_select}, ou seja, {personagem_descricao}.
    O contexto histórico é o seguinte:
    - **Período Histórico**: {periodo_historico_select}
    - **Fatores Históricos**: {fatores_historicos_select}
    - **Idioma**: {idioma_select}. {idiomas_descricao[idioma_select]}
    
    Quanto ao idioma, utilize todas as palavras e expressões dele que você tenha conhecimento, colocando a respectiva tradução.
    """

    # Configuração do prompt para o modelo com os dados dinâmicos
    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_message_content),
            HumanMessage(content=user_input),
            AIMessage(content=f"{personagem_select}, com sua voz característica, começa a responder..."),
        ]
    )

    prompt = chat_template.format_messages(
        personagem=personagem_select,
        periodo_historico=periodo_historico_select,
        idioma=idioma_select,
        fatores_historicos=fatores_historicos_select
    )

    # Configuração do modelo OpenAI
    openai = ChatOpenAI(
        model_name='gpt-3.5-turbo', 
        frequency_penalty=2,
        presence_penalty=2,
        temperature=1,
        max_tokens=500,
        n=1
    )

    response = openai.invoke(prompt)
    
    return response.content

# Entrada do usuário para a mensagem
user_input = st.text_input("Sua mensagem:")

# Enviar mensagem
if st.button('Enviar Mensagem') and user_input:
    resposta = gerar_resposta(user_input)
    st.session_state.messages.append(f"Você: {user_input}")
    st.session_state.messages.append(f"{personagem_select}: {resposta}")

# Exibição do chat com interface de balões de chat
for msg in st.session_state.messages:
    if msg.startswith("Você:"):
        st.chat_message("user").markdown(msg)
    else:
        # Se o personagem usar um idioma diferente, mostra a tradução
        if personagem_select == "Elfo" and idioma_select == "Sindarin":
            traducao = "Tradução: " + msg  # Tradução simulada para exemplo
            st.chat_message(personagem_select).markdown(f"{msg}\n{traducao}")
        else:
            st.chat_message(personagem_select).markdown(msg)
