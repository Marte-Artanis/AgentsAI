from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os
from langchain_community.utilities import SQLDatabase


# Carregar a chave de API do OpenAI
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

class InstagramPostsAgent:
    def __init__(self):
        self.db = self.connect_to_database()
        self.openai = ChatOpenAI(model_name="gpt-3.5-turbo")

    def connect_to_database(self):
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_HOST = os.getenv("DB_HOST", "localhost")
        DB_PORT = os.getenv("DB_PORT", 3306)
        DB_NAME = "teste_idfy"
        database_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        print(f"Connecting to database: {database_url}")
        return SQLDatabase.from_uri(database_url)

    def process_query(self, query):
        print(f"Processing query: {query}")
        sql_query = self.generate_sql_query(query)
        
        # Extrair apenas a consulta SQL, ignorando instruções extras do modelo
        sql_query = self.extract_sql_from_response(sql_query)
        print(f"Generated SQL Query: {sql_query}")
        
        # Corrigir para usar _execute em vez de execute
        result = self.db._execute(sql_query)
        return self.analyze_data(result)

    def generate_sql_query(self, query):
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="Você é um agente especializado em gerar consultas SQL para dados de posts e seguidores do Instagram."),
            HumanMessage(content=f"""Gere uma consulta SQL para obter dados das tabelas `instagram_posts` e `follower_data` com base na pergunta: '{query}'.
            
            A tabela `instagram_posts` possui as seguintes colunas:
            - `post_date` (datetime): Data e hora em que o post foi realizado.
            - `post_type` (varchar(10)): Tipo de post (ex.: imagem, vídeo, carrossel).
            - `caption` (text): Legenda do post.
            - `hashtags` (varchar(255)): Lista de hashtags associadas ao post.
            - `likes_count` (int): Número total de curtidas no post.
            - `comments_count` (int): Número total de comentários no post.
            - `shares_count` (int): Número total de compartilhamentos do post.
            - `views_count` (int): Número total de visualizações do post (caso seja um vídeo).
            - `impressions` (int): Número total de vezes que o post foi exibido.
            - `reach` (int): Alcance total do post (número de contas únicas alcançadas).
            - `saved_count` (int): Número total de vezes que o post foi salvo.
            - `location_tag` (varchar(100)): Localização associada ao post.
            - `media_url` (varchar(255)): URL da mídia associada ao post (imagem ou vídeo).
            - `media_id` (varchar(50)): ID da mídia associada ao post.
            - `ad_spend` (decimal(10,2)): Valor gasto com anúncios para promover o post (se aplicável).

            A tabela `follower_data` possui as seguintes colunas:
            - `date` (datetime): Data do registro das informações de seguidores.
            - `followers_count` (int): Número total de seguidores no momento do registro.
            - `new_followers` (int): Número de novos seguidores ganhos no período.
            - `lost_followers` (int): Número de seguidores perdidos no período.
            - `profile_views` (int): Número total de visualizações de perfil no período.
            - `reach` (int): Alcance total (número de contas únicas que visualizaram o perfil).
            - `impressions` (int): Número total de impressões (quantas vezes o perfil foi exibido).
            - `engaged_followers` (int): Número total de seguidores que interagiram com o perfil (ex.: curtidas, comentários, compartilhamentos).
            - `demographics` (JSON): Dados demográficos dos seguidores, armazenados em formato JSON (ex.: idade, localização, gênero).
            """)
        ])
        messages = chat_template.format_messages()
        sql_query_response = self.openai.invoke(messages)
        return sql_query_response.content

    def analyze_data(self, data):
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="Você é um analista de dados de Instagram que responde em linguagem natural com base nos dados fornecidos."),
            HumanMessage(content=f"Analise os seguintes resultados dos posts do Instagram e dados dos seguidores: {data}")
        ])
        messages = chat_template.format_messages()
        analysis_response = self.openai.invoke(messages)
        return analysis_response.content

    def extract_sql_from_response(self, response_content):
        # Extrair apenas a consulta SQL do conteúdo do modelo
        start = response_content.find("```sql")
        end = response_content.find("```", start + 1)
        if start != -1 and end != -1:
            return response_content[start + 6:end].strip()
        return response_content.strip()


class MetaAdsAgent:
    def __init__(self):
        self.db = self.connect_to_database()
        self.openai = ChatOpenAI(model_name="gpt-3.5-turbo")

    def connect_to_database(self):
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_HOST = os.getenv("DB_HOST", "localhost")
        DB_PORT = os.getenv("DB_PORT", 3306)
        DB_NAME = "teste_idfy"
        database_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        print(f"Connecting to database: {database_url}")
        return SQLDatabase.from_uri(database_url)

    def process_query(self, query):
        print(f"Processing query: {query}")
        sql_query = self.generate_sql_query(query)
        
        sql_query = self.extract_sql_from_response(sql_query)
        print(f"Generated SQL Query: {sql_query}")
        
        result = self.db._execute(sql_query)
        return self.analyze_data(result)

    def generate_sql_query(self, query):
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="Você é um agente especializado em criar consultas SQL para dados de anúncios do Meta."),
            HumanMessage(content=f"""Gere uma consulta SQL para obter dados de anúncios do Meta com base na pergunta: '{query}'.
                A tabela `meta_ads` possui as seguintes colunas:
                ad_id: ID único do anúncio.
                campaign_name: Nome da campanha associada ao anúncio.
                objective: Objetivo do anúncio (ex: brand awareness, lead generation).
                ad_set_name: Nome do conjunto de anúncios.
                targeting_criteria: Critérios de segmentação utilizados no anúncio.
                bidding_strategy: Estratégia de lances aplicada.
                placement: Localização onde o anúncio foi exibido (ex: Facebook, Instagram).
                creative_type: Tipo de criativo utilizado (ex: imagem, vídeo).
                call_to_action: Chamada para ação do anúncio (ex: "Saiba mais", "Compre agora").
                impressions: Número total de impressões do anúncio.
                clicks: Número total de cliques no anúncio.
                conversions: Número total de conversões geradas.
                ctr: Taxa de cliques (Click-through Rate).
                cpc: Custo por clique (Cost per Click).
                cpm: Custo por mil impressões (Cost per Mille).
                cost: Custo total do anúncio.
                revenue_generated: Receita gerada pelo anúncio.
                roas: Retorno sobre o investimento em anúncios (Return on Ad Spend).
                interaction_date: Data em que o anúncio foi exibido.""")
        ])
        messages = chat_template.format_messages()
        sql_query_response = self.openai.invoke(messages)
        return sql_query_response.content

    def analyze_data(self, data):
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="Você é um analista de dados de anúncios Meta que responde em linguagem natural com base nos dados fornecidos."),
            HumanMessage(content=f"Analise os seguintes resultados de anúncios Meta: {data}")
        ])
        messages = chat_template.format_messages()
        analysis_response = self.openai.invoke(messages)
        return analysis_response.content

    def extract_sql_from_response(self, response_content):
        start = response_content.find("```sql")
        end = response_content.find("```", start + 1)
        if start != -1 and end != -1:
            return response_content[start + 6:end].strip()
        return response_content.strip()

class ConsolidationAgent:
    def __init__(self):
        self.openai = ChatOpenAI(model_name="gpt-3.5-turbo")

    def create_fluent_response(self, instagram_response, meta_response):
        # Combina as respostas individuais em um texto coeso
        combined_content = f"""
        Análise dos dados do Instagram:
        {instagram_response}

        Análise dos dados do Meta Ads:
        {meta_response}
        """
        
        # Prompt para consolidar as respostas em um texto fluido
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="Você é um agente que consolida análises em um texto coeso."),
            HumanMessage(content=f"Consolide a análise a seguir em um único texto:\n{combined_content}")
        ])
        
        messages = chat_template.format_messages()
        consolidation_response = self.openai.invoke(messages)
        return consolidation_response.content


class MainAgent:
    def __init__(self):
        self.openai = ChatOpenAI(model_name="gpt-3.5-turbo")
        self.instagram_posts_agent = InstagramPostsAgent()
        self.meta_ads_agent = MetaAdsAgent()
        self.consolidation_agent = ConsolidationAgent()  # Adicionando o agente de consolidação

    def classify_query(self, user_query):
        # Detecta se é uma pergunta geral ou algo específico sobre bancos de dados
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="Você é um agente que classifica consultas em categorias com base no entendimento contextual. \
            Sua tarefa é analisar a consulta e determinar se ela é uma saudação/pergunta geral ou uma consulta para o banco de dados. \
            Considere o significado e o contexto da pergunta para decidir qual tipo de dado é mais adequado. \
            Existem dois tipos de dados principais atualmente: 'Meta Ads' e 'Instagram Posts'. \
            - 'Meta Ads' inclui dados relacionados a anúncios pagos, campanhas publicitárias e métricas de desempenho de anúncios. \
            - 'Instagram Posts' inclui dados sobre postagens de redes sociais, como curtidas, comentários, engajamento orgânico e métricas de rede social. \
            Se a consulta fizer sentido apenas para dados de anúncios e campanhas, classifique como 'meta ads'. \
            Se a consulta fizer sentido para dados de redes sociais e postagens orgânicas, classifique como 'instagram posts'. \
            Se a consulta fizer sentido para ambos ou parecer envolver dados mistos, classifique como 'ambos'. \
            Classifique como 'geral' se a consulta for uma saudação ou pergunta genérica não relacionada aos dados do banco."),

            HumanMessage(content=f"A pergunta é '{user_query}'. Classifique-a como 'geral', 'instagram posts', 'meta ads', ou 'ambos'.")
        ])
        messages = chat_template.format_messages()
        classification_response = self.openai.invoke(messages)
        
        classification = classification_response.content.strip().lower()
        print(f"Classification Response (raw): {classification_response}")
        print(f"Classification Result (processed): {classification}")

        # Retorna a classificação apropriada
        if "instagram posts" in classification:
            return "instagram posts"
        elif "meta ads" in classification:
            return "meta ads"
        elif "ambos" in classification:
            return "ambos"
        elif "geral" in classification:
            return "geral"
        else:
            print("Não foi possível classificar. Respondendo como 'geral'.")
            return "geral"

    def handle_query(self, user_query):
        print(f"Handling query: {user_query}")
        classification = self.classify_query(user_query)
        print(f"Classification Result: {classification}")
        
        if classification == "instagram posts":
            return self.instagram_posts_agent.process_query(user_query)
        elif classification == "meta ads":
            return self.meta_ads_agent.process_query(user_query)
        elif classification == "ambos":
            # Obtém respostas individuais dos agentes
            instagram_response = self.instagram_posts_agent.process_query(user_query)
            meta_response = self.meta_ads_agent.process_query(user_query)
            
            # Consolida as respostas
            return self.consolidation_agent.create_fluent_response(instagram_response, meta_response)
        elif classification == "geral":
            return self.general_response(user_query)

    def general_response(self, user_query):
        # Define uma resposta para saudações ou perguntas gerais
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="Você é um agente conversacional amigável que responde a perguntas gerais e saudações de forma amigável."),
            HumanMessage(content=f"{user_query}")
        ])
        messages = chat_template.format_messages()
        general_response = self.openai.invoke(messages)
        return general_response.content

# Exemplo de uso
if __name__ == "__main__":
    main_agent = MainAgent()
        
    while True:
        user_query = input("Você: ")
        
        if user_query.lower() == "sair":
            print("Encerrando o chat. Até mais!")
            break
        
        response = main_agent.handle_query(user_query)
        print(f"Agente: {response}")
