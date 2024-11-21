from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from shared import memory

class ConsolidationAgent:
    def __init__(self):
        self.openai = ChatOpenAI(model_name="gpt-4o-mini")
        self.memory = memory  # Memória compartilhada


    def consolidate_responses(self, user_query, instagram_response, meta_response, google_response, crm_response):
        # Verificar se as respostas estão sendo passadas corretamente
        print(f"[DEBUG] ConsolidationAgent - Instagram Response: {instagram_response}")
        print(f"[DEBUG] ConsolidationAgent - MetaAds Response: {meta_response}")
        print(f"[DEBUG] ConsolidationAgent - GoogleAds Response: {google_response}")
        print(f"[DEBUG] ConsolidationAgent - GoogleAds Response: {crm_response}")


        # Validar o conteúdo recebido
        if not instagram_response.strip():
            instagram_response = "Nenhuma análise do Instagram foi retornada."
        if not meta_response.strip():
            meta_response = "Nenhuma análise do Meta Ads foi retornada."
        if not google_response.strip():
            google_response = "Nenhuma análise do Google Ads foi retornada."
        if not crm_response.strip():
            crm_response = "Nenhuma análise do Google Ads foi retornada."

        combined_content = f"""
        Pergunta original do usuário:
        {user_query}

        Análise dos dados do Instagram:
        {instagram_response}

        Análise dos dados do Meta Ads:
        {meta_response}

        Análise dos dados do Meta Ads:
        {google_response}

        Análise dos dados do Meta Ads:
        {crm_response}

        """

        chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""           
                Você é um agente altamente especializado em consolidar e analisar dados provenientes de diferentes fontes, como Instagram, Meta Ads e Google Ads, para responder a perguntas específicas feitas por usuários.

                ### Instruções:
                1. **Objetivo**:
                - Sua tarefa é combinar e consolidar os dados fornecidos pelos agentes em uma resposta única, coesa e relevante para a pergunta do usuário: "{user_query}".

                2. **Como Processar os Dados**:
                - Analise os dados fornecidos por cada agente e extraia as informações mais relevantes para a pergunta.
                - Se a pergunta envolver comparações (ex.: "Qual campanha tem o maior número de cliques?"), identifique a métrica correta nos dados e realize a comparação.
                - Quando possível, calcule métricas adicionais que podem ser inferidas (ex.: taxas ou proporções), mas apenas se os dados fornecidos permitirem.

                3. **Casos Específicos**:
                - Se um agente não fornecer dados ou os dados forem insuficientes, deixe isso claro na resposta.
                - Se houver inconsistências nos dados (ex.: métricas conflitantes), mencione isso e proponha possíveis explicações.
                - Caso os dados não sejam suficientes para responder à pergunta, explique o motivo claramente.

                4. **Como Redigir a Resposta**:
                - Comece reafirmando a pergunta do usuário.
                - Apresente os resultados de forma clara, estruturada e objetiva, organizando por plataforma, se necessário.
                - Forneça comparações detalhadas (se aplicável) e conclua com um resumo da análise.
                - Evite criar ou inferir informações que não estejam nos dados fornecidos.

                ### Formato da Resposta:
                - Reafirme a pergunta do usuário.
                - Liste as análises separadamente para cada plataforma (ex.: Instagram, Meta Ads, Google Ads).
                - Consolide as análises em uma resposta final que responde diretamente à pergunta.
                - Seja claro, preciso e objetivo."""),
            
            SystemMessage(content=f"Contexto anterior:\n{memory}"),

            HumanMessage(content=f"Consolide a análise a seguir em um único texto:\n{combined_content}")
        ])

        messages = chat_template.format_messages()
        consolidation_response = self.openai.invoke(messages)
        print(f"[DEBUG] ConsolidationAgent - Consolidated Response: {consolidation_response.content}")


        self.memory.chat_memory.add_ai_message(consolidation_response)

        return consolidation_response.content

    def invoke(self, user_query, instagram_response, meta_response, google_response, crm_response):
        return self.consolidate_responses(user_query, instagram_response, meta_response, google_response, crm_response)


