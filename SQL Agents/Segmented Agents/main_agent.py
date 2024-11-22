from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from instagram_agent import InstagramPostsAgent
from meta_ads_agent import MetaAdsAgent
from consolidation_agent import ConsolidationAgent
from general_agent import GeneralAgent
from google_ads_agent import GoogleAdsAgent
from crm_agent import CRMAgent
from shared import memory

class MainAgent:
    def __init__(self):
        self.openai = ChatOpenAI(model_name="gpt-3.5-turbo")
        self.instagram_posts_agent = InstagramPostsAgent()
        self.meta_ads_agent = MetaAdsAgent()
        self.consolidation_agent = ConsolidationAgent()
        self.general_agent = GeneralAgent()  # Adicione o GeneralAgent
        self.google_ads_agent = GoogleAdsAgent()
        self.crm_agent = CRMAgent()

        self.memory = memory  # Memória compartilhada


    def classify_query(self, user_query, selected_tab=None):
        """
        Classifica a consulta do usuário com base no histórico de interações e na aba atualmente selecionada.
        """
    # Adiciona a pergunta do usuário à memória
        self.memory.chat_memory.messages.append(HumanMessage(content=user_query))

    # Formata o histórico da memória para ser usado no prompt
        formatted_memory = "\n".join(
            [f"{'User' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}" for msg in self.memory.chat_memory.messages]
        )

        print(formatted_memory)

    # Cria o prompt de classificação com as regras de decisão
        classification_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""
            Classifique a consulta como uma combinação dos seguintes tipos:
            - 'instagram posts' para consultas relacionadas ao Instagram.
            - 'meta ads' para consultas relacionadas ao Meta Ads.
            - 'google ads' para consultas relacionadas ao Google Ads.
            - 'crm' para consultas relacionadas ao CRM.
            - 'geral' para consultas amplas ou não relacionadas.
            Se a consulta atual for relacionada a uma anterior (ex.: pergunta sobre dados já fornecidos), mantenha o agente anterior. 
            Considere o contexto das conversas para tomar sua decisão:
            Aba atualmente selecionada: '{selected_tab}'.

            ### Regras de Decisão:
            1. **Prioridade da Aba Selecionada**: Se a consulta for exclusivamente relevante para a aba selecionada, escolha o agente correspondente.
            2. **Consulta Envolve Outras Áreas**: Se a consulta do usuário envolver dados de outras tabelas ou áreas além da aba selecionada, inclua os agentes necessários, mesmo que a aba selecionada não esteja diretamente relacionada.
            3. **Combinação de Agentes**: Caso a consulta envolva múltiplos tópicos (ex.: dados de campanhas de Google Ads e Meta Ads), escolha todos os agentes relevantes.
            4. **Histórico e Continuidade**: Se a consulta estiver relacionada a uma interação anterior, priorize o agente que respondeu anteriormente, mesmo que ele não esteja relacionado à aba atual.

            Histórico de interações:
            {formatted_memory}
            Retorne os tipos relevantes, separados por vírgulas, ou 'geral' se não estiver relacionado a nenhum dos dados.
            """),
            HumanMessage(content=f"A consulta é: '{user_query}'")
        ])

    # Formata as mensagens para o modelo
        messages = classification_template.format_messages()

    # Invoca o modelo para classificar a consulta
        response = self.openai.invoke(messages)

    # Processa a resposta do modelo em uma lista de agentes
        classifications = [c.strip() for c in response.content.lower().split(",")]
        print(f"[DEBUG] Classifications: {classifications}")
        
        return classifications

    def handle_query(self, user_query, selected_tab=None):
        """
        Processa a consulta do usuário e invoca o(s) agente(s) apropriado(s) com base na classificação.
        Prioriza a aba selecionada, mas também considera múltiplos agentes ou interações anteriores.

        Args:
            user_query (str): Consulta feita pelo usuário.
            selected_tab (str, optional): Aba atualmente selecionada, usada para priorização.

        Returns:
            str: Resposta consolidada dos agentes ou uma mensagem de erro.
        """
        print(f"[DEBUG] Received query: {user_query}")

    # Classifica a consulta para determinar quais agentes devem ser invocados
        classifications = self.classify_query(user_query, selected_tab=selected_tab)
        print(f"[DEBUG] Selected Tab: {selected_tab}")

        try:
        # Caso apenas um agente seja necessário e não seja uma consulta geral
            if len(classifications) == 1 and classifications[0] != "geral":
                agent = classifications[0]
            # Invoca o agente correspondente
                if agent == "instagram posts":
                    return self.instagram_posts_agent.invoke(user_query)["state"]["content"]
                elif agent == "meta ads":
                    return self.meta_ads_agent.invoke(user_query)["state"]["content"]
                elif agent == "google ads":
                    return self.google_ads_agent.invoke(user_query)["state"]["content"]
                elif agent == "crm":
                    return self.google_ads_agent.invoke(user_query)["state"]["content"]

        # Caso a consulta envolva múltiplos agentes
            if len(classifications) > 1:
                responses = {}  # Caso a consulta envolva múltiplos agentes

            # Itera sobre os agentes classificados para invocá-los
                for agent in classifications:
                    try:
                        if agent == "instagram posts":
                            print("[DEBUG] Invoking 'InstagramPostsAgent'")
                            instagram_response = self.instagram_posts_agent.invoke(user_query).get("state", {}).get("content", "")
                            responses["instagram"] = instagram_response

                        elif agent == "meta ads":
                            print("[DEBUG] Invoking 'MetaAdsAgent'")
                            meta_response = self.meta_ads_agent.invoke(user_query).get("state", {}).get("content", "")
                            responses["meta ads"] = meta_response

                        elif agent == "google ads":
                            print("[DEBUG] Invoking 'GoogleAdsAgent'")
                            google_response = self.google_ads_agent.invoke(user_query).get("state", {}).get("content", "")
                            responses["google ads"] = google_response

                        elif agent == "crm":
                            print("[DEBUG] Invoking 'CRM'")
                            google_response = self.crm_agent.invoke(user_query).get("state", {}).get("content", "")
                            responses["google ads"] = google_response

                    except Exception as e:
                        print(f"[ERROR] Failed to invoke {agent}: {e}")
                        responses[agent] = f"Erro ao processar a análise para {agent}."

                print(f"[DEBUG] Collected responses: {responses}")

            # Consolidar as respostas usando o ConsolidationAgent
                consolidated_response = self.consolidation_agent.invoke(
                    user_query,
                    responses.get("instagram", "Nenhuma análise do Instagram foi retornada."),
                    responses.get("meta ads", "Nenhuma análise do Meta Ads foi retornada."),
                    responses.get("google ads", "Nenhuma análise do Google Ads foi retornada."),
                    responses.get("crm", "Nenhuma análise do CRM foi retornada."),

                )
                return consolidated_response

        # Caso a classificação seja 'geral' ou nenhuma classificação relevante foi encontrada
            elif "geral" in classifications:
                return self.general_agent.invoke(user_query)

        # Caso nenhuma classificação seja relevante
            else:
                return "Desculpe, não entendi sua consulta."

        except Exception as e:
        # Captura erros inesperados durante o processamento
            print(f"[ERROR] Error processing query: {e}")
            return f"Erro ao processar a consulta: {e}"

