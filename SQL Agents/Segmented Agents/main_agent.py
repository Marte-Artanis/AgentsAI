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

    def classify_query(self, user_query):
        self.memory.chat_memory.messages.append(HumanMessage(content=user_query))

        classification_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Classifique a consulta como uma combinação dos seguintes tipos:
            - 'instagram posts' para consultas relacionadas ao Instagram.
            - 'meta ads' para consultas relacionadas ao Meta Ads.
            - 'google ads' para consultas relacionadas ao Google Ads.
            - 'crm' para consultas relacionadas ao CRM.
            - 'geral' para consultas amplas ou não relacionadas.
            Retorne os tipos relevantes, separados por vírgulas, ou 'geral' se não estiver relacionado a nenhum dos dados."""),
            HumanMessage(content=f"A consulta é: '{user_query}'")
        ])
        messages = classification_template.format_messages()
        response = self.openai.invoke(messages)

        # Processar a resposta em uma lista de agentes
        classifications = [c.strip() for c in response.content.lower().split(",")]
        print(classifications)
        return classifications


    def handle_query(self, user_query):
        print(f"[DEBUG] Received query: {user_query}")
        classifications = self.classify_query(user_query)

        try:
            # Se apenas um agente for necessário
            if len(classifications) == 1 and classifications[0] != "geral":
                agent = classifications[0]
                if agent == "instagram posts":
                    return self.instagram_posts_agent.invoke(user_query)["state"]["content"]
                elif agent == "meta ads":
                    return self.meta_ads_agent.invoke(user_query)["state"]["content"]
                elif agent == "google ads":
                    return self.google_ads_agent.invoke(user_query)["state"]["content"]
                elif agent == "crm":
                    return self.google_ads_agent.invoke(user_query)["state"]["content"]

            if len(classifications) > 1:
                responses = {}

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

                # Consolidate responses
                consolidated_response = self.consolidation_agent.invoke(
                    user_query,
                    responses.get("instagram", "Nenhuma análise do Instagram foi retornada."),
                    responses.get("meta ads", "Nenhuma análise do Meta Ads foi retornada."),
                    responses.get("google ads", "Nenhuma análise do Google Ads foi retornada."),
                    responses.get("crm", "Nenhuma análise do CRM foi retornada."),

                )
                return consolidated_response

            # Caso seja 'geral' ou nenhuma classificação relevante
            elif "geral" in classifications:
                return self.general_agent.invoke(user_query)

            else:
                return "Desculpe, não entendi sua consulta."

        except Exception as e:
            print(f"[ERROR] Error processing query: {e}")
            return f"Erro ao processar a consulta: {e}"

