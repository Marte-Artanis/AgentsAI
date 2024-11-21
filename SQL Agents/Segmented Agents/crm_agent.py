from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor, as_completed
from shared import connect_to_database, StateSchema, memory

class CRMAgent:
    def __init__(self):
        self.db = connect_to_database()
        self.openai = ChatOpenAI(model_name="gpt-4o-mini")
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

        # Definindo o StateGraph
        self.graph = StateGraph(state_schema=StateSchema)
        self.setup_states()  # Configura estados e transições

        self.memory = memory  # Memória compartilhada

    def setup_states(self):
        # Adiciona nós ao grafo para cada estado
        self.graph.add_node("GenerateSQL", self.generate_sql)
        self.graph.add_node("ExecuteSQL", self.execute_sql)
        self.graph.add_node("AnalyzeData", self.analyze_data)

        # Define transições entre os estados
        self.graph.add_edge("GenerateSQL", "ExecuteSQL")
        self.graph.add_edge("ExecuteSQL", "AnalyzeData")

        # Define o ponto de entrada e de saída
        self.graph.set_entry_point("GenerateSQL")
        self.graph.set_finish_point("AnalyzeData")

        # Compila o grafo
        self.graph.compile()

    def generate_sql(self, state_data, user_query):
        query = state_data["query"]  # Use o 'query' do state_data
        print(f"[DEBUG] Generating SQL for query: {query}")
        print(f"[DEBUG] User Query: {user_query}")
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
                Você é um agente especializado em criar consultas SQL para dados de anúncios do CRM. 
                
                ### Recomendações de Sintaxe SQL:
                1. Para consultas que pedem o "maior" ou "menor" valor:
                - Use `ORDER BY coluna DESC LIMIT 1` para encontrar o maior valor.
                - Use `ORDER BY coluna ASC LIMIT 1` para encontrar o menor valor.
                - Evite usar `MAX` ou `MIN` em conjunto com `GROUP BY`.

                2. Para evitar problemas com o modo `only_full_group_by`:
                - Certifique-se de que todas as colunas mencionadas na consulta (que não são funções de agregação) estejam na cláusula `GROUP BY`.
                - Sempre que calcular métricas agregadas como `SUM`, `AVG` ou `COUNT`, garanta que as colunas necessárias estejam agrupadas corretamente.

                **Instruções específicas**:
                - Analise apenas as partes da pergunta que se referem diretamente a dados de anúncios do CRM.
                - Ignore e não tente responder ou considerar partes da pergunta que envolvam métricas ou dados fora de anúncios, como métricas específicas de posts do Instagram (por exemplo, curtidas, comentários, compartilhamentos, visualizações).
                - Se a pergunta contiver múltiplas partes, extraia e responda apenas àquelas relacionadas aos dados de anúncios do CRM.
            """),

            SystemMessage(content=f"Contexto anterior:\n{memory}"),

            HumanMessage(content=f"""Gere uma consulta SQL para obter dados de anúncios do CRM com base na pergunta: '{query}'.
                A tabela `crm` possui as seguintes colunas:
                crm_id: ID único do CRM (chave primária).
                cliente_id: ID único do cliente.
                nome_cliente: Nome completo do cliente.
                email_cliente: Endereço de e-mail do cliente.
                telefone_cliente: Número de telefone do cliente.
                cidade_cliente: Cidade onde o cliente está localizado.
                estado_cliente: Estado onde o cliente está localizado.
                pais_cliente: País onde o cliente está localizado.
                valor_venda: Valor da venda realizada (em moeda decimal).
                data_venda: Data em que a venda foi realizada.
                status_venda: Status atual da venda (ex: "Concluída", "Pendente").
                metodo_pagamento: Método de pagamento utilizado na venda (ex: "Cartão de Crédito", "Boleto").
                campanha_id: ID único da campanha associada à venda.
                nome_campanha: Nome da campanha de marketing associada.
                objetivo_campanha: Objetivo principal da campanha (ex: "Aumentar Vendas", "Gerar Leads").
                data_inicio_campanha: Data de início da campanha.
                data_fim_campanha: Data de término da campanha.
                orcamento_campanha: Orçamento alocado para a campanha (em moeda decimal).
                resposta_campanha: Feedback obtido com a campanha (ex: "Positiva", "Negativa").
                tipo_interacao: Tipo de interação realizada com o cliente (ex: "E-mail", "Ligação").
                data_interacao: Data em que a interação foi realizada.
                resultado_interacao: Resultado da interação (ex: "Sucesso", "Sem Retorno").
                segmento_id: ID único do segmento de mercado do cliente.
                nome_segmento: Nome do segmento de mercado do cliente.
                ticket_suporte_id: ID único do ticket de suporte associado ao cliente.
                data_abertura_ticket: Data em que o ticket de suporte foi aberto.
                data_fechamento_ticket: Data em que o ticket de suporte foi fechado.
                status_ticket: Status atual do ticket de suporte (ex: "Resolvido", "Pendente").
                tipo_problema: Tipo de problema reportado pelo cliente.
                total_gasto: Total de gastos associados ao cliente ou campanha (em moeda decimal).
                retorno_sobre_investimento: Retorno sobre o investimento (ROI) obtido.
                quantidade_interacoes: Quantidade total de interações realizadas com o cliente.
                taxa_conversao: Taxa de conversão obtida.
                feedback_cliente: Feedback geral do cliente após as interações.""")
        ])
        messages = chat_template.format_messages()
        sql_query_response = self.openai.invoke(messages)
        sql_query = self.extract_sql_from_response(sql_query_response.content)
        state_data["sql_query"] = sql_query

        self.memory.chat_memory.add_ai_message(sql_query)

        print(f"[DEBUG] Generated SQL Query: {sql_query}")

    def execute_sql(self, state_data):
        sql_query = state_data.get("sql_query", "")  # Pega o SQL da consulta
        print(f"[DEBUG] Executing SQL Query: {sql_query}")
        try:
            result = self.db._execute(sql_query)
            state_data["sql_result"] = result
            print(f"[DEBUG] Raw execution result: {result}")
        except Exception as e:
            print(f"[ERROR] SQL execution failed: {e}")
            state_data["sql_result"] = f"Error in SQL execution: {e}"

    def analyze_data(self, state_data, user_query):
        data = state_data["sql_result"]
        query = state_data["query"]  # Pergunta original do usuário
        
        print(f"[DEBUG] Generating SQL for query: {query}")  # Usa 'query' corretamente aqui

        print(f"[DEBUG] Data received in analyze_data: {data}")
        print(f"[DEBUG] Query do usuário: {user_query}")

        # Convertendo os dados em texto formatado
        data_as_text = "\n".join(str(entry) for entry in data)

        # Dividindo em chunks, caso necessário
        chunks = self.splitter.split_text(data_as_text)

        # Função para processar um chunk
        def process_chunk(chunk):
            print(f"[DEBUG] Processing chunk: {chunk}")
            chat_template = ChatPromptTemplate.from_messages([
                SystemMessage(content=f"""Você é um analista de dados que processa informações relacionadas a campanhas de marketing digital.
                ### Instruções de resposta:
                - Analise os dados fornecidos no contexto da seguinte pergunta do usuário: "{user_query}".
                - Extraia e destaque os principais pontos que respondem diretamente à pergunta do usuário.
                - Evite informações irrelevantes ou que não respondam à pergunta.
                - Nunca invente dados, apenas informe que os dados não estão disponíveis."""),
                HumanMessage(content=f"Dados fornecidos: {chunk}")
            ])
            messages = chat_template.format_messages()
            analysis_response = self.openai.invoke(messages)
            return analysis_response.content

        # Verifica se há múltiplos chunks
        if len(chunks) > 1:
            # Processar chunks em paralelo usando ThreadPoolExecutor
            analysis_results = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        analysis_results.append(result)
                        print(f"[DEBUG] Analysis result for chunk: {result}")
                    except Exception as e:
                        print(f"[ERROR] Error processing chunk: {e}")

            # Combinar os resultados das análises individuais
            combined_analysis = "\n".join(analysis_results)

            # Síntese Global: Usar o OpenAI para consolidar e responder à pergunta do usuário
            synthesis_template = ChatPromptTemplate.from_messages([
                SystemMessage(content="""Você é um analista que sintetiza dados e responde a perguntas do usuário com base em análises anteriores.
                ### Instruções:
                - Responda à pergunta do usuário diretamente.
                - Use os resultados analisados anteriormente como base para gerar uma resposta coesa e direta.
                - Evite repetir informações desnecessárias; foque no que é relevante para responder à pergunta.
                - Nunca invente dados, apenas informe que os dados não estão disponíveis.
                """),

                SystemMessage(content=f"Contexto anterior:\n{memory}"),

                HumanMessage(content=f"""A pergunta do usuário foi: "{user_query}".
                Abaixo estão os resultados das análises individuais:
                {combined_analysis}

                Com base nisso, forneça uma resposta única que atenda à pergunta do usuário.""")
            ])
            synthesis_messages = synthesis_template.format_messages()
            synthesis_response = self.openai.invoke(synthesis_messages)
            final_response = synthesis_response.content
        else:
            # Caso não haja chunks, processa diretamente o texto inteiro
            final_response = process_chunk(data_as_text)

        print(f"[DEBUG] Final analysis result: {final_response}")

        self.memory.chat_memory.add_ai_message(final_response)

    
        return final_response

    def process_query(self, query):
        print(f"[DEBUG] Processing query in {self.__class__.__name__}: {query}")
        state_data = {"query": query}
        
        # Executa cada estado manualmente em sequência
        self.generate_sql(state_data)
        self.execute_sql(state_data)
        self.analyze_data(state_data)
        
        return {"state": {"content": state_data.get("analysis_result", "Análise indisponível.")}}

    def invoke(self, user_query):
        state_data = {"query": user_query}
        self.generate_sql(state_data, user_query)
        self.execute_sql(state_data)
        final_response = self.analyze_data(state_data, user_query)
        return {"state": {"content": final_response}}


    def extract_sql_from_response(self, response_content):
        """
        Extrai e limpa a consulta SQL de um texto retornado pelo modelo.
        """
        # Remover espaços e quebras de linha extras
        response_content = response_content.replace("\n", " ").replace("\r", " ").strip()

        # Identificar o bloco de código SQL entre ```sql e ```
        start = response_content.find("```sql")
        end = response_content.find("```", start + 1)
        
        if start != -1 and end != -1:
            # Extraia apenas o conteúdo SQL e remova espaços desnecessários
            sql = response_content[start + 6:end].strip()
            # Substituir múltiplos espaços consecutivos por um único espaço
            sql = " ".join(sql.split())
            return sql
        
        # Retornar o conteúdo inteiro limpo caso os delimitadores não sejam encontrados
        return " ".join(response_content.split())
