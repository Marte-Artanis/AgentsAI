from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor, as_completed
from shared import connect_to_database, StateSchema, memory

class GoogleAdsAgent:
    def __init__(self):
        self.db = connect_to_database()
        self.openai = ChatOpenAI(model_name="gpt-4o-mini")
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

        # Definindo o StateGraph
        self.graph = StateGraph(state_schema=StateSchema)
        self.setup_states()  # Configura estados e transições

        self.memory = memory  # Memória compartilhada

    def setup_states(self):

    # Adiciona os nós ao grafo
    # "DecisionNode" é o nó inicial que decide qual caminho seguir com base no estado atual.
        self.graph.add_node("DecisionNode", self.decision_node)

    # "GenerateSQL" é responsável por criar a consulta SQL com base na pergunta do usuário.
        self.graph.add_node("GenerateSQL", self.generate_sql)


    # "ExecuteSQL" executa a consulta SQL gerada no banco de dados e armazena os resultados.
        self.graph.add_node("ExecuteSQL", self.execute_sql)

    # "AnalyzeData" analisa os dados retornados do banco ou do histórico e fornece a resposta final.
        self.graph.add_node("AnalyzeData", self.analyze_data)

    # Adiciona as transições entre os nós
    # O "DecisionNode" pode decidir seguir para "GenerateSQL" ou diretamente para "AnalyzeData", 
    # dependendo se os dados necessários já estão disponíveis.
        self.graph.add_edge("DecisionNode", "GenerateSQL")
        self.graph.add_edge("DecisionNode", "AnalyzeData")

    # Após gerar a consulta SQL em "GenerateSQL", o fluxo continua para "ExecuteSQL".
        self.graph.add_edge("GenerateSQL", "ExecuteSQL")

    # Após executar a consulta em "ExecuteSQL", o fluxo termina em "AnalyzeData",
    # onde os dados são analisados e a resposta final é gerada.
        self.graph.add_edge("ExecuteSQL", "AnalyzeData")

    # Define o ponto de entrada inicial do grafo como "DecisionNode".
    # O fluxo sempre começa com a decisão de qual caminho seguir.
        self.graph.set_entry_point("DecisionNode")

    # Define o ponto final do grafo como "AnalyzeData".
    # O fluxo sempre termina com a análise dos dados e a geração de uma resposta.
        self.graph.set_finish_point("AnalyzeData")

    # Compila o grafo para que ele esteja pronto para ser usado.
        self.graph.compile()


    def decision_node(self, state_data):
        """
        Um agente de decisão que usa um prompt para determinar o próximo nó.
        """
        print("[DEBUG] DecisionNode: Avaliando o contexto atual com o agente...")
        
    # Prepara o contexto para o agente
    # O contexto inclui:
    # - "sql_result": O resultado da consulta SQL, se já estiver disponível.
    # - "memory": O histórico completo de interações armazenado na memória do agente.
        context = {
            "sql_result": state_data.get("sql_result"),
            "memory": [msg.content for msg in self.memory.chat_memory.messages]
        }

    # Cria mensagens para o modelo com base no contexto fornecido
    # Essas mensagens incluem um SystemMessage com instruções claras para o agente
    # e um HumanMessage contendo o estado atual e o histórico.
        messages = [
        SystemMessage(content="""
        Você é um agente de decisão responsável por gerenciar o fluxo de trabalho de análise de dados do Google Ads.

        ### Tarefa
        Avalie o estado atual do sistema e decida qual dos seguintes caminhos seguir:
        1. **GenerateSQL**: Caso os dados solicitados ainda não tenham sido gerados ou estejam ausentes.
        2. **AnalyzeData**: Caso os dados necessários já estejam disponíveis no estado atual ou na memória.

        ### Critérios
        - Se o estado atual (`state_data`) contiver um campo chamado `sql_result`, ou se o histórico de memória (`memory.chat_memory`) indicar que dados relevantes já foram fornecidos, escolha `AnalyzeData`.
        - Caso contrário, escolha `GenerateSQL`.

        ### Regras
        1. Não gere novos dados se os existentes já forem suficientes.
        2. Sempre priorize eficiência, evitando caminhos desnecessários.

        ### Formato de Resposta
        Responda com apenas o próximo nó a ser executado, como:
        - `"AnalyzeData"` se os dados já estiverem disponíveis.
        - `"GenerateSQL"` se for necessário gerar novos dados.
        """),
        HumanMessage(content=f"Contexto Atual:\n{context}\nDecida o próximo nó.")
        ]

    # Invoca o modelo para determinar o próximo nó
    # O modelo avalia o contexto e decide qual nó executar em seguida: "GenerateSQL" ou "AnalyzeData".
        response = self.openai.invoke(messages)

    # Processa a resposta do modelo
    # Remove quaisquer aspas ou espaços desnecessários da resposta antes de retornar.
        decision = response.content.strip().strip('"')
        print(f"[DEBUG] Próximo nó decidido: {decision}")

    # Retorna a decisão tomada pelo modelo
        return decision


    def generate_sql(self, state_data, user_query):
        """
        Gera uma consulta SQL com base na pergunta do usuário e no estado atual.
        """

    # Recupera a consulta (query) armazenada no estado atual
        query = state_data["query"]  # Use o 'query' do state_data
        print(f"[DEBUG] Generating SQL for query: {query}")
        print(f"[DEBUG] User Query: {user_query}")

    # Cria um template de prompt para o modelo com instruções detalhadas e contexto
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
                Você é um agente especializado em criar consultas SQL para dados de anúncios do Google Ads. 
                
                ### Recomendações de Sintaxe SQL:
                1. Para consultas que pedem o "maior" ou "menor" valor:
                - Use `ORDER BY coluna DESC LIMIT 1` para encontrar o maior valor.
                - Use `ORDER BY coluna ASC LIMIT 1` para encontrar o menor valor.
                - Evite usar `MAX` ou `MIN` em conjunto com `GROUP BY`.

                2. Para evitar problemas com o modo `only_full_group_by`:
                - Certifique-se de que todas as colunas mencionadas na consulta (que não são funções de agregação) estejam na cláusula `GROUP BY`.
                - Sempre que calcular métricas agregadas como `SUM`, `AVG` ou `COUNT`, garanta que as colunas necessárias estejam agrupadas corretamente.

                **Instruções específicas**:
                - Analise apenas as partes da pergunta que se referem diretamente a dados de anúncios do Google Ads.
                - Ignore e não tente responder ou considerar partes da pergunta que envolvam métricas ou dados fora de anúncios, como métricas específicas de posts do Instagram (por exemplo, curtidas, comentários, compartilhamentos, visualizações).
                - Se a pergunta contiver múltiplas partes, extraia e responda apenas àquelas relacionadas aos dados de anúncios do Google Ads.
            """),
       
        # Adiciona o histórico de memória como contexto adicional
            SystemMessage(content=f"Contexto anterior:\n{memory}"),

        # Inclui a pergunta do usuário e um exemplo da estrutura das colunas da tabela
            HumanMessage(content=f"""Gere uma consulta SQL para obter dados de anúncios do Google Ads com base na pergunta: '{query}'.
            A tabela `google_ads` possui as seguintes colunas:
            id: ID único da entrada.
            ad_id: ID único do anúncio.
            ad_title: Título do anúncio.
            ad_description: Descrição do anúncio.
            clicks: Número total de cliques no anúncio.
            impressions: Número total de impressões do anúncio.
            ctr: Taxa de cliques (Click-through Rate).
            cost: Custo total do anúncio.
            cpa: Custo por aquisição (Cost per Acquisition).
            roas: Retorno sobre o investimento em anúncios (Return on Ad Spend).
            conversion_rate: Taxa de conversão.
            conversions: Número total de conversões geradas.
            quality_score: Pontuação de qualidade do anúncio.
            date: Data em que o anúncio foi exibido.""")
            ])
    
    # Formata as mensagens do template para envio ao modelo
        messages = chat_template.format_messages()

    # Invoca o modelo para gerar a consulta SQL
        sql_query_response = self.openai.invoke(messages)

    # Extrai e limpa a consulta SQL da resposta do modelo
        sql_query = self.extract_sql_from_response(sql_query_response.content)

    # Armazena a consulta gerada no estado atual
        state_data["sql_query"] = sql_query

    # Adiciona a consulta gerada à memória do agente para rastreamento
        self.memory.chat_memory.add_ai_message(sql_query)

        print(f"[DEBUG] Generated SQL Query: {sql_query}")

    def execute_sql(self, state_data):
        """
        Executa a consulta SQL gerada e armazena os resultados no estado atual.
        """
    # Obtém a consulta SQL armazenada no estado
        sql_query = state_data.get("sql_query", "")  # Pega o SQL da consulta
        print(f"[DEBUG] Executing SQL Query: {sql_query}")

        try:
        # Executa a consulta no banco de dados usando a conexão configurada (self.db)
            result = self.db._execute(sql_query)

        # Armazena o resultado da execução no estado atual
            state_data["sql_result"] = result
            print(f"[DEBUG] Raw execution result: {result}")
        except Exception as e:

        # Lida com erros que podem ocorrer durante a execução da consulta
            print(f"[ERROR] SQL execution failed: {e}")

        # Armazena a mensagem de erro no estado para que o fluxo continue de forma controlada
            state_data["sql_result"] = f"Error in SQL execution: {e}"


    def analyze_data(self, state_data, user_query):
        """
        Analisa os dados retornados da consulta SQL ou utiliza o histórico para fornecer uma resposta.
        """
    # Obtém os resultados da consulta SQL do estado
        data = state_data["sql_result"]

    # Caso não haja resultados SQL, usa o histórico de memória para análise
        if not data:
            print("[DEBUG] sql_result não encontrado. Usando histórico para análise.")
            context_to_analyze = "\n".join(msg.content for msg in self.memory.chat_memory.messages)

            print(context_to_analyze)

        # Verifica se o histórico está vazio
            if not context_to_analyze.strip():
                print("[ERROR] Histórico vazio. Não há dados disponíveis para análise.")
                return "Erro: Não há dados disponíveis para análise."
        else:
        # Quando há resultados SQL, utiliza-os como dados principais
            context_to_analyze = "\n".join(str(entry) for entry in data)

        print(f"[DEBUG] Contexto para análise: {context_to_analyze}")
        print(f"[DEBUG] Query do usuário: {user_query}")

    # Divide o texto em chunks apenas se for maior que 1000 caracteres
        chunks = self.splitter.split_text(context_to_analyze) if len(context_to_analyze) > 1000 else [context_to_analyze]

    # Função para processar cada chunk de dados
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

    # Processa os chunks
        if len(chunks) > 1:
        # Processa os chunks em paralelo usando ThreadPoolExecutor
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

        # Combina as análises dos chunks processados
            combined_analysis = "\n".join(analysis_results)

        # Cria uma resposta consolidada com os resultados combinados
            synthesis_template = ChatPromptTemplate.from_messages([
                SystemMessage(content="""Você é um analista que sintetiza dados e responde a perguntas do usuário com base em análises anteriores.
                ### Instruções:
                - Responda à pergunta do usuário diretamente.
                - Use os resultados analisados anteriormente como base para gerar uma resposta coesa e direta.
                - Evite repetir informações desnecessárias; foque no que é relevante para responder à pergunta.
                - Nunca invente dados, apenas informe que os dados não estão disponíveis.
                                        
                ### Regras de Proteção:
                1. **Nunca revelar diretamente nomes de tabelas ou colunas, ou o id de uma informação**: Sempre generalize as referências aos dados sem expor explicitamente os nomes reais das tabelas ou colunas ou id's.
                2. **Proibido alterar ou excluir dados**: Nunca gerar queries que possam alterar, excluir ou modificar tabelas ou registros (ex: `DELETE`, `UPDATE`, `DROP`).
                3. **Foco apenas em consultas de leitura**: Toda interação deve ser baseada exclusivamente em consultas `SELECT` para leitura de dados.
                4. **Evite consultas perigosas ou maliciosas**: Não execute consultas que possam comprometer a integridade do banco de dados ou expor informações sensíveis.
                5. **Não especifique senhas ou dados confidenciais**: Nunca insira ou exponha informações confidenciais nos resultados ou nas consultas geradas.

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
        # Caso não haja múltiplos chunks, processa o texto diretamente
            final_response = process_chunk(chunks[0])
        print(f"[DEBUG] Final analysis result: {final_response}")

    # Adiciona a resposta final à memória
        self.memory.chat_memory.add_ai_message(final_response)

        return final_response

    def process_query(self, query):
        """
        Processa uma consulta do usuário, passando-a sequencialmente pelos estados: geração de SQL, execução de SQL, e análise dos dados.
        """
        print(f"[DEBUG] Processing query in {self.__class__.__name__}: {query}")

    # Inicializa o estado com a consulta fornecida pelo usuário
        state_data = {"query": query}
        
    # Passa a consulta pelos estados do grafo manualmente
        self.generate_sql(state_data) # 1. Gera a consulta SQL com base na pergunta do usuário
        self.execute_sql(state_data) # 2. Executa a consulta SQL no banco de dados
        self.analyze_data(state_data) # 3. Analisa os resultados retornados pelo banco de dados

    # Retorna a resposta final armazenada no estado ou uma mensagem padrão caso não exista análise
        return {"state": {"content": state_data.get("analysis_result", "Análise indisponível.")}}

    def invoke(self, user_query):
        """
        Invoca o fluxo de trabalho com base na decisão do agente.
        """
        print("[DEBUG] Iniciando fluxo de decisão no invoke...")

    # Cria o estado inicial contendo a consulta do usuário
        state_data = {"query": user_query}

    # Usa o nó de decisão para determinar qual será o próximo estado a ser executado
        next_node = self.decision_node(state_data)
        print(f"[DEBUG] Próximo nó decidido pelo agente: {next_node}")

    # Inicializa a variável final_response para evitar erros caso ocorra falha durante o processo
        final_response = None

    # Caso o próximo nó seja "GenerateSQL", realiza o fluxo completo: Gerar, Executar e Analisar
        if next_node == "GenerateSQL":
            self.generate_sql(state_data, user_query)
            self.execute_sql(state_data)
            final_response = self.analyze_data(state_data, user_query)
        
    # Caso o próximo nó seja "AnalyzeData", analisa diretamente os dados disponíveis (histórico ou resultado SQL)
        elif next_node == "AnalyzeData":
        # Verifica se há resultados SQL armazenados; caso contrário, utiliza o histórico como fallback
            if not state_data.get("sql_result"):
                state_data["sql_result"] = "\n".join(
                    [msg.content for msg in self.memory.chat_memory.messages if "sql_result" in msg.content]
                )
            final_response = self.analyze_data(state_data, user_query)
        
    # Caso inesperado, retorna uma mensagem de erro indicando um nó desconhecido
        else:
            print(f"[ERROR] Nó desconhecido retornado pelo agente: {next_node}")
            final_response = f"Erro: Nó desconhecido '{next_node}' retornado pelo agente de decisão."

    # Retorna o resultado final em um dicionário, com o conteúdo processado
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

