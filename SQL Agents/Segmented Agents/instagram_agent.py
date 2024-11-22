from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor, as_completed
from shared import connect_to_database, StateSchema, memory

class InstagramPostsAgent:
    def __init__(self):
        self.db = connect_to_database()
        self.openai = ChatOpenAI(model_name="gpt-4o-mini")
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        # StateGraph
        self.graph = StateGraph(state_schema=StateSchema)
        self.setup_states()

        self.memory = memory  # Memória compartilhada


    def setup_states(self):
        # Adiciona os nós ao grafo
        self.graph.add_node("DecisionNode", self.decision_node)
        self.graph.add_node("GenerateSQL", self.generate_sql)
        self.graph.add_node("ExecuteSQL", self.execute_sql)
        self.graph.add_node("AnalyzeData", self.analyze_data)

        # Adiciona as transições
        self.graph.add_edge("DecisionNode", "GenerateSQL")
        self.graph.add_edge("DecisionNode", "AnalyzeData")
        self.graph.add_edge("GenerateSQL", "ExecuteSQL")
        self.graph.add_edge("ExecuteSQL", "AnalyzeData")

        # Define o ponto de entrada inicial
        self.graph.set_entry_point("DecisionNode")

        # Define o ponto de saída
        self.graph.set_finish_point("AnalyzeData")

        # Compila o grafo
        self.graph.compile()
    

    def decision_node(self, state_data):
        """
        Um agente de decisão que usa um prompt para determinar o próximo nó.
        """
        print("[DEBUG] DecisionNode: Avaliando o contexto atual com o agente...")
        
        # Prepara o contexto para o agente
        context = {
            "sql_result": state_data.get("sql_result"),
            "memory": [msg.content for msg in self.memory.chat_memory.messages]
        }

        # Formata o prompt com o contexto

        # Chama o modelo para decidir
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

        response = self.openai.invoke(messages)

        # Retorna a decisão do modelo (o próximo nó)
        decision = response.content.strip().strip('"')
        print(f"[DEBUG] Próximo nó decidido: {decision}")
        return decision

    def generate_sql(self, state_data, user_query):
        query = state_data["query"]  # Use o 'query' do state_data
        print(f"[DEBUG] Generating SQL for query: {query}")
        print(f"[DEBUG] User Query: {user_query}")
        
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
            Você é um agente especializado em gerar consultas SQL para dados de posts e seguidores do Instagram. 

            ### Recomendações de Sintaxe SQL:
            1. Para consultas que pedem o "maior" ou "menor" valor:
            - Use `ORDER BY coluna DESC LIMIT 1` para encontrar o maior valor.
            - Use `ORDER BY coluna ASC LIMIT 1` para encontrar o menor valor.
            - Evite usar `MAX` ou `MIN` em conjunto com `GROUP BY`.
            - Certifique-se de que todas as colunas mencionadas na consulta (que não são funções de agregação) estejam na cláusula `GROUP BY`.
            - Ao calcular métricas como somas ou médias, agrupe corretamente os dados com `GROUP BY` quando necessário.

            **Instruções específicas**:
            - Analise apenas as partes da pergunta que se referem diretamente a posts ou dados de seguidores do Instagram.
            - Ignore e não tente responder ou considerar partes da pergunta que envolvam métricas ou dados fora do Instagram (por exemplo, métricas como CTR, CPC ou conversões que pertencem a anúncios).
            - Se a pergunta contiver múltiplas partes, extraia e responda apenas àquelas relacionadas aos dados de Instagram.
            """),
           
            SystemMessage(content=f"Contexto anterior:\n{memory}"),

            HumanMessage(content=f"""Gere uma consulta SQL para obter dados das tabelas `instagram_posts` e `follower_data` com base na pergunta: '{user_query}'.
            
            A tabela `instagram_posts` possui as seguintes colunas:
            - `post_id` (int): ID único identificador de cada post.
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
        sql_query = self.extract_sql_from_response(sql_query_response.content)
        state_data["sql_query"] = sql_query

        self.memory.chat_memory.add_ai_message(sql_query)

        print(f"[DEBUG] Generated SQL Query: {sql_query}")

    def execute_sql(self, state_data):
        sql_query = state_data.get("sql_query", "")
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


        # Caso não haja sql_result, usa o histórico diretamente no prompt
        if not data:
            print("[DEBUG] sql_result não encontrado. Usando histórico para análise.")
            context_to_analyze = "\n".join(msg.content for msg in self.memory.chat_memory.messages)

            print(context_to_analyze)

            if not context_to_analyze.strip():
                print("[ERROR] Histórico vazio. Não há dados disponíveis para análise.")
                return "Erro: Não há dados disponíveis para análise."
        else:
            # Quando há sql_result, ele é usado como dados principais
            context_to_analyze = "\n".join(str(entry) for entry in data)

        print(f"[DEBUG] Contexto para análise: {context_to_analyze}")
        print(f"[DEBUG] Query do usuário: {user_query}")

        # Dividindo em chunks apenas se necessário
        chunks = self.splitter.split_text(context_to_analyze) if len(context_to_analyze) > 1000 else [context_to_analyze]


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
            # Caso não haja chunks, processa diretamente o texto inteiro
            final_response = process_chunk(chunks[0])
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
        """
        Invoca o fluxo de trabalho com base na decisão do agente.
        """
        print("[DEBUG] Iniciando fluxo de decisão no invoke...")

        # Cria o estado inicial com a consulta do usuário
        state_data = {"query": user_query}

        # Decide o nó inicial
        next_node = self.decision_node(state_data)
        print(f"[DEBUG] Próximo nó decidido pelo agente: {next_node}")

        # Inicializa a variável final_response para evitar erros
        final_response = None

        # Fluxo completo (Gerar, Executar, Analisar)
        if next_node == "GenerateSQL":
            self.generate_sql(state_data, user_query)
            self.execute_sql(state_data)
            final_response = self.analyze_data(state_data, user_query)
        
        # Apenas análise
        elif next_node == "AnalyzeData":
            # Verifica se o histórico contém os dados necessários
            if not state_data.get("sql_result"):
                state_data["sql_result"] = "\n".join(
                    [msg.content for msg in self.memory.chat_memory.messages if "sql_result" in msg.content]
                )
            final_response = self.analyze_data(state_data, user_query)
        
        # Caso inesperado
        else:
            print(f"[ERROR] Nó desconhecido retornado pelo agente: {next_node}")
            final_response = f"Erro: Nó desconhecido '{next_node}' retornado pelo agente de decisão."

        # Retorna o resultado final
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

