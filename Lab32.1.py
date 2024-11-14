from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.agents import Tool, AgentExecutor, initialize_agent, create_react_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from dotenv import load_dotenv
import os

# Carregar a chave de API do OpenAI
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Configuração do cliente OpenAI
openai = ChatOpenAI(
    model_name='gpt-3.5-turbo', 
    frequency_penalty=2, 
    presence_penalty=2,
    temperature=1,        
    max_tokens=1000,
    n=1
)

# Template do prompt
prompt = """
    Você é Albus Dumbledore, o sábio e respeitado diretor de Hogwarts. Sua fala é sempre calma, ponderada e cheia de sabedoria. 
    Você aborda todas as situações com empatia e uma perspectiva filosófica profunda, sempre buscando o melhor para os outros, mas também 
    respeitando as dificuldades e complexidades da vida. 
    Quando fala, suas palavras são tranquilizadoras, mas firmes, e você sempre tenta orientar as pessoas a tomarem as melhores decisões possíveis.
    Responda às perguntas de maneira ponderada e com compaixão.
    Você responde as perguntas com a ajuda da internet.
    Perguntas: {q}
    """
prompt_template = PromptTemplate.from_template(prompt)

# Instruções do agente
react_instructions = hub.pull('hwchase17/react')
print(react_instructions)

# Ferramenta Python REPL
python_repl = PythonREPLTool()
python_repl_tool = Tool(
    name='Python REPL',
    func=python_repl.run,
    description='Qualquer tipo de cálculo deve lograr desta ferramenta. Você não deve realizar o cálculo diretamente. Você deve inserir o código Python'
)

# Ferramenta DuckDuckGoSearchResults do LangChain
duckduckgo_tool = Tool(
    name='DuckDuckGo Search',
    func=DuckDuckGoSearchResults().run,  # Utilizando a ferramenta integrada do LangChain
    description='Esta ferramenta utiliza o DuckDuckGo para buscar informações na web.'
)

# Configuração das ferramentas e do agente
tools = [python_repl_tool, duckduckgo_tool]
agent = create_react_agent(openai, tools, react_instructions)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10
)

# Pergunta de exemplo
question = 'Qual o impacto da mineração de lítio no mundo e, considerando os avanços da tecnologia, quanto de lítio seria necessário para alimentar as necessidades energéticas de um futuro mais sustentável?'

output = agent_executor.invoke({
    'input': prompt_template.format(q=question)
})

print(output['input'])
print(output['output'])


# 1. **Início com `agent_executor.invoke`**: 
#    - A pergunta do usuário é processada pelo `AgentExecutor`.

# 2. **Agente `react`**: 
#    - Recebe a pergunta e decide qual ferramenta usar com base na análise da entrada e nas instruções de `react`.

# 3. **Ferramentas**:
#    - Se a pergunta requer cálculos, `react` chama `PythonREPLTool`.
#    - Se a pergunta precisa de informações externas, `react` chama `DuckDuckGoSearchResults`.

# 4. **Execução das Ferramentas**:
#    - A ferramenta chamada (`PythonREPLTool` ou `DuckDuckGoSearchResults`) executa a tarefa e retorna o resultado ao agente `react`.

# 5. **Iteração**:
#    - `react` avalia o resultado e decide se deve continuar usando mais ferramentas ou se já possui uma resposta final.
#    - Esse processo itera, no máximo, 10 vezes.

# 6. **Resposta Final**:
#    - `AgentExecutor` reúne a resposta final de `react` e a retorna como o output, que é então impresso. 

# Este é o caminho percorrido no momento do processamento para gerar a resposta final.