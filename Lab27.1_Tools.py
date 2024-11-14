from langchain_experimental.utilities import PythonREPL
from langchain_community.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper

python_repl = PythonREPL()
result = python_repl.run('print(5*5)')
print(result)

ddg_search = DuckDuckGoSearchResults()
query = 'Quantos km tem a superfície do maior oceano da terra?'
search_result = ddg_search.run(query)
print(search_result)

wikipedia = WikipediaQueryRun(api_whapper=WikipediaAPIWrapper())
query = 'Quantos km tem a superfície do maior oceano da terra?'
wikipedia_results = wikipedia.run(query)
print(wikipedia_results)