from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

openai = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=1,
    frequency_penalty=2,
    presence_penalty=2,
    max_tokens=500
)

# Defina a mensagem
messages = [
    {'role': 'system', 'content': 'You are a delusion and crazy person that says random stuff not always related to the question, demand, or affirmation.'},
    {'role': 'user', 'content': 'Tell something you like'}
]

# Chame o invoke com apenas o parâmetro messages
response = openai.invoke(messages)

# Exibir a resposta
print(response.content)


# Use completions se você precisar de uma resposta autônoma e isolada, ou se estiver usando um modelo que não seja um modelo de chat.
# Use chat se estiver desenvolvendo uma aplicação que simule um diálogo, como um chatbot, ou que precise de respostas que considerem o contexto de mensagens anteriores.