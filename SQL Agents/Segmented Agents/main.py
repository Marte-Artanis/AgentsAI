from main_agent import MainAgent

class ChatInterface:
    def __init__(self):
        self.main_agent = MainAgent()

    def start_chat(self):
        print("Bem-vindo ao chat de análise de dados! Digite 'sair' para encerrar.")
        while True:
            user_query = input("Você: ")
            if user_query.lower() == "sair":
                print("Encerrando o chat. Até mais!")
                break
            response = self.main_agent.handle_query(user_query)
            print(f"Agente: {response}")

# Exemplo de uso
if __name__ == "__main__":
    chat = ChatInterface()
    chat.start_chat()
