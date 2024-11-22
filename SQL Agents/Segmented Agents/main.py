
from main_agent import MainAgent

class ChatInterface:
    def __init__(self):
        self.main_agent = MainAgent()
        self.selected_tab = None  # Aba inicial é indefinida
        self.tabs = ["google ads", "meta ads", "instagram posts", "crm", "geral"]  # Abas disponíveis

    def display_tabs(self):
        print("\nSelecione uma aba:")
        for idx, tab in enumerate(self.tabs, start=1):
            print(f"{idx}. {tab.capitalize()}")
        print("Digite o número da aba desejada.")

    def select_tab(self):
        while True:
            try:
                self.display_tabs()
                tab_choice = int(input("Número da aba: "))
                if 1 <= tab_choice <= len(self.tabs):
                    self.selected_tab = self.tabs[tab_choice - 1]
                    print(f"\n[Aba selecionada: {self.selected_tab.capitalize()}]")
                    break
                else:
                    print("Opção inválida. Escolha um número dentro da lista.")
            except ValueError:
                print("Entrada inválida. Digite um número correspondente à aba desejada.")

    def start_chat(self):
        print("Bem-vindo ao chat de análise de dados! Digite 'sair' para encerrar ou '0' para trocar de aba.")
        self.select_tab()  # Força o usuário a selecionar uma aba inicial
        while True:
            user_query = input("Você: ")
            if user_query.lower() == "sair":
                print("Encerrando o chat. Até mais!")
                break
            if user_query == "0":  # Opção para trocar de aba
                self.select_tab()
                continue
            # Passa a aba selecionada junto com a consulta para o agente principal
            response = self.main_agent.handle_query(user_query, selected_tab=self.selected_tab)
            print(f"Agente: {response}")

# Exemplo de uso
if __name__ == "__main__":
    chat = ChatInterface()
    chat.start_chat()
