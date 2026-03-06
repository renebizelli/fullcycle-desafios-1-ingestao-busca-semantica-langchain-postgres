from search import search_prompt

yellow = "\033[93m"
azul = "\033[34m"
reset = "\033[0m"

def main():


    while True:

        chain = search_prompt()

        if not chain:
            print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
            return
    
        question = input(f"\n\n{reset}Pergunta: {yellow}").strip()

        result = chain.invoke(question)

        print(f"\n{reset}Responta: {azul}{result.strip()}{reset}")

        print(f"\n\n{reset}Deseja enviar a pergunta para o modelo? (s/n)")

        confirm = input().strip().lower()
        if confirm == 'n':
            break
        
    print(f"\n\n{reset}Encerrando o chat.")

if __name__ == "__main__":
    main()