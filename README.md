# Desafio MBA Engenharia de Software com IA - Full Cycle

# Ambiente

- Todos os parâmetros foram inseridos no arquivo de ambiente .env.example
- O caminho completo com o nome do arquivo PDF deve ser inserido na chave PDF_PATH do arquivo .env.example
- Deve ser informado a KEY da OpenAI na chave OPENAI_API_KEY, substituindo o valor XXXX

# Subir Banco de dados.

- rodar > docker compose up -d

# Ingestão de dados

- rodar > python src/ingest.py

# Rodar o Chat

- rodar > python src/chat.py
