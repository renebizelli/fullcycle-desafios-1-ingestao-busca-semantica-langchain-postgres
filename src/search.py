import os
from dotenv import load_dotenv
from langchain_openai  import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector    
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import chain
from langchain_core.output_parsers import StrOutputParser

load_dotenv(dotenv_path= "./.env.example")

def validate_env():
    print('Validando variáveis de ambiente...')
    for key in ("PDF_PATH", "CHUNK_SIZE", "CHUNK_OVERLAP", "SIMILARITY_K", "OPENAI_MODEL", "OPENAI_EMBEDDINGS_MODEL", "PGVECTOR_HOST", "PGVECTOR_PORT", "PGVECTOR_USER", "PGVECTOR_PASSWORD", "PGVECTOR_DB"):
        if not os.getenv(key):
            raise RuntimeError(f"Variável de ambiente ausente: {key}")


PROMPT_TEMPLATE = """
CONTEXTO:
{context}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{question}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

validate_env()

embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"))

store  = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PGVECTOR_DB"),
    connection=os.getenv("PGVECTOR_HOST"),
    use_jsonb=True,
)

model = ChatOpenAI(model=os.getenv("OPENAI_MODEL"), temperature=os.getenv("OPENAI_TEMPERATURE", 0.0))

@chain
def _search(question: str):
    results = store.similarity_search_with_score(query=question, k=int(os.getenv("SIMILARITY_K", 10)))

    r = [
        doc.page_content.strip()
        for doc, score in results
    ]

    return { "context": "\n".join(r), "question": question }

prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

def search_prompt():

  chain = _search | prompt | model | StrOutputParser()

  return chain
      



