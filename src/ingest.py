import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_openai  import OpenAIEmbeddings
from langchain_postgres import PGVector    
from langchain_core.documents import Document    

load_dotenv(dotenv_path= "./.env.example")

def validate_env():
    print('Validando variáveis de ambiente...')
    for key in ("PDF_PATH", "CHUNK_SIZE", "CHUNK_OVERLAP", "OPENAI_MODEL", "OPENAI_EMBEDDINGS_MODEL", "PGVECTOR_HOST", "PGVECTOR_PORT", "PGVECTOR_USER", "PGVECTOR_PASSWORD", "PGVECTOR_DB"):
        if not os.getenv(key):
            raise RuntimeError(f"Variável de ambiente ausente: {key}")


def docs_loader() -> list[Document]:
    print('Carregando PDF...')
    loader = PyPDFLoader(os.getenv("PDF_PATH")) 
    return loader.load()


def get_chunks(docs: list[Document]) -> list[Document]:
    print('Dividindo documentos em chunks...')
    return RecursiveCharacterTextSplitter(
                    chunk_size= int(os.getenv("CHUNK_SIZE", 1000)),
                    chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 150)),
                    add_start_index=False).split_documents(docs)


def enrich_metadata(chunks: list[Document]) -> list[Document]:
    print('Enriquecendo metadados...')
    return [
        Document(
            page_content=chunk.page_content, 
            metadata={k:v for k, v in chunk.metadata.items() if v not in ("", None)}
        ) for  chunk in chunks
    ]


def add_documents(embeddings: OpenAIEmbeddings, enriched: list[Document]):

    print('Adicionando documentos ao armazenamento...')

    ids = [f"doc-{i}" for i in range(len(enriched))]

    store  = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PGVECTOR_DB"),
        connection=os.getenv("PGVECTOR_HOST"),
        use_jsonb=True,
    )

    store.add_documents(documents=enriched, ids=ids)   


def ingest_pdf():

    validate_env()

    docs = docs_loader()

    chunks = get_chunks(docs)

    if not chunks:
        raise SystemExit(0)    

    enriched = enrich_metadata(chunks)

    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"))

    add_documents(embeddings, enriched)     

    print('Ingestão concluída com sucesso!')

if __name__ == "__main__":
    ingest_pdf()