from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma


def load_documents(DATA_PATH):
    doc_loader = PyPDFDirectoryLoader(DATA_PATH, glob='*.pdf')
    return doc_loader.load()


def split_documents(documents: list):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        length_function = len,
        is_separator_regex = False
    )
    return text_splitter.split_documents(documents)


def get_embedding_function():
    embeddings = BedrockEmbeddings(
        credentials_profile_name = 'default',
        region_name = 'us-east-1'
    )
    return embeddings


def main():
    DATA_PATH = 'data/'
    docs = load_documents(DATA_PATH)
    chunks = split_documents(docs)
    print(chunks[0])

if __name__=='__main__':
    main()