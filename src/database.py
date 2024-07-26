from embedding_function import get_embedding_function
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma


def add_to_chroma(chunks: list):
    # Load database
    db = Chroma(
        persist_directory = CHROMA_PATH,
        embedding_function = get_embedding_function()
    )
    db.add_documents(new_chunks, ids=new_chunk_ids)
    db.persist()


def get_chunk_ids(chunks):
    last_page_id = None
    chunk_index = 0

    for chunk in chunks:
        src = chunk.metadata.get('source')
        page = chunk.metadata.get('page')
        curr_page_id = f'{src}:{page}'

        # If current page id is same as last page id, increment chunk index
        if curr_page_id == last_page_id:
            chunk_index += 1
        # Else reset chunk index and set last page id to current page id
        else:
            chunk_index = 0
            last_page_id = curr_page_id

        chunk.metadata['id'] = f'{curr_page_id}:{chunk_index}'
        

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


def main():
    DATA_PATH = 'data/'
    docs = load_documents(DATA_PATH)
    chunks = split_documents(docs)
    print(chunks[0])

if __name__=='__main__':
    main()