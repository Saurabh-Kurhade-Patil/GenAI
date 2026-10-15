from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('Datasets\dl-curriculum.pdf')
docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 200,
    separator='',
    chunk_overlap =0
)
result = splitter.split_documents(docs)
print(result[1].page_content)