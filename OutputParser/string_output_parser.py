from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'google/gemma-2-2b-it',
    task = 'conversational'
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template = 'Write a detailed report on {topic}',
    input_variable = ['topic']
)

template2 = PromptTemplate(
    template = 'Write a 5 line summary on the follwing text -- \n{text}',
    input_variable = ['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'Black Hole'})

print(result)
