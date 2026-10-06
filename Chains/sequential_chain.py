from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template = 'write a detailed report on {topic}',
    input_variables =['topic']
)

prompt2 = PromptTemplate(
    template = 'generate a 5 points summary on {text}',
    input_varables = ['text']
)

llm = HuggingFaceEndpoint(
    repo_id = 'google/gemma-2-2b-it',
    task = 'conversational'
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'singularity'})

print(result)

chain.get_graph().print_ascii()