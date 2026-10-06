from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class Person(BaseModel):

    name : str = Field(description='Name of the person')
    age : int = Field(gt=18, description='age of the person')
    city : str = Field(description='Name of the city person belongs to')

llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='conversational'
)

model = ChatHuggingFace(llm= llm)

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = 'Generate the name, age and city of the fictional person from {place} \n {format_instructions}',
    input_variables=['place'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'place':'Japan'})

print(result)