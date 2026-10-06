from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
'''
structured output parser is extended version of JSON output parser

Pros: 
    - can enforce schema

Cons:
    - data validation is not possible
'''
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'google/gemma-2-2b-it',
    task = 'conversational'
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact_1',description='fact 1 about the topic'),
    ResponseSchema(name='fact_2',description='fact 2 about the topic'),
    ResponseSchema(name='fact_3',description='fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = 'Give me the 3 facts about the topic {topic} \n {format_instructions}',
    input_variables = ['topic'],
    partial_variables = {'format_instructions': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic':'Black hole'})

print(result)