from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'Qwen/Qwen3-0.6B',
    task = 'conversational'
)

class Feedback(BaseModel):

    sentiment : Literal['positive','negative'] = Field(description='Give me the sentiment of the feeback')

parser = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template = 'Classify the sentiment of the review as Positive or Negative review {feedback} \n {format_instruction}',
    input_variables = ['feedback'],
    partial_variables = {'format_instruction':parser.get_format_instructions()}
)

model1 = ChatHuggingFace(llm=llm)

classifer_chain = prompt1 | model1 | parser
print(classifer_chain.invoke({'feedback':'This is a bad cellphone'}))

prompt2 = PromptTemplate(
    template = 'Give me a proper response for a positive feedback {feedback}',
    input_variables = ['feedback']
)

prompt3 = PromptTemplate(
    template = 'Give me a proper response for a negative feedback {feedback}',
    input_variables = ['feedback']
)

model2= ChatHuggingFace(llm=llm)

parser1 = StrOutputParser()

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model2 | parser1),
    (lambda x:x.sentiment == 'negative', prompt3 | model2 | parser1),
    RunnableLambda(lambda x: "could not find the sentiment")
)

chain = classifer_chain | branch_chain
print(chain.invoke({'feedback':'This is a bad cellphone'}))
chain.get_graph().print_ascii()