from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="conversational"
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template = 'Greet this person in 5 langauges, the name of the person is {name}',
    input_variable = ['name']
)
prompt = template.invoke({'name':'Muski'})
result = model.invoke(prompt)

print(result.content)