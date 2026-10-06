from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template = 'Generate a summary for the text \n {text}',
    input_variables = ['text']
)

prompt2 = PromptTemplate(
    template = 'generate a question and answer for the text \n {text}',
    input_variables = ['text']
)

prompt3 = PromptTemplate(
    template = 'merge the provided summary and quiz into single document \n notes->{chain1} , quiz->{chain2}',
    input_variables = ['chain1', 'chain2']
)

llm1 = HuggingFaceEndpoint(
    repo_id= 'google/gemma-2-2b-it',
    task = 'conversational'
)

llm2 = HuggingFaceEndpoint(
    repo_id = 'Qwen/Qwen3-0.6B',
    task = 'conversational'
)

model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'chain1': prompt1 | model1 | parser,
    'chain2': prompt2 | model2 | parser
})

merge_chain = prompt3 | model2 | parser

chain = parallel_chain | merge_chain
text = """
Earth is the third planet from the Sun and the only astronomical object known to harbor life. This is made possible by Earth being an ocean world, the only one in the Solar System sustaining liquid surface water. Almost all of Earth's water is contained in its ocean, which covers 70.8% of Earth's crust. The remaining 29.2% of Earth's crust is land, which is predominantly located within Earth's land hemisphere in the form of continental landmasses. Most of Earth's land is at least somewhat humid and covered by vegetation, while large ice sheets at Earth's polar deserts retain more water than Earth's groundwater, lakes, rivers, and atmospheric water combined. Earth's crust consists of slowly moving tectonic plates, which interact to produce mountain ranges, volcanoes, and earthquakes. Earth has a liquid outer core that generates a magnetosphere capable of deflecting most of the destructive solar winds and cosmic radiation.
"""
result = chain.invoke({'text':text})
print(result)
chain.get_graph().print_ascii()