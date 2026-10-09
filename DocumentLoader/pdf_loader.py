from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

loader = PyPDFLoader('DataSets\dl-curriculum.pdf')
docs = loader.load()

model_name = 'Qwen/Qwen3-0.6B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline(
    'text-generation',
    model = model,
    tokenizer = tokenizer
)
llm = HuggingFacePipeline(pipeline = pipe)

prompt = PromptTemplate(
    template = 'from the {text} is summarize for the {topic}',
    input_variables = ['text', 'topic']
)

parser = StrOutputParser()

chain = prompt | llm | parser
result = chain.invoke({'text':docs[0].page_content, 'topic': 'Backpropogation'})
print(result)