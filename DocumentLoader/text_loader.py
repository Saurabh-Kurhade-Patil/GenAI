from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

loader = TextLoader('Datasets\BigBangTheory.txt',encoding='utf-8')
docs = loader.load()

model_name = 'Qwen/Qwen3-0.6B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline(
    'text-generation',
    model = model,
    tokenizer = tokenizer
)
llm = HuggingFacePipeline(pipeline=pipe)

prompt = PromptTemplate(
    template = 'Generate a summary {length} for the {text}',
    input_variables = ['length','text']
)
parser = StrOutputParser()

chain = prompt | llm | parser
result = chain.invoke({'length':'short', 'text':docs[0]})

print(result)