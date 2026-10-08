from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

'''
prompt -- model -- parser
'''
model_name = 'Qwen/Qwen3-0.6B'

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer
)

llm = HuggingFacePipeline(
   pipeline=pipe
)

prompt = PromptTemplate(
    template='generate a joke for the {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

sequence_chain = RunnableSequence(prompt, llm, parser)

result = sequence_chain.invoke({
    'topic':'Black hole'
})

print(result)