from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableParallel

'''

prompt -- model -- parser [ 
                            
'''

model_name='Qwen/Qwen3-0.6B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    'text-generation',
    tokenizer=tokenizer,
    model=model
)

llm = HuggingFacePipeline(pipeline=pipe)

promt = PromptTemplate(
    template = 'generate a short joke on {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()
prompt1 = PromptTemplate(
    template = 'explain the joke in detail {text}',
    input_variables=['text']
)

sequetial_runnable = RunnableSequence(promt, llm, parser)
parallel_runnable = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'summary' : RunnableSequence(prompt1, llm, parser)
})
result = RunnableSequence(sequetial_runnable,parallel_runnable)
print(result.invoke({'topic':'black hole'}))