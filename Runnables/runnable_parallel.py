from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain.schema.runnable import RunnableParallel, RunnableSequence

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    'text-generation',
    model = model,
    tokenizer=tokenizer
)

llm = HuggingFacePipeline(pipeline=pipe)

prompt1 = PromptTemplate(
    template = 'Generate a description of Allosaurus in {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate a description of T-rex {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'Allosaurus': RunnableSequence(prompt1, llm, parser),
    'T-rex' : RunnableSequence(prompt2, llm, parser)
})

result = parallel_chain.invoke({'topic':'dinosaur'})

print(result['Allosaurus'])
print(result['T-rex'])