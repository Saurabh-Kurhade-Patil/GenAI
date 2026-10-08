from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda, RunnableParallel, RunnableSequence, RunnablePassthrough

model_name = 'Qwen/Qwen3-0.6B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline(
    'text-generation',
    model = model,
    tokenizer = tokenizer
)
llm = HuggingFacePipeline(pipeline=pipe)

prompt1 = PromptTemplate(
    template = 'Generate a detailed report for {topic}',
    input_variables = ['topic']
)
prompt2 = PromptTemplate(
    template = 'from this {text} generate a short summary',
    input_variables = ['text']
)
parser = StrOutputParser()

runnable_sequence = RunnableSequence(prompt1, llm, parser)
runnable_branch = RunnableBranch(
    (lambda x:len(x.split()) > 300 , RunnableSequence(prompt2, llm, parser)),
    RunnablePassthrough()
)
final_runnable = RunnableSequence(runnable_sequence, runnable_branch)
print(final_runnable.invoke({'topic':'sufi music vs k-pop music'}))