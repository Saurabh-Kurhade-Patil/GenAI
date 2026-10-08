from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnableParallel, RunnablePassthrough

'''                         pass_through
prompt -- llm -- parser [
                            lambda_word_count
'''

def CountWord(text):
    return len(text.split())

model_name = 'Qwen/Qwen3-0.6B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer
)

llm = HuggingFacePipeline(pipeline=pipe)

prompt = PromptTemplate(
    template='generate a joke on the {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

runnable_sequence = RunnableSequence(prompt, llm, parser)
runnable_parallel = RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':RunnableLambda(CountWord)
})
final_runnable = RunnableSequence(runnable_sequence, runnable_parallel)

print(final_runnable.invoke({'topic':'Black Hole'}))