from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN was not found in .env")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-0.6B",
    task="Conversational",
    #provider="hf-inference",
    #huggingfacehub_api_token=HF_TOKEN,
    #max_new_tokens=100,
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is outside the simulation?")

print(result.content)
