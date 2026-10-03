# GenAI
Hands-on LangChain projects and experiments for Generative AI and LLM applications.

LangChain Generative AI

A hands-on repository for learning and building Generative AI applications using LangChain, Large Language Models (LLMs), Hugging Face, prompt engineering, and Streamlit.

This repository contains practical examples, experiments, and small projects covering the fundamentals of working with LLMs and progressively moving toward building complete Generative AI applications.

🚀 About the Repository

The goal of this repository is to understand how modern Generative AI applications are built using LangChain and related tools.

The examples focus on:

Working with Large Language Models

Hugging Face models and inference

LangChain fundamentals

Prompt engineering

Prompt templates

Chat models

Chains

Streamlit-based GenAI applications

Structured interaction with LLMs

Building reusable AI components

Experimenting with different LLM configurations

The repository is primarily intended for learning, experimentation, and practical implementation.

🧠 Technologies & Tools

The repository uses technologies from the modern Generative AI ecosystem.

Core Technologies

Python

LangChain

Hugging Face

Large Language Models (LLMs)

Prompt Engineering

Streamlit

python-dotenv

LangChain Components

Examples may include:

PromptTemplate

ChatPromptTemplate

HuggingFaceEndpoint

ChatHuggingFace

Chains

Prompt loading and serialization

LLM invocation

Chat models

📁 Repository Structure

The repository is organized into different sections as the learning progresses.

LangChain-GenAI/
│
├── Prompts/
│   ├── prompt_ui.py
│   ├── prompt_generator.py
│   ├── template.json
│   └── ...
│
├── requirements.txt
├── .env
├── .gitignore
└── README.md


The folder structure may evolve as new concepts and projects are added.

🔑 Environment Setup
1. Clone the Repository
git clone <YOUR_GITHUB_REPOSITORY_URL>


Navigate into the repository:

cd LangChain-GenAI

2. Create a Virtual Environment

Creating a virtual environment is recommended to keep project dependencies isolated.

python -m venv venv


Activate the environment on Windows:

venv\Scripts\activate


For macOS/Linux:

source venv/bin/activate

3. Install Dependencies

Install the required Python packages:

pip install -r requirements.txt


Some of the primary packages used in this repository include:

langchain
langchain-core
langchain-huggingface
huggingface-hub
python-dotenv
streamlit

🔐 Environment Variables

Some examples use Hugging Face models through the Hugging Face Inference API.

Create a .env file in the root directory:

HF_TOKEN=your_huggingface_token


Replace:

your_huggingface_token


with your actual Hugging Face access token.

⚠️ Important

Never commit your .env file or API keys to GitHub.

Add the following to .gitignore:

.env
venv/
__pycache__/
*.pyc

🤗 Hugging Face Integration

This repository uses Hugging Face models with LangChain.

A typical setup looks like:

from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="conversational"
)


For chat-based interaction, ChatHuggingFace can be used:

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="conversational"
)

model = ChatHuggingFace(llm=llm)

response = model.invoke(
    "What is the capital of Japan?"
)

print(response.content)

📝 Prompt Engineering

Prompt engineering is an important part of building Generative AI applications.

This repository explores how prompts can be created, structured, reused, and passed to LLMs.

For example:

from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""
    Please provide details about the car "{car_input}".

    Explanation style: {car_details}
    Explanation length: {length_input}

    If you don't have sufficient information, respond with
    "Insufficient Data available" instead of guessing.
    """,
    input_variables=[
        "car_input",
        "car_details",
        "length_input"
    ],
    validate_template=True
)


The template can then be combined with a model:

chain = template | model


And invoked with dynamic values:

result = chain.invoke({
    "car_input": "Porsche 911",
    "car_details": "Technical",
    "length_input": "Medium"
})

🔗 LangChain Chains

One of the important concepts explored in this repository is the LangChain expression:

chain = prompt | model


This allows multiple components to be connected together.

For example:

User Input
    ↓
Prompt Template
    ↓
LLM
    ↓
Generated Response


A simple implementation:

prompt = PromptTemplate(
    template="Explain {topic} in simple terms.",
    input_variables=["topic"]
)

chain = prompt | model

response = chain.invoke({
    "topic": "Generative AI"
})

print(response.content)

🖥️ Streamlit Applications

Some examples use Streamlit to create simple user interfaces for Generative AI applications.

Example:

import streamlit as st

st.header("Car Information Generator")

car_input = st.selectbox(
    "Select a car",
    [
        "Porsche 911",
        "Ford Mustang",
        "Mazda MX-5 Miata",
        "Toyota GR Supra",
        "Chevrolet Corvette"
    ]
)

if st.button("Show"):
    # Invoke LangChain pipeline
    pass


Run a Streamlit application using:

streamlit run prompt_ui.py


The application will open in your browser.

🧩 Prompt Templates

Prompt templates allow dynamic values to be inserted into prompts.

For example:

template = PromptTemplate(
    template="""
    Explain the following topic:

    Topic: {topic}
    Audience: {audience}
    Explanation Length: {length}
    """,
    input_variables=[
        "topic",
        "audience",
        "length"
    ]
)


The template can then be invoked using:

prompt = template.invoke({
    "topic": "Machine Learning",
    "audience": "Beginner",
    "length": "Short"
})


This approach makes prompts reusable instead of hardcoding every request.

📄 Saving and Loading Prompts

LangChain allows prompts to be serialized and reused.

For example:

template.save("template.json")


The saved prompt can later be loaded:

from langchain_core.prompts import load_prompt

template = load_prompt("template.json")


This makes it possible to separate prompt configuration from application logic.

🛠️ Learning Roadmap

The repository is being developed progressively.

Fundamentals

 Python environment setup

 Environment variables

 Hugging Face integration

 LLM invocation

 Chat model integration

Prompt Engineering

 PromptTemplate

 Dynamic prompt variables

 Prompt validation

 Saving prompts

 Loading prompts

LangChain

 HuggingFaceEndpoint

 ChatHuggingFace

 Chains

 Prompt + Model pipelines

Applications

 Streamlit integration

 Interactive LLM applications

 More practical GenAI applications

 Retrieval-Augmented Generation (RAG)

 Vector databases

 Embeddings

 Document loaders

 Agents and tools

 Advanced LLM workflows

🎯 Learning Objectives

By working through this repository, the goal is to understand:

How LLMs are accessed programmatically.

How Hugging Face models can be integrated with LangChain.

How prompts can be designed and reused.

How dynamic user inputs can be passed to prompts.

How LangChain components can be connected using chains.

How to build simple user interfaces for GenAI applications.

How to structure a Python project for Generative AI development.

How to gradually move from simple LLM calls to more advanced AI applications.

🧪 Example Workflow

A typical application in this repository follows a workflow similar to:

                User
                 │
                 ▼
        ┌─────────────────┐
        │ Streamlit UI    │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ User Inputs     │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Prompt Template │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ LangChain       │
        │ Chain           │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Hugging Face    │
        │ LLM             │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Generated       │
        │ Response        │
        └─────────────────┘

⚠️ Common Issues
Prompt variable mismatch

Make sure the variables in your prompt match the variables supplied to invoke().

For example, if the template contains:

{car_input}
{car_details}
{length_input}


then the invocation should contain:

{
    "car_input": car_input,
    "car_details": car_details,
    "length_input": length_input
}


The names must match exactly.

Environment variable not found

Make sure .env exists and contains:

HF_TOKEN=your_token


And load it before creating the model:

from dotenv import load_dotenv

load_dotenv()

Model/provider compatibility

Different Hugging Face models and inference providers may support different tasks.

For example:

task="conversational"


may be required for a conversational model/provider combination.

Always verify the model's current Hugging Face configuration when changing models.

🔒 Security

Do not commit secrets to this repository.

Never add:

.env
API keys
Access tokens
Passwords
Private credentials


to GitHub.

Recommended .gitignore:

.env
venv/
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/

📚 Topics Covered

This repository aims to cover a broad range of Generative AI concepts over time:

Python
  │
  ├── LLMs
  │
  ├── Hugging Face
  │
  ├── LangChain
  │     ├── Prompts
  │     ├── Models
  │     ├── Chains
  │     ├── Parsers
  │     └── Agents
  │
  ├── Prompt Engineering
  │
  ├── Streamlit
  │
  ├── Embeddings
  │
  ├── Vector Databases
  │
  ├── RAG
  │
  └── AI Applications

🤝 Contributions

This repository is primarily a learning project.

Suggestions, improvements, corrections, and additional examples are welcome.

If you find an issue or have an idea for improvement, feel free to open an issue or submit a pull request.

📌 Disclaimer

This repository is created for educational and experimental purposes.

LLM-generated responses may contain inaccurate or incomplete information. Applications built using these examples should validate important information before relying on generated output.

👨‍💻 Author

Your Name

Learning and building with:

Generative AI

LangChain

Large Language Models

Hugging Face

Python

AI Application Development

⭐ If You Find This Repository Useful

Feel free to star ⭐ the repository and follow along as more LangChain and Generative AI concepts are added.

Happy Learning and Building! 🚀
