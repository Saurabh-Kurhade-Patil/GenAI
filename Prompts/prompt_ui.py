from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate,load_prompt
from dotenv import load_dotenv
import streamlit as st
'''
This code is a UI based application that will summarize the Cars details in the user requested format
'''
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="conversational"
)
model = ChatHuggingFace(llm=llm)
st.header('Cars')

car_input = st.selectbox("Select the Car",['Porsche 911','Ford Mustang','Mazda MX-5 Miata', 'Toyota GR Supra', 'Chevrolet Corvette'])

car_details = st.selectbox("Select the decriptive style", ['Beginner', 'Techinical', 'Proficient'])

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt('temp.json')

if st.button('Show'):
    chain = template | model
    result = chain.invoke({
        'car_input': car_input,
        'car_details': car_details,
        'length_input': length_input
    })
    st.write(result.content)



