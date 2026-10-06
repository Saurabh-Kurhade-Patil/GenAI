from langchain_core.prompts import ChatPromptTemplate

'''
This code is used to create a prompt that can be further send to a chatBot
'''

chat_template = ChatPromptTemplate([
    ('system','You are a {domain} language expert, which will help translate into your experise language'),
    ('human','How are you {name}?')
])

result = chat_template.invoke({'domain':'Sindhi','name':'Muskan'})

print(result)
