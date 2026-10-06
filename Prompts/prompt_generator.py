from langchain_core.prompts import PromptTemplate
'''
This code will generate a json file 
'''
# template
template = PromptTemplate(
    template = """
    Please Help me with the details of the Sport car "{car_input}" with following specification:
    Explaination style: {car_details}
    Explaination Legth: {length_input}
    
    Share the details of the car as per the required explaination style, if don't have data about the car respond with "Insufficient Data avliable" instead guessing.
    Keep it clear for real world data and it's about the Car and not the movie Cars
    """,
    input_variables = ['car_input', 'car_details', 'length_input'],
    validate_template = True
)

template.save('temp.json')