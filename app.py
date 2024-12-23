import pandas as pd
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

groq_api = os.getenv('GROQ_API_KEY')

st.set_page_config('Time Table Creator')

st.title('__Create Your Time Table__',)

# loading the csv data and converting it to be compatible in the prompt

T1 = pd.read_csv('Period Allotment.csv')

T1_ = T1.to_json(orient='records')

T2 = pd.read_csv('Class-Teacher Mapping.csv')

T2_ = T2.to_json(orient='records')

T3 = pd.read_csv('Subject-Class Mapping.csv')

T3_ = T3.to_json(orient='records')

# creating prompt for the LLM model

template = '''
You're a data analysis expert and time table creator for schools with extensive experience in interpreting structured data formats, especially list of dictionaries 
derived from CSV files. Your specialty is breaking down complex datasets into clear, concise explanations that highlight key insights and trends.
Your task is to analyze three different lists of dictionaries that I have converted from CSV files and create the time table with the help of it. 
Here are the details of list:

- {list1} : This list1 is having multiple dictionaries with keys as Subject,Jr. KG,Sr. KG and I. The Subject key have subjects like english,science as values
            and keys Jr. KG,Sr. KG,I are having values as 2,3,4 etc. which represents the number periods of that subject for the class in a week.

- {list2} : This is a list of dictionaries representing information about various classes and their home room teachers. Each dictionary in the list has the following structure:
            Class : The name of the class (e.g., "Nursery", "Jr.KG A").
            Home_room_Teachers : The names of the teachers assigned to the home room for the class (e.g., "Riya", "Swarali", or "Kim", "Rukhsaar").
            Subjects : The subjects taught in that class (e.g., "All" if it's a general home room with all subjects).
            Class_alloted : The role or allocation of the class, which in this case seems to be "Home Room", indicating the class is managed by a home room teacher.

- {list3} : "This list3 having multiple dictionaries where each dictionary contains information about the teachers assigned to different subjects and their respective class sections. Here's a breakdown of the structure:
            The 'Subject' key represents the name of the subject (e.g., 'Eng' for English, 'Jr. KG A' for Junior Kindergarten A, etc.).
            The other keys, such as 'Jr. KG A', 'Jr. KG B', etc., represent the class sections (e.g., Junior Kindergarten A, B, C, etc.), and the values are the names of the teachers assigned to those class sections for that particular subject.
            For example : For the subject 'Eng', 'Jr. KG A' has teachers 'Nupur FT' and 'Pooja R' and 'Jr. KG B' has 'Maria' and 'Banu' as teachers, and so on.

This JSON format is created by using .to_json(orient='records') in pandas, where each dictionary corresponds to a record/row in the original CSV, 
with keys representing column names and values representing the data for that row.

important instructions to create time table : 

{user_input} and use the period allotment and Teacher Subject Mapping from the list1,list2 and list3.
Create the time tables for each division with respect to their classes.
Don't provide any additional instruction and infomation other than the Time Tables

'''

prompt = PromptTemplate(
    template=template,
    input_variables=['list1','list2','list3','user_input']
)


# using llama3 model with groq api 

llm = ChatGroq(model='llama3-70b-8192',api_key=groq_api)


# creating the chain 

chain = prompt|llm|StrOutputParser()


user_input = st.text_input(label='***Give Details***',placeholder='Provide detail regarding time interval,lunch break,assembly etc.')


timetable_format = '''
The Days must be in vertical axis and time-period in the horizontal axis
'''

if user_input:
    result = chain.invoke({'list1':T1_,'list2':T2_,'list3':T3_,'user_input':user_input+timetable_format})

    st.write(result)
    st.error('The result is AI generated make sure to make changes according to your need')

else:
    st.warning('provide the details')
