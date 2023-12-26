import gradio as gr
import numpy as np
import pandas as pd
import pyresparser
import joblib
from datetime import datetime

# Define the personality mapping
personality_mapping = {
    0: "Dependable",
    1: "Serious",
    2: "Responsible",
    3: "Extraverted",
    4: "Lively"
}

# Load the pre-trained ML model
model = joblib.load('personality_model.joblib')

def predict_personality(resume_file, question):
    # Use pyresparser to extract information from the resume
    resume_data = pyresparser.read_file(resume_file)

    # Extract relevant information from the resume_data dictionary
    gender_value = 0 if resume_data['gender'] == "M" else 1
    dob = resume_data.get('date_of_birth', '')
    age_value = calculate_age(dob)
    
    # age_value = resume_data['date_of_birth']  # You may need to calculate age from the date of birth
    openness_value = resume_data['skills']['openness'] if 'skills' in resume_data else 0
    neuroticism_value = resume_data['skills']['neuroticism'] if 'skills' in resume_data else 0
    conscientiousness_value = resume_data['skills']['conscientiousness'] if 'skills' in resume_data else 0
    agreeableness_value = resume_data['skills']['agreeableness'] if 'skills' in resume_data else 0
    extraversion_value = resume_data['skills']['extraversion'] if 'skills' in resume_data else 0

    # Define user_input with the extracted values
    user_input = np.array([gender_value, age_value, openness_value, neuroticism_value, conscientiousness_value, agreeableness_value, extraversion_value]).reshape(1, -1)

    # Predict personality using the trained model
    prediction = model.predict(user_input)
    value = int(np.floor(prediction[0]))

    # Map predicted personality to the actual personality
    predicted_personality = personality_mapping.get(value, "Unknown")

    # Calculate the percentage of skills match
    skills_required = ["Problem Solving and Technical skills", "OOPS", "data structures", "java", "python", "algorithms", "matlab"]
    skills = resume_data["skills"]
    match = sum(skill in skills for skill in skills_required)
    percentage = (match / len(skills_required)) * 100

    # Formulate a response based on the question
    if "personality" in question.lower():
        response = f"The predicted personality based on the resume is: {predicted_personality}"
    elif "skills" in question.lower():
        response = f"The person possesses {percentage:.2f}% of the required skills."
    else:
        response = "I'm sorry, I couldn't understand the question."

    return response
def calculate_age(date_of_birth):
    try:
        birth_date = datetime.strptime(date_of_birth, '%Y-%m-%d')
        today = datetime.now()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return age
    except ValueError:
        return 0 
    

# Gradio Interface
iface = gr.Interface(
    fn=predict_personality,
    inputs=[
        gr.File(type="file", label="Upload Resume"),
        gr.Textbox(type="text", label="Ask a question about the resume (e.g., 'What is the personality?')")
    ],
    outputs=gr.Textbox(type="text", label="Chatbot Response"),
    live=True
)

iface.launch()
