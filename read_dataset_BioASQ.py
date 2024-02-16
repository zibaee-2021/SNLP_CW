import pandas as pd
import json

# Assuming 'data.json' is your JSON file
file_path = 'dataset/Task11BGoldenEnriched/11B1_golden.json'

# Open the file and load its content into a dictionary
with open(file_path, 'r') as file:
    data_dict = json.load(file)

question_list = []

for question in data_dict['questions']:
    question_list.append(question)
    print(question['body'], question['ideal_answer'], '\n')
