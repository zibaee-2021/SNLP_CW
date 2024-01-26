import pandas as pd

# Assuming 'data.json' is your JSON file
file_path = 'dataset/11B1_golden.json'

# Read the JSON file into a DataFrame
df = pd.read_json(file_path)

questions = pd.DataFrame(df['questions'][0])

# Display the DataFrame
print("Column names as a list:", list(questions.columns))

print(questions[['body', 'ideal_answer', 'exact_answer']].loc[0])
