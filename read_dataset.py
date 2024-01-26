import json

file_path = '/Users/kristiapantazi/Desktop/My_documents/NLP/BioASQ-training11b/training11b.json'

# Open the file in read mode
with open(file_path, 'r') as file:
    # Load the JSON data from the file
    data = json.load(file)

print(json.dumps(data,indent=3))

yesno_data = []

# Iterate through the JSON data
for item in data['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'yesno':
        # Add the yesno data to the list
        yesno_data.append(item)
print(len(yesno_data))
# Write the yesno data to a new JSON file
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/Yesno_11.json', 'w') as output_file:
    json.dump(yesno_data, output_file, indent=3)


factoid_data_1 = []

# Iterate through the JSON data
for item in data['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'factoid':
        # Add the yesno data to the list
        factoid_data_1.append(item)
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/factoid_11.json', 'w') as output_file:
    json.dump(factoid_data_1, output_file, indent=3)

len(factoid_data_1)

list_data_1 = []

# Iterate through the JSON data
for item in data['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'list':
        # Add the yesno data to the list
        list_data_1.append(item)
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/list_11.json', 'w') as output_file:
    json.dump(list_data_1, output_file, indent=3)

len(list_data_1)

summary_data_1 = []

# Iterate through the JSON data
for item in data['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'summary':
        # Add the yesno data to the list
       summary_data_1.append(item)
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/summary_11.json', 'w') as output_file:
    json.dump(summary_data_1, output_file, indent=3)

len(summary_data_1)

file_path = '/Users/kristiapantazi/Desktop/My_documents/NLP/BioASQ-training10b/training10b.json'

# Open the file in read mode
with open(file_path, 'r') as file:
    # Load the JSON data from the file
    data_1 = json.load(file)

print(json.dumps(data, indent=3))


yesno_data_2 = []

# Iterate through the JSON data
for item in data_1['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'yesno':
        # Add the yesno data to the list
        yesno_data_2.append(item)
print(len(yesno_data_2))
# Write the yesno data to a new JSON file
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/Yesno_10.json', 'w') as output_file:
    json.dump(yesno_data_2, output_file, indent=3)


print(len(yesno_data_2))

factoid_data_2 = []

# Iterate through the JSON data
for item in data_1['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'factoid':
        # Add the yesno data to the list
        factoid_data_2.append(item)

with open('/Users/kristiapantazi/Desktop/My_documents/NLP/factoid_10.json', 'w') as output_file:
    json.dump(factoid_data_2, output_file, indent=3)

len(factoid_data_1)
len(factoid_data_2)

list_data_2 = []

# Iterate through the JSON data
for item in data_1['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'list':
        # Add the yesno data to the list
        list_data_2.append(item)
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/list_10.json', 'w') as output_file:
    json.dump(list_data_2, output_file, indent=3)

len(list_data_2)


summary_data_2 = []

# Iterate through the JSON data
for item in data_1['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'summary':
        # Add the yesno data to the list
       summary_data_2.append(item)
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/summary_10.json', 'w') as output_file:
    json.dump(summary_data_2, output_file, indent=3)

len(summary_data_2)

type_counts = {}

# Iterate through the JSON data
for item in data['questions']:
    # Get the type of the question
    question_type = item['type']

    # If the type is already in the dictionary, increment the count
    if question_type in type_counts:
        type_counts[question_type] += 1
    # If the type is not in the dictionary, initialize the count to 1
    else:
        type_counts[question_type] = 1

# Print the counts for each type
for question_type, count in type_counts.items():
    print(f"Type: {question_type}, Count: {count}")

type_counts_1 = {}

# Iterate through the JSON data
for item in data_1['questions']:
    # Get the type of the question
    question_type = item['type']

    # If the type is already in the dictionary, increment the count
    if question_type in type_counts_1:
        type_counts_1[question_type] += 1
    # If the type is not in the dictionary, initialize the count to 1
    else:
        type_counts_1[question_type] = 1

# Print the counts for each type
for question_type, count in type_counts_1.items():
    print(f"Type: {question_type}, Count: {count}")

print(len(yesno_data))
print(len(yesno_data_2))
print(len(factoid_data_1))
print(len(factoid_data_2))
print(len(list_data_1))
print(len(list_data_2))
print(len(summary_data_1))
print(len(summary_data_2))