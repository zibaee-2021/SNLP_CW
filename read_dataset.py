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

file_path = '/Users/kristiapantazi/Desktop/My_documents/NLP/Task10BGoldenEnriched/10B1_golden.json'

# Open the file in read mode
with open(file_path, 'r') as file:
    # Load the JSON data from the file
    data_3 = json.load(file)

print(json.dumps(data,indent=3))

yesno_data_3 = []

# Iterate through the JSON data
for item in data_3['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'yesno':
        # Add the yesno data to the list
        yesno_data_3.append(item)
print(len(yesno_data_3))
# Write the yesno data to a new JSON file
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/Yesno_golden1.json', 'w') as output_file:
    json.dump(yesno_data_3, output_file, indent=3)


factoid_data_3 = []

# Iterate through the JSON data
for item in data_3['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'factoid':
        # Add the yesno data to the list
        factoid_data_3.append(item)
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/factoid_golden1.json', 'w') as output_file:
    json.dump(factoid_data_3, output_file, indent=3)

len(factoid_data_3)

list_data_3 = []

# Iterate through the JSON data
for item in data_3['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'list':
        # Add the yesno data to the list
        list_data_3.append(item)
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/list_golden1.json', 'w') as output_file:
    json.dump(list_data_3, output_file, indent=3)

len(list_data_3)

summary_data_3 = []

# Iterate through the JSON data
for item in data_3['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'summary':
        # Add the yesno data to the list
       summary_data_3.append(item)
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/summary_golden1.json', 'w') as output_file:
    json.dump(summary_data_3, output_file, indent=3)

len(summary_data_3)


type_counts_3 = {}

# Iterate through the JSON data
for item in data_3['questions']:
    # Get the type of the question
    question_type = item['type']

    # If the type is already in the dictionary, increment the count
    if question_type in type_counts_3:
        type_counts_3[question_type] += 1
    # If the type is not in the dictionary, initialize the count to 1
    else:
        type_counts_3[question_type] = 1

# Print the counts for each type
for question_type, count in type_counts_3.items():
    print(f"Type: {question_type}, Count: {count}")




file_path = '/Users/kristiapantazi/Desktop/My_documents/NLP/Task10BGoldenEnriched/10B2_golden.json'

# Open the file in read mode
with open(file_path, 'r') as file:
    # Load the JSON data from the file
    data_4 = json.load(file)

print(json.dumps(data,indent=3))

yesno_data_4 = []

# Iterate through the JSON data
for item in data_4['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'yesno':
        # Add the yesno data to the list
        yesno_data_4.append(item)
print(len(yesno_data_4))
# Write the yesno data to a new JSON file
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/Yesno_golden2.json', 'w') as output_file:
    json.dump(yesno_data_4, output_file, indent=3)


factoid_data_4 = []

# Iterate through the JSON data
for item in data_4['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'factoid':
        # Add the yesno data to the list
        factoid_data_4.append(item)
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/factoid_golden2.json', 'w') as output_file:
    json.dump(factoid_data_4, output_file, indent=3)

print(len(factoid_data_4))

list_data_4 = []

# Iterate through the JSON data
for item in data_4['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'list':
        # Add the yesno data to the list
        list_data_4.append(item)
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/list_golden2.json', 'w') as output_file:
    json.dump(list_data_4, output_file, indent=3)

print(len(list_data_4))

summary_data_4 = []

# Iterate through the JSON data
for item in data_4['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'summary':
        # Add the yesno data to the list
       summary_data_4.append(item)
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/summary_golden_2.json', 'w') as output_file:
    json.dump(summary_data_4, output_file, indent=3)

print(len(summary_data_4))


type_counts_4 = {}

# Iterate through the JSON data
for item in data_4['questions']:
    # Get the type of the question
    question_type = item['type']

    # If the type is already in the dictionary, increment the count
    if question_type in type_counts_4:
        type_counts_4[question_type] += 1
    # If the type is not in the dictionary, initialize the count to 1
    else:
        type_counts_4[question_type] = 1

# Print the counts for each type
for question_type, count in type_counts_4.items():
    print(f"Type: {question_type}, Count: {count}")




file_path = '/Users/kristiapantazi/Desktop/My_documents/NLP/Task10BGoldenEnriched/10B3_golden.json'

# Open the file in read mode
with open(file_path, 'r') as file:
    # Load the JSON data from the file
    data_5 = json.load(file)

print(json.dumps(data_5,indent=3))

yesno_data_5 = []

# Iterate through the JSON data
for item in data_5['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'yesno':
        # Add the yesno data to the list
        yesno_data_5.append(item)
print(len(yesno_data_5))
# Write the yesno data to a new JSON file
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/Yesno_golden3.json', 'w') as output_file:
    json.dump(yesno_data_5, output_file, indent=3)


factoid_data_5 = []

# Iterate through the JSON data
for item in data_5['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'factoid':
        # Add the yesno data to the list
        factoid_data_5.append(item)
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/factoid_golden3.json', 'w') as output_file:
    json.dump(factoid_data_5, output_file, indent=3)

len(factoid_data_5)

list_data_5 = []

# Iterate through the JSON data
for item in data_5['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'list':
        # Add the yesno data to the list
        list_data_5.append(item)
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/list_golden3.json', 'w') as output_file:
    json.dump(list_data_5, output_file, indent=3)

len(list_data_5)

summary_data_5 = []

# Iterate through the JSON data
for item in data_5['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'summary':
        # Add the yesno data to the list
       summary_data_5.append(item)
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/summary_golden3.json', 'w') as output_file:
    json.dump(summary_data_5, output_file, indent=3)

len(summary_data_5)


type_counts_5 = {}

# Iterate through the JSON data
for item in data_5['questions']:
    # Get the type of the question
    question_type = item['type']

    # If the type is already in the dictionary, increment the count
    if question_type in type_counts_5:
        type_counts_5[question_type] += 1
    # If the type is not in the dictionary, initialize the count to 1
    else:
        type_counts_5[question_type] = 1

# Print the counts for each type
for question_type, count in type_counts_5.items():
    print(f"Type: {question_type}, Count: {count}")




file_path = '/Users/kristiapantazi/Desktop/My_documents/NLP/Task10BGoldenEnriched/10B4_golden.json'

# Open the file in read mode
with open(file_path, 'r') as file:
    # Load the JSON data from the file
    data_6 = json.load(file)

print(json.dumps(data_6,indent=3))

yesno_data_6 = []

# Iterate through the JSON data
for item in data_6['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'yesno':
        # Add the yesno data to the list
        yesno_data_6.append(item)
print(len(yesno_data_6))
# Write the yesno data to a new JSON file
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/Yesno_golden4.json', 'w') as output_file:
    json.dump(yesno_data_6, output_file, indent=3)


factoid_data_6 = []

# Iterate through the JSON data
for item in data_6['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'factoid':
        # Add the yesno data to the list
        factoid_data_6.append(item)
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/factoid_golden4.json', 'w') as output_file:
    json.dump(factoid_data_6, output_file, indent=3)

print(len(factoid_data_6))

list_data_6 = []

# Iterate through the JSON data
for item in data_6['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'list':
        # Add the yesno data to the list
        list_data_6.append(item)
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/list_golden4.json', 'w') as output_file:
    json.dump(list_data_6, output_file, indent=3)

print(len(list_data_6))

summary_data_6 = []

# Iterate through the JSON data
for item in data_6['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'summary':
        # Add the yesno data to the list
       summary_data_6.append(item)
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/summary_golden4.json', 'w') as output_file:
    json.dump(summary_data_6, output_file, indent=3)

len(summary_data_6)


type_counts_6 = {}

# Iterate through the JSON data
for item in data_6['questions']:
    # Get the type of the question
    question_type = item['type']

    # If the type is already in the dictionary, increment the count
    if question_type in type_counts_6:
        type_counts_6[question_type] += 1
    # If the type is not in the dictionary, initialize the count to 1
    else:
        type_counts_6[question_type] = 1

# Print the counts for each type
for question_type, count in type_counts_6.items():
    print(f"Type: {question_type}, Count: {count}")






file_path = '/Users/kristiapantazi/Desktop/My_documents/NLP/Task10BGoldenEnriched/10B5_golden.json'

# Open the file in read mode
with open(file_path, 'r') as file:
    # Load the JSON data from the file
    data_7 = json.load(file)

print(json.dumps(data_7,indent=3))

yesno_data_7 = []

# Iterate through the JSON data
for item in data_7['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'yesno':
        # Add the yesno data to the list
        yesno_data_7.append(item)
print(len(yesno_data_7))
# Write the yesno data to a new JSON file
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/Yesno_golden5.json', 'w') as output_file:
    json.dump(yesno_data_7, output_file, indent=3)


factoid_data_7 = []

# Iterate through the JSON data
for item in data_7['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'factoid':
        # Add the yesno data to the list
        factoid_data_7.append(item)
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/factoid_golden5.json', 'w') as output_file:
    json.dump(factoid_data_7, output_file, indent=3)

print(len(factoid_data_7))

list_data_7 = []

# Iterate through the JSON data
for item in data_7['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'list':
        # Add the yesno data to the list
        list_data_7.append(item)
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/list_golden5.json', 'w') as output_file:
    json.dump(list_data_7, output_file, indent=3)

print(len(list_data_7))

summary_data_7 = []

# Iterate through the JSON data
for item in data_7['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'summary':
        # Add the yesno data to the list
       summary_data_7.append(item)
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/summary_golden5.json', 'w') as output_file:
    json.dump(summary_data_7, output_file, indent=3)

len(summary_data_7)


type_counts_7 = {}

# Iterate through the JSON data
for item in data_7['questions']:
    # Get the type of the question
    question_type = item['type']

    # If the type is already in the dictionary, increment the count
    if question_type in type_counts_7:
        type_counts_7[question_type] += 1
    # If the type is not in the dictionary, initialize the count to 1
    else:
        type_counts_7[question_type] = 1

# Print the counts for each type
for question_type, count in type_counts_7.items():
    print(f"Type: {question_type}, Count: {count}")


file_path = '/Users/kristiapantazi/Desktop/My_documents/NLP/Task10BGoldenEnriched/10B6_golden.json'

# Open the file in read mode
with open(file_path, 'r') as file:
    # Load the JSON data from the file
    data_8 = json.load(file)

print(json.dumps(data_8,indent=3))

yesno_data_8 = []

# Iterate through the JSON data
for item in data_8['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'yesno':
        # Add the yesno data to the list
        yesno_data_8.append(item)
print(len(yesno_data_8))
# Write the yesno data to a new JSON file
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/Yesno_golden6.json', 'w') as output_file:
    json.dump(yesno_data_8, output_file, indent=3)


factoid_data_8 = []

# Iterate through the JSON data
for item in data_8['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'factoid':
        # Add the yesno data to the list
        factoid_data_8.append(item)
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/factoid_golden6.json', 'w') as output_file:
    json.dump(factoid_data_8, output_file, indent=3)

print(len(factoid_data_8))

list_data_8 = []

# Iterate through the JSON data
for item in data_8['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'list':
        # Add the yesno data to the list
        list_data_8.append(item)
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/list_golden6.json', 'w') as output_file:
    json.dump(list_data_8, output_file, indent=3)

print(len(list_data_8))

summary_data_8 = []

# Iterate through the JSON data
for item in data_8['questions']:
    # Check if the type is 'yesno'
    if item['type'] == 'summary':
        # Add the yesno data to the list
       summary_data_8.append(item)
with open('/Users/kristiapantazi/Desktop/My_documents/NLP/summary_golden6.json', 'w') as output_file:
    json.dump(summary_data_8, output_file, indent=3)

print(len(summary_data_8))


type_counts_8 = {}

# Iterate through the JSON data
for item in data_8['questions']:
    # Get the type of the question
    question_type = item['type']

    # If the type is already in the dictionary, increment the count
    if question_type in type_counts_8:
        type_counts_8[question_type] += 1
    # If the type is not in the dictionary, initialize the count to 1
    else:
        type_counts_8[question_type] = 1

# Print the counts for each type
for question_type, count in type_counts_8.items():
    print(f"Type: {question_type}, Count: {count}")





file_paths = ['/Users/kristiapantazi/Desktop/My_documents/NLP/factoid_golden1.json','/Users/kristiapantazi/Desktop/My_documents/NLP/factoid_golden2.json',
              '/Users/kristiapantazi/Desktop/My_documents/NLP/factoid_golden3.json', '/Users/kristiapantazi/Desktop/My_documents/NLP/factoid_golden4.json',
              '/Users/kristiapantazi/Desktop/My_documents/NLP/factoid_golden5.json', '/Users/kristiapantazi/Desktop/My_documents/NLP/factoid_golden6.json']
output_file_path = '/Users/kristiapantazi/Desktop/My_documents/NLP/factoid_golden_merged.json'

# Initialize an empty list to store all data from individual files
merged_data = []
print(len(merged_data))
# Iterate over each file path
for file_path in file_paths:
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Load the JSON data from the file
        data = json.load(file)
        # Extend the merged_data list with the data from the current file
        merged_data.extend(data)

# Write the merged data to a new JSON file
with open(output_file_path, 'w') as output_file:
    json.dump(merged_data, output_file, indent=4)



file_paths = ['/Users/kristiapantazi/Desktop/My_documents/NLP/list_golden1.json','/Users/kristiapantazi/Desktop/My_documents/NLP/list_golden2.json',
              '/Users/kristiapantazi/Desktop/My_documents/NLP/list_golden3.json', '/Users/kristiapantazi/Desktop/My_documents/NLP/list_golden4.json',
              '/Users/kristiapantazi/Desktop/My_documents/NLP/list_golden5.json', '/Users/kristiapantazi/Desktop/My_documents/NLP/list_golden6.json']
output_file_path = '/Users/kristiapantazi/Desktop/My_documents/NLP/list_golden_merged.json'

# Initialize an empty list to store all data from individual files
merged_data = []

# Iterate over each file path
for file_path in file_paths:
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Load the JSON data from the file
        data = json.load(file)
        # Extend the merged_data list with the data from the current file
        merged_data.extend(data)

# Write the merged data to a new JSON file
with open(output_file_path, 'w') as output_file:
    json.dump(merged_data, output_file, indent=4)


file_paths = ['/Users/kristiapantazi/Desktop/My_documents/NLP/summary_golden1.json','/Users/kristiapantazi/Desktop/My_documents/NLP/summary_golden_2.json',
              '/Users/kristiapantazi/Desktop/My_documents/NLP/summary_golden3.json', '/Users/kristiapantazi/Desktop/My_documents/NLP/summary_golden4.json',
              '/Users/kristiapantazi/Desktop/My_documents/NLP/summary_golden5.json', '/Users/kristiapantazi/Desktop/My_documents/NLP/summary_golden6.json']
output_file_path = '/Users/kristiapantazi/Desktop/My_documents/NLP/summary_golden_merged.json'

# Initialize an empty list to store all data from individual files
merged_data = []

# Iterate over each file path
for file_path in file_paths:
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Load the JSON data from the file
        data = json.load(file)
        # Extend the merged_data list with the data from the current file
        merged_data.extend(data)

# Write the merged data to a new JSON file
with open(output_file_path, 'w') as output_file:
    json.dump(merged_data, output_file, indent=4)



file_paths = ['/Users/kristiapantazi/Desktop/My_documents/NLP/Yesno_golden1.json','/Users/kristiapantazi/Desktop/My_documents/NLP/Yesno_golden2.json',
              '/Users/kristiapantazi/Desktop/My_documents/NLP/Yesno_golden3.json', '/Users/kristiapantazi/Desktop/My_documents/NLP/Yesno_golden4.json',
              '/Users/kristiapantazi/Desktop/My_documents/NLP/Yesno_golden5.json', '/Users/kristiapantazi/Desktop/My_documents/NLP/Yesno_golden6.json']
output_file_path = '/Users/kristiapantazi/Desktop/My_documents/NLP/Yesno_golden_merged.json'

# Initialize an empty list to store all data from individual files
merged_data = []

# Iterate over each file path
for file_path in file_paths:
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Load the JSON data from the file
        data = json.load(file)
        # Extend the merged_data list with the data from the current file
        merged_data.extend(data)

# Write the merged data to a new JSON file
with open(output_file_path, 'w') as output_file:
    json.dump(merged_data, output_file, indent=4)


input_file_path = '/Users/kristiapantazi/Desktop/My_documents/NLP/factoid_golden_merged.json'
output_file_path = '/Users/kristiapantazi/Desktop/My_documents/NLP/factoid_golden_merged_norep.json'

with open(input_file_path, 'r') as input_file:
    # Load the JSON data from the input file
    data = json.load(input_file)

# Define a function to check for duplicates based on specific criteria
def remove_duplicates(data):
    seen = set()
    unique_data = []
    for entry in data:
        # Assuming 'id' is the key based on which duplicates are identified
        identifier = tuple(entry.get('documents'))  # Convert to tuple
        if identifier not in seen:
            unique_data.append(entry)
            seen.add(identifier)
    return unique_data

# Remove duplicates from the data
filtered_data = remove_duplicates(data)

# Write the filtered data to the output file
with open(output_file_path, 'w') as output_file:
    json.dump(filtered_data, output_file, indent=4)


input_file_path = '/Users/kristiapantazi/Desktop/My_documents/NLP/list_golden_merged.json'
output_file_path = '/Users/kristiapantazi/Desktop/My_documents/NLP/list_golden_merged_norep.json'

with open(input_file_path, 'r') as input_file:
    # Load the JSON data from the input file
    data = json.load(input_file)

# Define a function to check for duplicates based on specific criteria
def remove_duplicates(data):
    seen = set()
    unique_data = []
    for entry in data:
        # Assuming 'id' is the key based on which duplicates are identified
        identifier = tuple(entry.get('documents'))  # Convert to tuple
        if identifier not in seen:
            unique_data.append(entry)
            seen.add(identifier)
    return unique_data

# Remove duplicates from the data
filtered_data = remove_duplicates(data)

# Write the filtered data to the output file
with open(output_file_path, 'w') as output_file:
    json.dump(filtered_data, output_file, indent=4)






input_file_path = '/Users/kristiapantazi/Desktop/My_documents/NLP/Yesno_golden_merged.json'
output_file_path = '/Users/kristiapantazi/Desktop/My_documents/NLP/Yesno_golden_merged_norep.json'

with open(input_file_path, 'r') as input_file:
    # Load the JSON data from the input file
    data = json.load(input_file)

# Define a function to check for duplicates based on specific criteria
def remove_duplicates(data):
    seen = set()
    unique_data = []
    for entry in data:
        # Assuming 'id' is the key based on which duplicates are identified
        identifier = tuple(entry.get('documents'))  # Convert to tuple
        if identifier not in seen:
            unique_data.append(entry)
            seen.add(identifier)
    return unique_data

# Remove duplicates from the data
filtered_data = remove_duplicates(data)

# Write the filtered data to the output file
with open(output_file_path, 'w') as output_file:
    json.dump(filtered_data, output_file, indent=4)





input_file_path = '/Users/kristiapantazi/Desktop/My_documents/NLP/summary_golden_merged.json'
output_file_path = '/Users/kristiapantazi/Desktop/My_documents/NLP/summary_golden_merged_norep.json'

with open(input_file_path, 'r') as input_file:
    # Load the JSON data from the input file
    data = json.load(input_file)

# Define a function to check for duplicates based on specific criteria
def remove_duplicates(data):
    seen = set()
    unique_data = []
    for entry in data:
        # Assuming 'id' is the key based on which duplicates are identified
        identifier = tuple(entry.get('documents'))  # Convert to tuple
        if identifier not in seen:
            unique_data.append(entry)
            seen.add(identifier)
    return unique_data

# Remove duplicates from the data
filtered_data = remove_duplicates(data)

# Write the filtered data to the output file
with open(output_file_path, 'w') as output_file:
    json.dump(filtered_data, output_file, indent=4)



