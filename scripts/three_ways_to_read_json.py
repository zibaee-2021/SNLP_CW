from langchain_community.document_loaders import JSONLoader
import json
from pathlib import Path
from pprint import pprint
import pandas as pd
import os


# 1.a. OPENING JSONL FILES - USING CONTEXT MANAGER & JSON LIBRARY:
def read_json_1a(file_path: str):
    data = list()

    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            pprint(json.loads(line))

    pprint(data)


# 1.b. OPENING JSON FILES - USING JSON LIBRARY:
def read_json_1b(file_path: str):
    data = json.loads(Path(file_path).read_text())  # also indirect way to confirm json file has valid json structure.
    pprint(data)
    questions = list()
    for person in data['person']:
        questions.append(person['age'])


# 2.a. OPENING JSONL FILES - USING PANDAS:
def read_json_2a(file_path: str):
    df = pd.read_json(file_path, lines=True)
    print('Loaded all questions. Options: ', df['answer'].unique())


# 2.b. OPENING JSON FILES -USING PANDAS:
def read_json_2b(file_path: str):
    data_df = pd.read_json(file_path)  # leads to 1 row per person, but all nested data is crammed into one column.
    persons = data_df['person']
    ages = list()
    for person in persons:
        ages.append(person['age'])


# 3.a. OPENING JSONL FILES - USING JQ AND LANGCHAIN:
def read_json_3a(jsonl_file_path):
    loader = JSONLoader(
        file_path=jsonl_file_path,
        jq_schema='.text',
        text_content=False,
        json_lines=True)

    data = loader.load()
    pprint(data)


# 3.b. OPENING JSONL FILES - USING JQ AND LANGCHAIN:
# SHOWING 3 DIFFERENT SYNTAXES YOU MIGHT USE:
def read_json_3b(file_path):
    # SYNTAX 1:
    loader = JSONLoader(
        file_path=file_path,
        jq_schema='.person[] | {person_name: .name, person_age: .age}',
        text_content=False
    )
    data1 = loader.load()
    print(f'data1: {data1}')

    # SYNTAX 2:
    loader2 = JSONLoader(
        file_path=file_path,
        jq_schema='.person[].name'
    )
    data2 = loader2.load()
    print(f'data2: {data2}')

    # SYNTAX 3:
    loader3 = JSONLoader(
        file_path=jsonl_file_path,
        jq_schema='.person[]',
        content_key='name',
        metadata_func=metadata_func
    )
    data3 = loader3.load()
    print(f'data3: {data3}')


# Define the metadata extraction function.
# (But all I'm doing in this one actually is to trim down the absolute path to a relative path string).
def metadata_func(record: dict, metadata: dict) -> dict:

    if "source" in metadata:
        source = metadata["source"].split("/")
        source = source[source.index("sandpit"):]
        metadata["source"] = "/".join(source)

    return metadata


if __name__ == '__main__':
    print(os.getcwd())
    jsonl_file_path = 'datasets/QALM/test/mcq/bioasq_mcq_test.jsonl'
    read_json_3a(jsonl_file_path)


# Other possible jq_schema:
# JSON        -> [{"text": ...}, {"text": ...}, {"text": ...}]
# jq_schema   -> ".[].text"
#
# JSON        -> {"key": [{"text": ...}, {"text": ...}, {"text": ...}]}
# jq_schema   -> ".key[].text"
#
# JSON        -> ["...", "...", "..."]
# jq_schema   -> ".[]"

