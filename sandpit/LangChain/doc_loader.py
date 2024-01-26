from langchain_community.document_loaders import JSONLoader
import json
from pathlib import Path
from pprint import pprint
import pandas as pd


if __name__ == '__main__':
    file_path = 'docs/sample.json'

    # 1. USING JSON LIBRARY:
    data = json.loads(Path(file_path).read_text())  # also indirect way to confirm json file has valid json structure.
    pprint(data)
    ages = list()
    for person in data['person']:
        ages.append(person['age'])

    # 2. USING PANDAS:
    data_df = pd.read_json(file_path)  # leads to 1 row per person, but all nested data is crammed into one column.
    persons = data_df['person']
    ages = list()
    for person in persons:
        ages.append(person['age'])

    # 3. USING JQ AND LANGCHAIN:
    loader = JSONLoader(
        file_path=file_path,
        jq_schema='.person[] | {person_name: .name, person_age: .age}',
        text_content=False
    )
    data1 = loader.load()
    print(f'data1: {data1}')

    loader2 = JSONLoader(
        file_path=file_path,
        jq_schema='.person[].name'
    )
    data2 = loader2.load()
    print(f'data2: {data2}')

    # Define the metadata extraction function.
    # (But all I'm doing in this one actually is to trim down the absolute path to a relative path string).
    def metadata_func(record: dict, metadata: dict) -> dict:

        if "source" in metadata:
            source = metadata["source"].split("/")
            source = source[source.index("sandpit"):]
            metadata["source"] = "/".join(source)

        return metadata

    loader3 = JSONLoader(
        file_path=file_path,
        jq_schema='.person[]',
        content_key='name',
        metadata_func=metadata_func
    )
    data3 = loader3.load()
    print(f'data3: {data3}')
    pass

# Other possible jq_schema:
# JSON        -> [{"text": ...}, {"text": ...}, {"text": ...}]
# jq_schema   -> ".[].text"
#
# JSON        -> {"key": [{"text": ...}, {"text": ...}, {"text": ...}]}
# jq_schema   -> ".key[].text"
#
# JSON        -> ["...", "...", "..."]
# jq_schema   -> ".[]"

