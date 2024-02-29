# Use a pipeline as a high-level helper
from transformers import pipeline
# Load model directly
from transformers import AutoTokenizer, AutoModel

import time
import pandas as pd
from tqdm import tqdm


if __name__ == '__main__':
    pipe = pipeline("feature-extraction", model="dmis-lab/biobert-v1.1")

    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")


    st = time.time()
    # Assuming 'data.json' is your JSON file
    # Open the file and load its content into a dictionary
    df = pd.read_json('dataset/QALM/test/mcq/bioasq_mcq_test.jsonl', lines=True)

    # Print all option choice in the MCQ dataset
    print('Loaded all questions. Options: ', df['answer'].unique())
    print(f'1. Read QALM mcq test = {time.time() - st} secs')

    # Go through the questions in the dataframe
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):



