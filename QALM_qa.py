import pandas as pd

from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import(
StreamingStdOutCallbackHandler
)

def load_llama(model_path):
    llm = LlamaCpp(
        model_path=model_path,
        streaming=False,
        n_ctx=6000,
        n_gpu_layers=1024,
        # n_batch=30,
        # temperature=0.9,
        max_tokens=4095,
        # n_parts=1,
        # verbose=0
    )

    return llm

if __name__ == '__main__':
    # Assuming 'data.json' is your JSON file
    json_file_path = 'dataset/QALM/train/qa/bioasq_qa_train.jsonl'

    # Open the file and load its content into a dictionary
    df = pd.read_json(json_file_path, lines=True)

    # Load the llama model
    llm = load_llama("../llama.cpp/models/llama-2-7b-chat.Q4_K_M.gguf")

    for index, row in df.iterrows():
        prompt = row['prompt']
        text = row['text']

        prompt = PromptTemplate.from_template(f'{prompt}: {text}')
        final_prompt = prompt.format(
            prompt=prompt,
            text=text
        )

        print('------------------------------------------------------------')

        print(f'{prompt}: {text}')

        print('\nLLM generated answer:', llm(final_prompt))

        print('\nGolden answer:', row['answer'])

        print('------------------------------------------------------------')
