from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema, RetryOutputParser
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import(
StreamingStdOutCallbackHandler
)

from langchain_core.runnables import RunnableLambda, RunnableParallel

from langchain_openai import ChatOpenAI

from tqdm import tqdm

import torch
import pandas as pd
import json
import ast
import re

def load_llama(model_path):
    llm = LlamaCpp(
        model_path=model_path,
        streaming=False,
        n_ctx=6000,
        #n_threads=30,
        #n_gpu_layers=40,
        # n_batch=30,
        # temperature=0.9,
        max_tokens=4095,
        # n_parts=1,
        # verbose=0
    )

    return llm

def read_dict(input_string):
    # Define a regular expression to capture content between curly braces
    pattern1 = re.compile(r'```json\n(.*?)\n```', re.DOTALL)
    pattern2 = re.compile(r'```\n(.*?)\n```', re.DOTALL)

    # Find the first match in the input string
    if pattern1.search(input_string):
        return 1, pattern1.search(input_string).group(1)
    elif pattern2.search(input_string):
        return 2, pattern2.search(input_string).group(1)
    else:
        return 0, None

if __name__ == '__main__':
    # Load the llama model
    # model = load_llama("../llama.cpp/models/llama-2-7b-chat.Q4_K_M.gguf")
    model = load_llama("../llama.cpp/models/llama-2-13b-chat.Q4_K_M.gguf")
    # model = ChatOpenAI(temperature=0)

    # Assuming 'data.json' is your JSON file
    json_file_path = 'dataset/QALM/test/mcq/bioasq_mcq_test.jsonl'

    # Open the file and load its content into a dictionary
    df = pd.read_json(json_file_path, lines=True)

    print('Options: ', df['answer'].unique())

    print('Cuda is available: ', torch.cuda.is_available())

    # Set the schema for structured output
    response_schemas = [
        ResponseSchema(name="answer", description="The correct option answer."),
        ResponseSchema(name="explanation", description="The explanation to the right option.")
    ]

    # Set the output parser with format instructions
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt_template = PromptTemplate(input_variables=['text', 'prompt', 'question_type'],
                                     partial_variables={"format_instructions": format_instructions},
                                     template='{prompt} about {question_type}: {text} '
                                              'Provide the answer as one capital letter. {format_instructions}.')

    retry_parser = RetryOutputParser.from_llm(parser=output_parser, llm=model)

    n_true, n_false, n_invalid = 0, 0, 0

    # Go through the questions in the dataframe
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
        text = row['text']
        prompt = row['prompt']
        question_type = row['question_type']

        print('------------------------------------------------------------')

        print(f'{prompt} about {question_type}: {text}')

        print('\nGolden answer: ', row['answer'])

        final_prompt = prompt_template.format(text=text,
                                              prompt=prompt,
                                              question_type=question_type,
                                              format_instructions=format_instructions)

        output = model(final_prompt)

        i, captured_dict = read_dict(output)
        print(i, captured_dict)

        if captured_dict:
            '''
            if i == 1:
                try:
                    dict = json.loads(captured_dict)
                except json.JSONDecodeError as e:
                    print(f"\nError decoding Dict with json: {captured_dict}")
                    n_invalid += 1
                    continue
            '''
            if i == 1 or i == 2:
                try:
                    dict = ast.literal_eval(captured_dict)
                except (ValueError, SyntaxError) as e:
                    print(f"\nError decoding Dict with text: {captured_dict}")
                    n_invalid += 1
                    continue
            else:
                raise NotImplementedError

            print(f"\nCorrectly captured json dictionary: {dict}")

            if dict['answer'] == row['answer']:
                n_true += 1
            else:
                n_false += 1
        else:
            print(f"\nNot captured json with text: {output}")
            n_invalid += 1

        print('------------------------------------------------------------')

        print(n_true, n_false, n_invalid)
