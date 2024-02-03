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

import torch
import pandas as pd
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

def find_json(input_string):
    # Define a regular expression to capture content between curly braces
    pattern = re.compile(r'```\{ \}```')

    # Find the first match in the input string
    match = pattern.search(input_string)

    # Return the captured content or None if no match is found
    return match.group(1) if match else None

if __name__ == '__main__':
    # Load the llama model
    # model = load_llama("../llama.cpp/models/llama-2-7b-chat.Q4_K_M.gguf")
    model = load_llama("../llama.cpp/models/llama-2-13b-chat.Q4_K_M.gguf")
    # model = ChatOpenAI(temperature=0)

    # Assuming 'data.json' is your JSON file
    json_file_path = 'dataset/QALM/train/mcq/bioasq_mcq_train.jsonl'

    # Open the file and load its content into a dictionary
    df = pd.read_json(json_file_path, lines=True)

    print('Options: ', df['answer'].unique())

    print('Cuda is available: ', torch.cuda.is_available())

    # Set the schema for structured output
    response_schemas = [
        ResponseSchema(name="option", description="The right option that answers the question."),
        ResponseSchema(name="explanation", description="The explanation to the right option.")
    ]

    # Set the output parser with format instructions
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt_template = PromptTemplate(input_variables=['text', 'prompt', 'question_type'],
                                     partial_variables={"format_instructions": format_instructions},
                                     template='{text} {prompt} about {question_type}. \n{format_instructions}.')

    retry_parser = RetryOutputParser.from_llm(parser=output_parser, llm=model)

    for index, row in df.iterrows():
        text = row['text']
        prompt = row['prompt']
        question_type = row['question_type']

        print('------------------------------------------------------------')

        print(f'{text} {prompt} under the topic of {question_type}.')

        chain = prompt_template | model | output_parser

        main_chain = RunnableParallel(
            completion=chain, prompt_value=prompt_template
        ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

        output = main_chain.invoke({'text': text, 'prompt': prompt, 'question_type': question_type})

        #final_prompt = prompt_template.format(text=text,
        #                                      prompt=prompt,
        #                                      question_type=question_type,
        #                                      format_instructions=format_instructions)
        #output = model(final_prompt)

        print('\nLLM: ', output)

        print('\nGolden answer: ', row['answer'])

        print('------------------------------------------------------------')
