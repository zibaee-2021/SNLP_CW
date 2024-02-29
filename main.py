from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.output_parsers import OutputFixingParser
from langchain_core import exceptions

from operator import itemgetter
from tqdm import tqdm

import pandas as pd

from RAG import RAG
from LLM import Llama, GPT


def format_docs(docs):
    return "\n".join(doc.metadata['title'] for doc in docs)


if __name__ == '__main__':
    # Assuming 'data.json' is your JSON file
    json_file_path = 'dataset/QALM/test/mcq/bioasq_mcq_test.jsonl'

    # Open the file and load its content into a dictionary
    df = pd.read_json(json_file_path, lines=True)

    # Print all option choice in the MCQ dataset
    print('Loaded all questions. Options: ', df['answer'].unique())

    # ------------------------------------------------------------------------------------------------------------------

    rag = RAG('refs/pubmed_2020-2023.json')

    retriever = rag.get_retriever(k=2)

    # ------------------------------------------------------------------------------------------------------------------

    llm = GPT(temp=0).model

    # Set the schema for structured output
    response_schemas = [
        ResponseSchema(name="answer", description="The correct option answer."),
        ResponseSchema(name="explanation", description="The explanation to the right option.")
    ]

    # Set the output parser with format instructions
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    if RAG:
        prompt_template = PromptTemplate.from_template("""{format_instructions}. {prompt} about {question_type} 
                                                    based only on the following context: {docs}. {question}""")
    else:
        prompt_template = PromptTemplate.from_template("""{format_instructions}. {prompt} about {question_type}. 
                                                    {question}""")

    n_true, n_false, n_invalid = 0, 0, 0

    # ------------------------------------------------------------------------------------------------------------------

    # Go through the questions in the dataframe
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
        question = row['text']
        prompt = row['prompt']
        question_type = row['question_type']
        few_shot_prompt = row['few_shot_prompt']

        print('------------------------------------------------------------')

        print(f'{prompt} about {question_type}: {question}')

        print('\nGolden answer: ', row['answer'])

        rag_doc = retriever.invoke(question)
        print('Retrieved Docs: ', format_docs(rag_doc))

        if RAG:
            # Chain the final prompt to the LLM model
            input_dict = {"docs": itemgetter("question") | retriever,
                          "question": itemgetter("question"),
                          "prompt": itemgetter("prompt"),
                          "question_type": itemgetter("question_type"),
                          "format_instructions": itemgetter("format_instructions")}
        else:
            input_dict = {"question": itemgetter("question"),
                          "prompt": itemgetter("prompt"),
                          "question_type": itemgetter("question_type"),
                          "format_instructions": itemgetter("format_instructions")}

        chain = (input_dict | prompt_template | llm)

        output = chain.invoke({"question": question,
                               "prompt": prompt,
                               "question_type": question_type,
                               "format_instructions": format_instructions}).content

        print('------------------------------------------------------------')

        # Use output parser to read the output json dictionary
        try:
            answer = output_parser.invoke(output)
        except exceptions.OutputParserException as e:
            new_parser = OutputFixingParser.from_llm(parser=output_parser, llm=llm)

            try:
                answer = new_parser.parse(output)
            except exceptions.OutputParserException as e:
                answer = {'answer': 'EXCEPTION'}

        # Compare the LLM Answer to the Normal One
        if answer['answer'] != 'EXCEPTION':
            # True Answer
            if answer['answer'][0] == row['answer']:
                print(f"\nCaptured True Answer: {answer}")
                n_true += 1
            # False Answer
            else:
                print(f"\nCaptured False Answer: {answer}")
                n_false += 1
        else:
            # Invalid / Not Captured
            print(f"\nInvalid / Not Captured Answer: {answer}")
            n_invalid += 1

        # Print a count on true and false answers
        print('True: ', n_true, 'False: ', n_false, 'Invalid: ', n_invalid)

        print('------------------------------------------------------------')

    # Calculate the final evaluation statistics and print the result
    n_total = n_true + n_false + n_invalid

    print('True Accuracy: %.3f; False: %.3f; Invalid: %.3f.' %
          (n_true / n_total, n_false / n_total, n_invalid / n_total))
