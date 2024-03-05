import math

from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.output_parsers import OutputFixingParser
from langchain.chains import LLMChain
from langchain_core import exceptions

from tqdm import tqdm
import argparse
import csv

from RAG import RAG
from LLM import Llama2, GPT
from Datasets import Question, QALM_mcq, BioASQ


class PromptLibrary:
    def __init__(self, prompt_template_csv_file):
        self.prompt_template_dict = []

        with open(prompt_template_csv_file, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for prompt_dict in reader:
                self.prompt_template_dict.append(prompt_dict)

    def get_prompt_template(self, dataset, question_type, with_rag):
        for prompt_dict in self.prompt_template_dict:
            if prompt_dict['Dataset'] == dataset \
                    and prompt_dict['Question_Type'] == question_type \
                    and prompt_dict['RAG'] == str(with_rag):
                return PromptTemplate.from_template(prompt_dict['prompt_template'])

        raise ModuleNotFoundError


def format_docs(docs):
    return "\n".join(doc.metadata['title'] for doc in docs)


def get_questions(dataset, question_type):
    if dataset == 'QALM_mcq':
        questions = QALM_mcq(['dataset/QALM/test/mcq/bioasq_mcq_test.jsonl']).questions
    elif dataset == 'BioASQ':
        data = BioASQ(['dataset/Task11BGoldenEnriched/11B1_golden.json',
                       'dataset/Task11BGoldenEnriched/11B2_golden.json',
                       'dataset/Task11BGoldenEnriched/11B3_golden.json',
                       'dataset/Task11BGoldenEnriched/11B4_golden.json'])

        if question_type == 'yesno':
            questions = data.get_type_questions('yesno')
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return questions


def get_llm_model(llm_model):
    if llm_model == 'Llama2':
        model = Llama2('../llama.cpp/models/llama-2-13b-chat.Q4_K_M.gguf').model
    elif llm_model == 'OpenAI':
        model = GPT(temp=0).model
    else:
        raise NotImplementedError

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Biomedical Question Answering')

    parser.add_argument('-d', '--dataset', default='BioASQ', choices=['QALM_mcq', 'BioASQ'], type=str,
                        help='Biomedical Question-Answering Dataset')
    parser.add_argument('-q', '--question_type', default='yesno', choices=['mcq', 'yesno'], type=str,
                        help='Specified Question Type in the QA Dataset')
    parser.add_argument('-m', '--llm_model', default='OpenAI', choices=['OpenAI', 'Llama2'], type=str,
                        help='LLM Model utilized to process context and answer questions')
    parser.add_argument('--rag', default=True, type=bool, help='If RAG Pipeline is activated')
    parser.add_argument('--load_db', default=True, type=bool, help='If Load FAISS database')

    args = parser.parse_args()
    print(args)

    # ------------------------------------------------------------------------------------------------------------------

    # Initialize the Questions
    questions = get_questions(dataset=args.dataset, question_type=args.question_type)

    # Initialize the Model
    model = get_llm_model(llm_model=args.llm_model)

    # Extract the Prompt Template
    prompt_lib = PromptLibrary('prompt_template.csv')
    prompt_template = prompt_lib.get_prompt_template(dataset=args.dataset,
                                                     question_type=args.question_type,
                                                     with_rag=args.rag)

    if args.rag:
        if load:
            rag_database = load_faiss_database(documents='BioASQ_11B_test',
                                               embedding_model='OpenAI')
        else:
            rag_database = save_faiss_database(documents='BioASQ_11B_test',
                                               document_file_path='refs/retrieved_BioASQ_test.json',
                                               embedding_model='OpenAI')

    # ------------------------------------------------------------------------------------------------------------------

    # Set the schema for structured output
    response_schemas = [
        ResponseSchema(name="answer", description="The correct option answer."),
        ResponseSchema(name="explanation", description="The explanation to the right option.")
    ]

    # Set the output parser with format instructions
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    print(format_instructions)

    # ------------------------------------------------------------------------------------------------------------------

    n_true, n_false, n_invalid = 0, 0, 0

    # Go through the questions in the dataframe
    for question in tqdm(questions):
        print('-------------------------------------------------------------------------------------------------------')

        print(f'{question.prompt} {question.question_body}')

        print('\nGolden answer: ', question.answer)

        print('\nGolden refs: ', question.get_golden_refs(num=1))

        chain = LLMChain(llm=model, prompt=prompt_template, output_parser=output_parser)

        output = chain.invoke(
            {"question": question.question_body,
             "prompt": question.prompt,
             # Golden References
             # "docs": question.get_golden_refs(num=10000),
             # Retrieved References
             "docs":
             "format_instructions": format_instructions
             }
        )['text']

        print('\n\nLLM output: ', output)

        if args.question_type == 'mcq':
            answer = output['answer'][0]
        elif args.question_type == 'yesno':
            answer = output['answer'].lower()
        else:
            raise NotImplementedError

        if answer == question.answer:
            print(f"\nCaptured True Answer: {answer}")
            n_true += 1
        else:
            print(f"\nCaptured False Answer: {answer}")
            n_false += 1

        # Print a count on true and false answers
        print('True: ', n_true, 'False: ', n_false, 'Invalid: ', n_invalid)

        print('-------------------------------------------------------------------------------------------------------')

    # Calculate the final evaluation statistics and print the result
    n_total = n_true + n_false + n_invalid

    print(args)
    print(prompt_template)
    print('True Accuracy: %.3f; False: %.3f; Invalid: %.3f.' %
          (n_true / n_total, n_false / n_total, n_invalid / n_total))
