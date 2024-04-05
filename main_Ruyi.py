# Download all necessary packages
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chains import LLMChain
from tqdm import tqdm
import argparse
import csv
import random
from LLM import Llama2, GPT
from Datasets import QALM_mcq, BioASQ
import RAG

# Flag to use Golden, Random, RAG, and random sort
golden_use = True
random_use = False
rag_use = True
random_sort = False

# Class to manage prompt templates
class PromptLibrary:
    def __init__(self, prompt_template_csv_file):
        self.prompt_template_dict = [] # Create list to store prompt templates
        with open(prompt_template_csv_file, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for prompt_dict in reader:
                self.prompt_template_dict.append(prompt_dict) # If in reader, add to dict

    def get_prompt_template(self, dataset, question_type):
        # Iterate every prompt templates
        for prompt_dict in self.prompt_template_dict:
            # For every prompt matched with data and question type
            if prompt_dict['Dataset'] == dataset and prompt_dict['Question_Type'] == question_type:
                # Return its content of prompt
                return PromptTemplate.from_template(prompt_dict['prompt_template'])
        raise ModuleNotFoundError


# Function to format documents
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs) # Change list of document into a single string

# Function to get questions from dataset
def get_questions(dataset, question_type):
    # Get question from data set with set question type
    if dataset == 'QALM_mcq':
        # Get multiple choice question in QALM data
        questions = QALM_mcq(['dataset/QALM/test/mcq/bioasq_mcq_test.jsonl']).questions
    elif dataset == 'BioASQ':
        # Load data set BioASQ
        data = BioASQ(['dataset/Task11BGoldenEnriched/11B1_golden.json',
                       'dataset/Task11BGoldenEnriched/11B2_golden.json',
                       'dataset/Task11BGoldenEnriched/11B3_golden.json',
                       'dataset/Task11BGoldenEnriched/11B4_golden.json'])
        # Get Yes or No question from the BioASQ data
        if question_type == 'yesno':
            questions = data.get_type_questions('yesno')
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return questions

# Create a function to get the LLM model
def get_llm_model(llm_model):
    if llm_model == 'Llama2':
        # Load llama2 model
        model = Llama2('../llama.cpp/models/llama-2-13b-chat.Q4_K_M.gguf').model
    elif llm_model == 'OpenAI':
        # Load OpenAI model
        model = GPT(temp=0).model
    else:
        raise NotImplementedError

    return model

if __name__ == '__main__':
    # Create an ArgumentParser object to manage command-line arguments
    parser = argparse.ArgumentParser(description='Biomedical Question Answering')
    # Argument for specifying the dataset, with default value 'BioASQ' and choices between 'QALM_mcq' and 'BioASQ'
    parser.add_argument('-d', '--dataset', default='BioASQ', choices=['QALM_mcq', 'BioASQ'], type=str,
                        help='Biomedical Question-Answering Dataset')
    # Argument for specifying the question type, with default value 'yesno' and choices between 'mcq' and 'yesno'
    parser.add_argument('-q', '--question_type', default='yesno', choices=['mcq', 'yesno'], type=str,
                        help='Specified Question Type in the QA Dataset')
    # Argument for specifying the LLM model, with default value 'OpenAI' and choices between 'OpenAI' and 'Llama2'
    parser.add_argument('-m', '--llm_model', default='OpenAI', choices=['OpenAI', 'Llama2'], type=str,
                        help='LLM Model utilized to process context and answer questions')
    # Argument for specifying whether to load the FAISS database, with default value True
    parser.add_argument('--load_db', default=True, type=bool, help='If Load FAISS database')
    # Parse the command-line arguments provided by the user
    args = parser.parse_args()
    print(args)

    # ------------------------------------------------------------------------------------------------------------------

    # Initialize the Questions
    questions = get_questions(dataset=args.dataset, question_type=args.question_type)
    # Initialize the Model
    model = get_llm_model(llm_model=args.llm_model)
    # Extract the Prompt Template
    prompt_lib = PromptLibrary('prompt_template.csv')
    prompt_template = prompt_lib.get_prompt_template(dataset=args.dataset, question_type=args.question_type)

    # When RAG is used
    if rag_use:
        if args.load_db:
            # Import the constructed database
            # Golden_679 case
            rag_database = RAG.load_faiss_database(documents='BioASQ_11B_test',
                                                   embedding_model='OpenAI')

            # MedRAG_133000 case
            # rag_database = RAG.load_faiss_database(documents='MedRAG_133000', embedding_model='OpenAI')

            # Merged_133679 case
            # rag_database = RAG.load_faiss_database(documents='BioASQ_11B_test', embedding_model='OpenAI')
            # rag_database_2 = RAG.load_faiss_database(documents='MedRAG_133000', embedding_model='OpenAI')
            # rag_database.merge_from(rag_database_2)

        else:
            # Create and save a new FAISS database for RAG using specified documents and embedding model
            # Golden_679 case
            rag_database = RAG.save_faiss_database(documents='BioASQ_11B_test',
                                                   file_path_list=['refs/BioASQ_11B_train_yesno_1.json'],
                                                   embedding_model='OpenAI')

            # MedRAG_133000 case
            # rag_database = RAG.save_faiss_database(documents='MedRAG_133000',
                                                    # file_path_list=['refs/BioASQ_11B_train_yesno_1.json'],
                                                    # embedding_model='OpenAI')

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
        #import time
        #time_between_calls = 10 #seconds
        #time.sleep(time_between_calls)
        print('-------------------------------------------------------------------------------------------------------')

        print(f'{question.prompt} {question.question_body}')

        print('\nGolden answer: ', question.answer)

        # Initialize docs list
        docs = []
        # Values for G : R : K, here we use G(4) : R(4) : K(4) as example
        G, R, K = 4, 4, 4

        # If Golden is used
        if golden_use:
            # Retrieved golden answer documents with number equal to G
            golden_docs = question.get_golden_refs(num=G) if not isinstance(docs, str) else [docs]
            # Split each chunks
            golden_paragraphs = golden_docs.split("\n")
            # print("Golden:", golden_docs)
            if isinstance(docs, str):
                docs = [docs]
            # Store it into one dimensional list
            golden_list = list(filter(lambda x: x.strip(), golden_paragraphs))
            # print("Golden List:", golden_list)
            # Add it into docs
            docs.append(golden_list)
            # print("Docs with golden", docs)

        # If Random is used
        if random_use:
            # Load document obtained by random_retrieve.py
            output_file = "abstract_chunks.txt"
            with open(output_file, 'r', encoding='utf-8') as f:
                abstract_chunks = f.readlines()
            # Load all chunks
            abstract_chunks = [chunk.strip() for chunk in abstract_chunks]
            # Retrieved random documents with the setting value R
            if len(abstract_chunks) > R:
                random_chunks = random.sample(abstract_chunks, R)
                # print("Random chunks:", random_chunks)
                # Split each chunks
                random_docs = "\n".join(random_chunks)
                random_paragraphs = random_docs.split("\n")
                if isinstance(docs, str):
                    docs = [docs]
                # Store it into one dimensional list
                random_list = list(filter(lambda x: x.strip(), random_paragraphs))
                # print("Random List:", random_list)
                # Add it into docs
                docs.append(random_list)
                #print("Random:", random_docs)
            # print("Docs with Random", docs)

        # If RAG is used
        if rag_use:
            # Retrieved documents with number equal to K
            rag_docs = format_docs(rag_database.similarity_search(question.question_body, k=K))
            # Split each chunks
            rag_paragraphs = rag_docs.split("\n")
            # print("RAG:", rag_docs)
            if isinstance(docs, str):
                docs = [docs]
            # Store it into one dimensional list
            rag_list = list(filter(lambda x: x.strip(), rag_paragraphs))
            # print("RAG List:", rag_list)
            # Add it into docs
            docs.append(rag_list)
            # print("Docs with RAG", docs)

        print("Docs before random sort:", docs)

        # Change docs from two dimension list into one dimension
        docs = [doc for sublist in docs for doc in sublist]

        # If Random sort needed
        if random_sort:
            # Shuffle docs by random order
            random.shuffle(docs)
            print("Docs after random sort:", docs)

        # If docs is empty
        if docs is None:
            raise ValueError("golden_use, random_use, or rag_use are all set to False.")

        # Create an instance of LLMChain with specified LLM model, prompt template, and output parser
        chain = LLMChain(llm=model, prompt=prompt_template, output_parser=output_parser)

        # Invoke the LLMChain to generate output
        # Using input as question, question prompt, retrieved docs and format instructions for output
        output = chain.invoke(
            {"question": question.question_body,
             "prompt": question.prompt,
             "docs": docs,
             "format_instructions": format_instructions
             }
        )['text']
        print(docs) # Retrieve the text content from the output dictionary

        print('\n\nLLM output: ', output)

        # Determine answer based on the question type
        # If multiple choice question
        if args.question_type == 'mcq':
            # Store the first position output from LLM as the answer
            answer = output['answer'][0]
        # If Yes or No question
        elif args.question_type == 'yesno':
            # Store LLM output as string in lower format as the answer
            answer = output['answer'].lower()
        else:
            raise NotImplementedError

        # Check if the captured answer matches the golden answer
        if answer == question.answer:
            print(f"\nCaptured True Answer: {answer}")
            n_true += 1  # If matches, add one to true count
        else:
            print(f"\nCaptured False Answer: {answer}")
            n_false += 1  # If not match, add one to false count

        # Print a count on true and false answers
        print('True: ', n_true, 'False: ', n_false, 'Invalid: ', n_invalid)

        print('-------------------------------------------------------------------------------------------------------')

        # Calculate the final evaluation statistics and print the result
    n_total = n_true + n_false + n_invalid

    print(args)
    print(prompt_template)
    print('True Accuracy: %.3f; False: %.3f; Invalid: %.3f.' %
          (n_true / n_total, n_false / n_total, n_invalid / n_total))