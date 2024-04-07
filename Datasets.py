# Download all necessary packages
from itertools import islice
import json
import re

import aiohttp
import asyncio
from bs4 import BeautifulSoup

# Define the Question class to represent BioASQ questions
class Question:
    def __init__(self, question_body, answer, type, prompt, docs, refs):
        # Initializes a Question object
        self.question_body = question_body # question body as string
        self.answer = answer # question answer as string
        self.type = type # question type as string
        self.prompt = prompt # question prompt as string
        self.docs = docs # Document URLs associated with the question as list
        self.refs = refs # Snippets (references) associated with the question as list

    # Method to get concatenated golden reference texts
    def get_golden_refs(self, num):
        golden_refs = '' # Initialize empty string to store concatenated golden reference
        # Iterate golden references up to the specified number
        for ref in islice(self.refs, num):
            golden_refs += ref['text'] + '; ' # Concatenate each golden reference text followed by a semicolon
        return golden_refs

# Define the BioASQ class to handle BioASQ dataset
class BioASQ:
    def __init__(self, file_path_list):
        # Initializes a BioASQ object
        self.questions = []
        # Parse BioASQ JSON files and extract questions
        for file_path in file_path_list:
            with open(file_path, 'r', encoding='UTF-8') as file:
                question_dict = json.load(file) # Load JSON data from the file
                # Iterate through each question
                for question in question_dict["questions"]:
                    # Set answer equal to exact answer, if not exist, use ideal answer
                    answer = question['exact_answer'] if 'exact_answer' in question.keys() else 'ideal_answer'
                    # Create a Question object and append it to the list of questions
                    self.questions.append(Question(question_body=question['body'],
                                                   answer=answer,
                                                   type=question['type'],
                                                   prompt=self.get_prompt(question['type']),
                                                   docs=question['documents'],
                                                   refs=question['snippets']))
        # Extract unique question types
        self.question_type = list({question.type for question in self.questions})

    # Method to generate prompt based on question type
    @staticmethod
    def get_prompt(q_type):
        # If question type is Yes or No
        if q_type == 'yesno':
            return 'Please answer yes or no that answers the question.'
        else:
            return ''

    # Method to filter questions based on type
    def get_type_questions(self, q_type):
        type_questions = []
        for question in self.questions: # For each question
            if question.type in q_type: # Get its question type
                type_questions.append(question) # Get and append question type into list
        return type_questions

    # Method to extract unique document URLs based on question type
    def get_document_urls(self, q_type):
        document_urls = []
        for question in self.questions: # For each question
            if question.type in q_type: # With give question type
                for doc in question.docs: # Get its matched documents
                    if doc not in document_urls:
                        document_urls.append(doc) # Get and append document URLs into list
        return document_urls

# Define the QALM_mcq class to handle QALM multiple choice questions
class QALM_mcq:
    def __init__(self, file_path_list):
        self.questions = []
        # Parse QALM JSON files and extract multiple choice questions
        for file_path in file_path_list:
            with open(file_path, 'r', encoding='UTF-8') as file:
                for line in file:
                    question = json.loads(line)
                    # Create a Question object and append it to the list of questions
                    self.questions.append(Question(question_body=question['text'],
                                                   answer=question['answer'],
                                                   type='mcq',
                                                   prompt=question['prompt'],
                                                   docs=[],
                                                   refs=[]))

# Async function to extract URL content asynchronously
async def extract_url(session, url):
    try:
        async with session.get(url) as response:
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            # Extract abstract content from the HTML
            abstract_section = soup.find('div', class_='abstract-content')
            abstract = abstract_section.get_text(
                separator='\n').strip() if abstract_section else "Abstract not available"
            clean_abstract = re.sub(r'\s+', ' ', abstract)
            # Extract title from HTML
            title_tag = soup.find('title')
            title = title_tag.text if title_tag else "Title not found"
            return title, clean_abstract
    except aiohttp.ClientConnectorError:
        return "Connection Error", ""
    except aiohttp.ServerDisconnectedError:
        return "Server Disconnected Error", ""

# Async function to extract data from multiple URLs concurrently
async def extract_data(yesno_urls):
    docs = []
    async with aiohttp.ClientSession() as session:
        for url in yesno_urls:
            title, abstract = await extract_url(session, url)
            docs.append({'url': url,
                         'title': title,
                         'abstract': abstract})
    return docs

# Main async function to orchestrate URL extraction process
async def main(yesno_urls, file_path):
    retrieved_docs = await extract_data(yesno_urls)
    # Check the number of retrieved documents
    print(len(retrieved_docs))
    # Save retrieved documents to a JSON file
    with open(file_path, 'w', encoding='utf-8') as output_file:
        json.dump(retrieved_docs, output_file, indent=4)

if __name__ == '__main__':
    # Initialize BioASQ dataset
    data = BioASQ(['dataset/BioASQ-training11b/training11b.json'])
    # Print available question types
    print(data.question_type)

    # Get document URLs for specific question types
    all_urls = data.get_document_urls(['factoid', 'list', 'yesno', 'summary'])
    # Print specific URL for checking
    print(all_urls[10000])

    # Run the URL extraction process asynchronously
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(all_urls[10000:20000], file_path='refs/BioASQ_11B_train_yesno_2.json'))
