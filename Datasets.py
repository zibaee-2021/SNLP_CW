from itertools import islice
import json
import re

import aiohttp
import asyncio
from bs4 import BeautifulSoup

class Question:
    def __init__(self, question_body, answer, type, prompt, docs, refs):
        self.question_body = question_body
        self.answer = answer
        self.type = type
        self.prompt = prompt
        self.docs = docs
        self.refs = refs

    def get_golden_refs(self, num):
        golden_refs = ''
        for ref in islice(self.refs, num):
            golden_refs += ref['text'] + '; '
        return golden_refs


class BioASQ:
    def __init__(self, file_path_list):
        self.questions = []

        for file_path in file_path_list:
            with open(file_path, 'r', encoding='UTF-8') as file:
                question_dict = json.load(file)
                for question in question_dict["questions"]:
                    answer = question['exact_answer'] if 'exact_answer' in question.keys() else 'ideal_answer'

                    self.questions.append(Question(question_body=question['body'],
                                                   answer=answer,
                                                   type=question['type'],
                                                   prompt=self.get_prompt(question['type']),
                                                   docs=question['documents'],
                                                   refs=question['snippets']))

        self.question_type = list({question.type for question in self.questions})

    @staticmethod
    def get_prompt(q_type):
        if q_type == 'yesno':
            return 'Please answer yes or no that answers the question.'
        else:
            return ''

    def get_type_questions(self, q_type):
        type_questions = []

        for question in self.questions:
            if question.type == q_type:
                type_questions.append(question)

        return type_questions

    def get_document_urls(self, q_type):
        document_urls = []

        for question in self.questions:
            if question.type == q_type:
                for doc in question.docs:
                    if doc not in document_urls:
                        document_urls.append(doc)

        return document_urls


class QALM_mcq:
    def __init__(self, file_path_list):
        self.questions = []

        for file_path in file_path_list:
            with open(file_path, 'r', encoding='UTF-8') as file:
                for line in file:
                    question = json.loads(line)

                    self.questions.append(Question(question_body=question['text'],
                                                   answer=question['answer'],
                                                   type='mcq',
                                                   prompt=question['prompt'],
                                                   docs=[],
                                                   refs=[]))

async def extract_url(session, url):
    try:
        async with session.get(url) as response:
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')

            abstract_section = soup.find('div', class_='abstract-content')
            abstract = abstract_section.get_text(
                separator='\n').strip() if abstract_section else "Abstract not available"
            clean_abstract = re.sub(r'\s+', ' ', abstract)

            title_tag = soup.find('title')
            title = title_tag.text if title_tag else "Title not found"

            return title, clean_abstract
    except aiohttp.ClientConnectorError:
        return "Connection Error", ""
    except aiohttp.ServerDisconnectedError:
        return "Server Disconnected Error", ""

async def extract_data(yesno_urls):
    docs = []
    async with aiohttp.ClientSession() as session:
        for url in yesno_urls:
            title, abstract = await extract_url(session, url)
            docs.append({'url': url,
                         'title': title,
                         'abstract': abstract})

    return docs

async def main(yesno_urls):
    retrieved_docs = await extract_data(yesno_urls)

    print(len(retrieved_docs))

    with open('refs/BioASQ_11B_test_yesno.json', 'w', encoding='utf-8') as output_file:
        json.dump(retrieved_docs, output_file, indent=4)


if __name__ == '__main__':
    # data = QALM_mcq(['dataset/QALM/test/mcq/bioasq_mcq_test.jsonl'])
    # print('Total QALM BioASQ mcq Questions:　', len(data.questions))

    data = BioASQ(['dataset/Task11BGoldenEnriched/11B1_golden.json',
                   'dataset/Task11BGoldenEnriched/11B2_golden.json',
                   'dataset/Task11BGoldenEnriched/11B3_golden.json',
                   'dataset/Task11BGoldenEnriched/11B4_golden.json'])

    print('Total BioASQ 11B Questions:　', len(data.questions))

    print('Total BioASQ 11B yesno Questions:　', len(data.get_type_questions('yesno')))

    print('Total BioASQ 11B yesno urls:　', len(data.get_document_urls('yesno')))

    print(data.questions[1].get_golden_refs(num=5))

    yesno_urls = data.get_document_urls('yesno')

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(yesno_urls))
