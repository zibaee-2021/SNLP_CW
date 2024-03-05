from itertools import islice
import json


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

    def get_document_urls(self):
        document_urls = []

        for question in self.questions:
            document_urls += question.docs

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


if __name__ == '__main__':
    data = BioASQ(['dataset/Task11BGoldenEnriched/11B1_golden.json',
                   'dataset/Task11BGoldenEnriched/11B2_golden.json',
                   'dataset/Task11BGoldenEnriched/11B3_golden.json',
                   'dataset/Task11BGoldenEnriched/11B4_golden.json'])

    print(len(data.questions))

    print(len(data.get_type_questions('yesno')))

    print(len(data.get_document_urls()))

    print(data.questions[1].get_golden_refs(num=5))

    data = QALM_mcq(['dataset/QALM/test/mcq/bioasq_mcq_test.jsonl'])

    print(len(data.questions))
