import json


def read_file(file_path_list):
    questions_list = []

    for file_path in file_path_list:
        with open(file_path, 'r', encoding='UTF-8') as file:
            question_dict = json.load(file)
            questions_list += question_dict["questions"]

    return questions_list


class BioASQ:
    def __init__(self, file_path_list):
        self.questions = read_file(file_path_list)

        self.question_type = list({question['type'] for question in self.questions})

    def get_type_questions(self, type):
        yesno_questions = []

        for question in self.questions:
            if question['type'] == type:
                yesno_questions.append(question)

        return yesno_questions

    def get_document_urls(self):
        document_urls = []

        for question in self.questions:
            document_urls += question['documents']

        return document_urls


if __name__ == '__main__':
    data = BioASQ(['dataset/Task11BGoldenEnriched/11B1_golden.json',
                   'dataset/Task11BGoldenEnriched/11B2_golden.json',
                   'dataset/Task11BGoldenEnriched/11B3_golden.json',
                   'dataset/Task11BGoldenEnriched/11B4_golden.json'])

    print(len(data.questions))

    print(len(data.get_type_questions('yesno')))

    print(len(data.get_document_urls()))
