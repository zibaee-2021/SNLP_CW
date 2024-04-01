from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_openai import ChatOpenAI

import os

os.environ["OPENAI_API_KEY"] = 'sk-xxx'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class Llama2:
    def __init__(self, model_path):
        self.model = LlamaCpp(
            model_path=model_path,
            streaming=False,
            n_ctx=6000,
            max_tokens=4095
        )

    def invoke(self, prompt):
        return self.model(prompt)


class GPT:
    def __init__(self, temp):
        self.model = ChatOpenAI(temperature=temp, openai_api_key=OPENAI_API_KEY)


if __name__ == '__main__':
    question_topic = "biomedical/clinical"
    user_question = str(input("Please enter your question："))

    prompt_template = f"Please answer this question: \"{user_question}\", around the topic of \"{question_topic}\""

    prompt = PromptTemplate.from_template(prompt_template)
    final_prompt = prompt.format(
        topic=question_topic,
        question=user_question
    )
