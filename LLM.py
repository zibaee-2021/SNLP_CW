# Download all necessary packages
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_openai import ChatOpenAI
import os

# OpenAI key
# os.environ["OPENAI_API_KEY"] = 'sk-xxx'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class Llama2:
    def __init__(self, model_path):
        # Initialize Llama2 model with specific model path
        self.model = LlamaCpp(
            model_path=model_path,
            streaming=False,
            n_ctx=6000,
            max_tokens=4095
        )

    # Invoke the Llama2 model with the given prompt
    def invoke(self, prompt):
        return self.model(prompt)


class GPT:
    def __init__(self, temp):
        # Initialize GPT model with the provided temperature and OpenAI API key
        self.model = ChatOpenAI(temperature=temp, openai_api_key=OPENAI_API_KEY)


if __name__ == '__main__':
    # Define question topic
    question_topic = "biomedical"
    # Get user input for the question
    user_question = str(input("Please enter your question："))
    # Create a prompt template with the user question and topic
    prompt_template = f"Please answer this question: \"{user_question}\", around the topic of \"{question_topic}\""
    # Format the prompt template with topic and question
    prompt = PromptTemplate.from_template(prompt_template)
    final_prompt = prompt.format(
        topic=question_topic,
        question=user_question
    )
