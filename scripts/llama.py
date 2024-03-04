from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import(
StreamingStdOutCallbackHandler
)

# Replace with the .gguf model file in your repository
# Don't put your model file under the SNLP_CW repo unless you add it to .gitignore
your_model_path = "../../llama.cpp/models/llama-2-7b-chat.Q4_K_M.gguf"

question_topic = "biomedical/clinical"
user_question = str(input("Please enter your question："))

prompt_template = f"Please answer this question: \"{user_question}\", around the topic of \"{question_topic}\""

prompt = PromptTemplate.from_template(prompt_template)
final_prompt = prompt.format(
    topic=question_topic,
    question=user_question
)
CallbackManager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path=your_model_path,
    n_ctx=6000,
    n_gpu_layers=512,
    n_batch=30,
    callback_manager=CallbackManager,
    temperature=0.9,
    max_tokens=4095,
    n_parts=1,
    verbose=0
)

print(llm(final_prompt))
