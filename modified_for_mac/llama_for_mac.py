from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import(StreamingStdOutCallbackHandler)
import os
import sys
from time import time

# If your Macbook's CPU has x86 architecture, OpenBLAS seems best option for hardware acceleration, therefore use:
# `CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python` to install.

# This will run without separately installing and building `llama.cpp` or installing torch torchvision torchaudio.

if __name__ == '__main__':
    start = time()

    # Replace with the .gguf model file in your repository
    # Do not put your model file under the SNLP_CW repo unless you add it to .gitignore
    your_model_path = '../../llama_cpp/models/llama-2-7b-chat.Q2_K.gguf'

    if not os.path.exists(your_model_path):
        sys.exit('Your pretrained model (gguf) is not in the expected location.')

    else:
        question_topic = "biomedical/clinical"
        user_question = str(input("Please enter your question："))

        prompt_template = f"Please answer this question: \"{user_question}\", around the topic of \"{question_topic}\""

        prompt = PromptTemplate.from_template(prompt_template)
        final_prompt = prompt.format(topic=question_topic, question=user_question)
        CallbackManager = CallbackManager([StreamingStdOutCallbackHandler()])

        llm = LlamaCpp(
            model_path=your_model_path,
            n_ctx=6000,
            # n_gpu_layers=512, # commented out as there are no GPUs in our MacBooks.
            n_batch=30,
            callback_manager=CallbackManager,
            temperature=0.9,
            max_tokens=4095,
            n_parts=1,
            verbose=0
        )
        answer = llm(final_prompt)
        print(f'time to start of giving answer {time() - start} secs')
        print(answer)
        print(f'time to end of giving answer {time() - start} secs')


