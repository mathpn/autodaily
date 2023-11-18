import os

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# from gpt4all import GPT4All
#
# model = GPT4All(model_name="orca-mini-3b-gguf2-q4_0.gguf")
# with model.chat_session():
#     while True:
#         try:
#             user_input = input("enter your question:\n")
#             response1 = model.generate(prompt=user_input.strip(), temp=0)
#             print(response1)
#         except KeyboardInterrupt:
#             print(model.current_chat_session)
#             exit()
#
local_path = f"{os.environ["HOME"]}/.cache/gpt4all/orca-mini-3b-gguf2-q4_0.gguf"
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

template = """Question: {question}

Answer: Let's think step by step.

"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = GPT4All(model=local_path, callback_manager=callback_manager, verbose=True)
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = input("Enter your question: ")

llm_chain.run(question)
