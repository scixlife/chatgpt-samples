# chatgpt clone with memory using openai and langchain
# no streaming 
# version = 1.0 
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.chains import ConversationChain

from langchain.schema.messages import (
    AIMessage,
    HumanMessage,
)

# Set up logging
import logging
LOGFORMAT = '%(asctime)s %(name)s-%(levelname)s: %(message)s'
LOGFILE = 'chatgpt.py.log'
try:
    logging.basicConfig(level=logging.INFO, 
                        filename=LOGFILE, 
                        format=LOGFORMAT, 
                    )
except OSError as error:
    logging.error("OS error %s", error)
    logging.info("Creating log file failed, logging to console instead.")
    logging.basicConfig(level=logging.INFO, 
                        # filename='app.log', 
                        format=LOGFORMAT, 
                    )

# initialize the objects 
llm = ChatOpenAI(temperature=0.7, max_tokens=500)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, llm=llm)

# generate a response to the user chat message
def chat_response(question: str):
    response = conversation.predict(input=question) 
    messages = conversation.memory.load_memory_variables({})
    inp_message = [HumanMessage(content=question)]
    out_message = [AIMessage(content=response)]
    inp_tokens = llm.get_num_tokens_from_messages(inp_message)
    out_tokens = llm.get_num_tokens_from_messages(out_message)
    all_tokens = llm.get_num_tokens_from_messages(messages['history'])
    
    logging.info("User: %s", question)
    logging.info("ChatGPT: %s", response)
    logging.info("%s input tokens", inp_tokens)
    logging.info("%s output tokens", out_tokens)
    logging.info("%s total tokens", inp_tokens+out_tokens)
    logging.info("%s tokens in this conversation altogether", all_tokens)

    return response

# chat with chatgpt
def chatgpt(): 
    while True:
        user_input = input("User: ")
        if user_input.lower() == "bye":
            print("ChatGPT: Goodbye! Have a great day!")
            break
        response = chat_response(user_input) # no streaming
        print("ChatGPT: ", response) # no streaming 

if __name__ == "__main__":
    chatgpt()
    memory.clear()