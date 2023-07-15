import openai
import os
import logging
# import time

# Set up logging
FORMAT = '%(asctime)s %(name)s-%(levelname)s: %(message)s'
try:
    logging.basicConfig(level=logging.INFO, 
                        filename='app.log', 
                        format=FORMAT, 
                    )
except OSError as error:
    logging.error("OS error %s", error)
    logging.info("Creating log file failed, logging to console instead.")
    logging.basicConfig(level=logging.INFO, 
                        # filename='app.log', 
                        format=FORMAT, 
                    )

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create a function to generate a chat response from GPT-3.5
def chat_response(user_prompt):
    model = "gpt-3.5-turbo"
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt} # "Hello!"
        ],
        temperature=0.7,
    )

    # print(completion.choices[0].message)
    logging.info("Chat completion: %s", completion.id)
    logging.info("Chat model: %s", model)
    logging.info("Input tokens: %s", completion.usage.prompt_tokens)
    logging.info("Output tokens: %s", completion.usage.completion_tokens)
    logging.info("Total tokens: %s", completion.usage.total_tokens)

    return completion.choices[0].message.content

def chat_response_stream(user_prompt):
    model = "gpt-3.5-turbo"
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt} # "Hello!"
        ],
        temperature=0.7,
        stream=True, # This enables streaming 
    )

    return completion #.choices[0].message.content

# Chat loop
print("ChatGPT: Hello! How can I assist you today?")

while True:
    user_prompt = input("User: ")
    if user_prompt.lower() == "bye":
        print("ChatGPT: Goodbye! Have a great day!")
        break
    # response = chat_response(user_prompt) # no streaming
    # print("ChatGPT: ", response) # no streaming 
    
    response = chat_response_stream(user_prompt)
    print("ChatGPT: ", end="")
    for chunk in response:
        chunk_message = chunk['choices'][0]['delta']
        if(chunk_message):
            print(chunk_message.content, end="")
        else: # last empty chunk
            logging.info("Chat completion: %s", chunk.id)
            logging.info("Chat model: %s", chunk.model)
        # time.sleep(0.1)
    print("\n")

# cleaning up
logging.shutdown()