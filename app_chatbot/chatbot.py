## streamlit run chatbot.py --server.port 6006 --server.maxUploadSize 6

import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint
from typing import Dict
import json
from io import StringIO, BytesIO
from random import randint
from transformers import AutoTokenizer
from PIL import Image
import boto3
import numpy as np
import json
import os
import base64

client = boto3.client('runtime.sagemaker')
def query_endpoint_with_json_payload(encoded_json, endpoint_name):
    response = client.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json', Body=encoded_json)
    return response

def parse_response(query_response):
    response_dict = json.loads(query_response['Body'].read())
    return response_dict['generated_images'], response_dict['prompt']

st.set_page_config(page_title="Document Analysis", page_icon=":robot:")


Falcon_endpoint_name = os.getenv("falcon_ep_name", default="falcon-40b-instruct-12xl")
whisper_endpoint_name = os.getenv('wp_ep_name', default="wisper-large-v2")

endpoint_names = {
    "NLP":Falcon_endpoint_name,
    "Audio":whisper_endpoint_name
}


class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"
    len_prompt = 0

    def transform_input(self, prompt: str, model_kwargs: Dict={}) -> bytes:
        self.len_prompt = len(prompt)
        input_str = json.dumps({"inputs": prompt, "parameters":{"max_new_tokens": st.session_state.max_token, "temperature":st.session_state.temperature, "seed":st.session_state.seed, "stop": ["Human:"], "num_beams":1}})
        print(input_str)
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> str:
        response_json = output.read()
        res = json.loads(response_json)
        print(res)
        ans = res[0]['generated_text'][self.len_prompt:]
        ans = ans[:ans.rfind("Human")].strip()
        
        return ans 


    
content_handler = ContentHandler()


@st.cache_resource
def load_chain(endpoint_name: str=Falcon_endpoint_name):

    llm = SagemakerEndpoint(
            endpoint_name=endpoint_name,
            region_name="us-east-1",
            content_handler=content_handler,
    )
    memory = ConversationBufferMemory(return_messages=True)
    chain = ConversationChain(llm=llm, memory=memory)
    return chain

chatchain = load_chain()


# initialise session variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
    chatchain.memory.clear()
    
if 'widget_key' not in st.session_state:
    st.session_state['widget_key'] = str(randint(1000, 100000000))
if 'max_token' not in st.session_state:
    st.session_state.max_token = 200
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.1
if 'seed' not in st.session_state:
    st.session_state.seed = 0
if 'extract_audio' not in st.session_state:
    st.session_state.extract_audio = False
if 'option' not in st.session_state:
    st.session_state.option = "NLP"
    
def clear_button_fn():
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['widget_key'] = str(randint(1000, 100000000))
    st.widget_key = str(randint(1000, 100000000))
    st.session_state.extract_audio = False
    chatchain = load_chain(endpoint_name=endpoint_names['NLP'])
    chatchain.memory.clear()
    
    
def on_file_upload():
    st.session_state.extract_audio = True
    st.session_state['generated'] = []
    st.session_state['past'] = []
    # st.session_state['widget_key'] = str(randint(1000, 100000000))
    chatchain.memory.clear()
    
        

with st.sidebar:
    # Sidebar - the clear button is will flush the memory of the conversation
    st.sidebar.title("Conversation setup")
    clear_button = st.sidebar.button("Clear Conversation", key="clear", on_click=clear_button_fn)

    # upload file button
    uploaded_file = st.sidebar.file_uploader("Upload a file (text, image, or audio)", 
                                             key=st.session_state['widget_key'], 
                                             on_change=on_file_upload,
                                            )
    if uploaded_file:
        filename = uploaded_file.name
        print(filename)
        if filename.lower().endswith(('.flac', '.wav', '.webm', 'mp3')):
            st.session_state.option = "Audio"
            byteio = BytesIO(uploaded_file.getvalue())
            data = byteio.read()
            st.audio(data, format='audio/webm')
        else:
            st.session_state.option = "NLP"
    
    
    
        
    

left_column, _, right_column = st.columns([50, 2, 20])

with left_column:
    st.header("Building a multifunctional chatbot with Amazon SageMaker")
    # this is the container that displays the past conversation
    response_container = st.container()
    # this is the container with the input text box
    container = st.container()
    
    with container:
        # define the input text box
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("Input text:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send')
        
        
        # when the submit button is pressed we send the user query to the chatchain object and save the chat history
        if submit_button and user_input:
            st.session_state.option = "NLP"
            output = chatchain(user_input)["response"]
            print(output)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
        # when a file is uploaded we also send the content to the chatchain object and ask for confirmation
        elif uploaded_file is not None:
            if st.session_state.option == "Audio" and st.session_state.extract_audio:
                byteio = BytesIO(uploaded_file.getvalue())
                data = byteio.read()
                response = client.invoke_endpoint(EndpointName=whisper_endpoint_name, ContentType='audio/x-audio', Body=data)  
                output = json.loads(response['Body'].read())["text"]
                st.session_state['past'].append("I have uploaded an audio file. Plese extract the text from this audio file")
                st.session_state['generated'].append(output)
                content = "=== BEGIN AUDIO FILE ===\n"
                content += output
                content += "\n=== END AUDIO FILE ===\nPlease remember the audio file by saying 'Yes, I remembered the audio file'"
                output = chatchain(content)["response"]
                print(output)
                st.session_state.extract_audio = False
            elif st.session_state.option == "NLP":
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                content = "=== BEGIN FILE ===\n"
                content += stringio.read().strip()
                content += "\n=== END FILE ===\nPlease confirm that you have read that file by saying 'Yes, I have read the file'"
                output = chatchain(content)["response"]
                st.session_state['past'].append("I have uploaded a file. Please confirm that you have read that file.")
                st.session_state['generated'].append(output)

        st.write(f"Currently using a {st.session_state.option} model")


    # this loop is responsible for displaying the chat history
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
                

with right_column:

    max_tokens= st.slider(
        min_value=8,
        max_value=1024,
        step=1,
        # value=200,
        label="Number of tokens to generate",
        key="max_token"
    )
    temperature = st.slider(
        min_value=0.1,
        max_value=2.5,
        step=0.1,
        # value=0.4,
        label="Temperature",
        key="temperature"
    )
    seed = st.slider(
        min_value=0,
        max_value=1000,
        # value=0,
        step=1,
        label="Random seed to use for the generation",
        key="seed"
    )

    
