import os

import streamlit as st

from service.haystack_documentation_pipeline import return_haystack_documentation_agent

st.title('Haystack Documentation Chatbot')

if 'image_bytes' not in st.session_state:
    st.session_state.image_bytes = open('service/assets/bot.png', 'rb').read()

if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', 'content': 'Hello there!'}]

with st.sidebar:
    st.image('service/assets/bot.png')
    st.markdown(
        """
        # Haystack Documentation Chatbot
        
        This chatbot can answer questions about the Haystack documentation.
        
        ## How to use
        
        1. Type your question in the chat input box.
        2. Press enter.
        3. Wait for the chatbot to respond (since it works as an agent responses may take a while).
        4. enjoy!
        """
    )

for message in st.session_state.messages:
    with st.chat_message(
        message['role'],
        avatar=st.session_state.image_bytes if message['role'] == 'assistant' else None,
    ):
        st.markdown(message['content'])

if 'agent' not in st.session_state:
    st.session_state.agent = return_haystack_documentation_agent(openai_key=os.environ['OPENAI_KEY'])

if prompt := st.chat_input('What is up?"'):
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    chat_message = st.chat_message(name='assistant', avatar=st.session_state.image_bytes)
    with chat_message:
        with st.spinner('Thinking...'):
            response = st.session_state.agent.run(query=prompt)
    answer = response['answers'][0].answer
    chat_message.markdown(answer)

    st.session_state.messages.append({'role': 'assistant', 'content': answer})
