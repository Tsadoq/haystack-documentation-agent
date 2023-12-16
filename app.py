import os

import streamlit as st

from service.haystack_documentation_pipeline import return_haystack_documentation_agent

st.title("Haystack Documentation Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello there!"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if 'agent' not in st.session_state:
    st.session_state.agent = return_haystack_documentation_agent(openai_key=os.environ['OPENAI_KEY'])

# React to user input
if prompt := st.chat_input("What is up?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    chat_message = st.chat_message("assistant")
    with chat_message:
        with st.spinner("Thinking..."):
            response = st.session_state.agent.run(query=prompt)
    answer = response["answers"][0].answer
    chat_message.markdown(answer)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
