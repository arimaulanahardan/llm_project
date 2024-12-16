import streamlit as st

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ChatMessage
)

from workflow import initialize_chatbot, stream_events

# Function to initialize ChatBot
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def cached_initialize_chatbot():
    # Initialize the chatbot with stock-specific context
    return initialize_chatbot()

if __name__ == "__main__":
    # Set Streamlit page configuration
    st.title('Stock AI Agent')

    # Sidebar: User input for stock symbol
    stock_symbol = st.sidebar.text_input("Enter Stock Symbol", "")

    if not stock_symbol:
        st.sidebar.warning("Please enter a stock symbol to get started.")
    else:
        # Initialize the chatbot with the stock symbol
        if "chatbot" not in st.session_state:
            st.session_state.agent = cached_initialize_chatbot()

        # Initialize session state variables
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": f"Ask me anything about {stock_symbol}."}]

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Get user input question
        if prompt := st.chat_input("Ask about the stock"):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            outputs = stream_events(st.session_state.agent, prompt, stock_symbol)

            for output in outputs:
                try:
                    messages = output['messages']
                    last_message = messages[-1]
                    if isinstance(last_message, AIMessage) and last_message.content != '':
                        with st.chat_message("assistant"):
                            st.markdown(f"{last_message.content}")
                except:
                    pass
