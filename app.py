import os
import streamlit as st
import google.generativeai as genai
import time # For simulating a slightly longer AI response if needed

from dotenv import load_dotenv
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="AIVORA 🤖", # App title
    page_icon="✨", # Choose a fun emoji icon
    layout="centered", # Can be "wide" or "centered"
    initial_sidebar_state="auto"
)

# --- Access and Configure Generative AI ---
secret_key = os.getenv('GOOGLE_API_KEY')

if not secret_key:
    st.error("🚨 GOOGLE_API_KEY not found! Please set it in your `.env` file.")
    st.stop() # Stop the app if the key is missing

genai.configure(api_key=secret_key)

# Initialize the model once
try:
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.error(f"Failed to initialize the Gemini model: {e}")
    st.stop()

# --- App Title and Description ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🤖 AIVORA ✨</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 1.1em;'>Your friendly AI assistant powered by Google Gemini.</p>",
    unsafe_allow_html=True
)
st.divider()

# --- Initialize chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Clear Chat Button ---
if st.sidebar.button("Clear Chat History", type="secondary"):
    st.session_state.messages = []
    st.rerun() # Rerun to clear the display

# --- Display chat messages from history on app rerun ---
for message in st.session_state.messages:
    # Use distinct avatars for user and assistant
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- Welcome Message (if chat is empty) ---
if not st.session_state.messages:
    st.info("👋 Hello! I'm AIVORA, your personal AI assistant. How can I help you today?")
    st.markdown("Feel free to ask me anything!")


# --- React to user input ---
if prompt := st.chat_input("Ask AIVORA anything..."):
    # Display user message in chat message container
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container with a spinner
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("AIVORA is thinking..."):
            try:
                # Play with this variable
                response_obj = model.generate_content(prompt)

                # Check if the response has valid content
                if response_obj and hasattr(response_obj, 'text'):
                    response = response_obj.text
                elif response_obj and response_obj.candidates:
                    # Fallback for some API responses that might structure content differently
                    response = response_obj.candidates[0].content.parts[0].text
                else:
                    response = "I'm sorry, I couldn't generate a response. Please try again."

                st.markdown(response)
            except Exception as e:
                response = f"Oops! Something went wrong: {e}"
                st.error(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- Footer ---
st.divider()
st.markdown(
    """
    <p style='text-align: center; color: #888; font-size: 0.8em;'>
        Created by Arnav Anand IITP.
    </p>
    """,
    unsafe_allow_html=True
)