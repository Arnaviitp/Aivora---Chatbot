import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# --- Load .env for local development ---
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="AIVORA 🤖", 
    page_icon="✨", 
    layout="centered", 
    initial_sidebar_state="auto"
)

# --- Access API Keys (Cloud -> Secrets, Local -> .env) ---
secret_key = (
    st.secrets.get("api", {}).get("GOOGLE_API_KEY")   # Streamlit Cloud
    if "api" in st.secrets 
    else os.getenv("GOOGLE_API_KEY")                  # Local .env
)

hf_key = (
    st.secrets.get("api", {}).get("HUGGINGFACE_API_KEY")
    if "api" in st.secrets
    else os.getenv("HUGGINGFACE_API_KEY")
)

if not secret_key:
    st.error("🚨 GOOGLE_API_KEY not found! Please set it in Streamlit Secrets (Cloud) or .env (Local).")
    st.stop()

# --- Configure Gemini ---
try:
    genai.configure(api_key=secret_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.error(f"Failed to initialize Gemini model: {e}")
    st.stop()

# --- Configure Hugging Face ---
hf_client = None
if hf_key:
    try:
        hf_client = InferenceClient(token=hf_key)
    except Exception as e:
        st.error(f"Failed to initialize Hugging Face client: {e}")

# --- App Title and Description ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🤖 AIVORA ✨</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 1.1em;'>Your Super Intelligent Assistant powered by AI.</p>",
    unsafe_allow_html=True
)
st.divider()

# --- Initialize chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Clear Chat Button ---
if st.sidebar.button("Clear Chat History", type="secondary"):
    st.session_state.messages = []
    st.rerun()

# --- Sidebar Extra Features ---
feature_choice = st.sidebar.selectbox(
    "✨ Extra AI Features (Hugging Face)",
    ["None", "Summarization", "Translation (EN → FR)", "Sentiment Analysis"]
)

# --- Helper function for Hugging Face features ---
def run_huggingface_feature(feature, text):
    if not hf_client:
        return "⚠️ Hugging Face API not configured."

    try:
        if feature == "Summarization":
            # summarization expects just the text
            result = hf_client.summarization(text, model="facebook/bart-large-cnn")
            if isinstance(result, dict) and "summary_text" in result:
                return result["summary_text"]
            if isinstance(result, list) and result and "summary_text" in result[0]:
                return result[0]["summary_text"]
            return str(result)

        elif feature == "Translation (EN → FR)":
    result = hf_client.translation(text, model="Helsinki-NLP/opus-mt-en-fr")
    # result is a TranslationOutput object
    if hasattr(result, "translation_text"):
        return result.translation_text
    return str(result)


        elif feature == "Sentiment Analysis":
            result = hf_client.text_classification(
                text, model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            if isinstance(result, list) and result:
                return f"Sentiment: {result[0]['label']} (score: {result[0]['score']:.2f})"
            return str(result)

    except Exception as e:
        return f"⚠️ Hugging Face request failed: {e}"

    return None


# --- Display chat messages ---
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- Welcome Message ---
if not st.session_state.messages:
    st.info("👋 Hello! I'm AIVORA, your personal AI assistant. How can I help you today?")
    st.markdown("Feel free to ask me anything!")

# --- Chat Input ---
if prompt := st.chat_input("Ask AIVORA anything..."):
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("AIVORA is thinking..."):
            try:
                # Step 1: Get Gemini response
                response_obj = model.generate_content(prompt)
                if response_obj and hasattr(response_obj, 'text'):
                    response = response_obj.text
                elif response_obj and response_obj.candidates:
                    response = response_obj.candidates[0].content.parts[0].text
                else:
                    response = "I'm sorry, I couldn't generate a response. Please try again."

                # Step 2: Apply Hugging Face feature if selected
                if feature_choice != "None":
                    extra_output = run_huggingface_feature(feature_choice, prompt)
                    response += f"\n\n---\n✨ **{feature_choice} Result:**\n{extra_output}"

                st.markdown(response)

            except Exception as e:
                response = f"Oops! Something went wrong: {e}"
                st.error(response)

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
