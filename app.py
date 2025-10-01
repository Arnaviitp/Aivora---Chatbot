import os
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import requests

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
deepseek_key = (
    st.secrets.get("api", {}).get("DEEPSEEK_API_KEY") if "api" in st.secrets else os.getenv("DEEPSEEK_API_KEY")
)
hf_key = (
    st.secrets.get("api", {}).get("HUGGINGFACE_API_KEY") if "api" in st.secrets else os.getenv("HUGGINGFACE_API_KEY")
)

if not deepseek_key:
    st.error("🚨 DEEPSEEK_API_KEY not found! Please set it in Streamlit Secrets (Cloud) or .env (Local).")
    st.stop()

# --- Configure Hugging Face ---
hf_client = None
if hf_key:
    try:
        hf_client = InferenceClient(token=hf_key)
    except Exception as e:
        st.error(f"Failed to initialize Hugging Face client: {e}")

# --- Helper function: DeepSeek API call ---
def deepseek_chat(prompt, history=None):
    """
    Calls DeepSeek V3.1 API with user prompt + optional history.
    """
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {deepseek_key}",
        "Content-Type": "application/json"
    }

    messages = [{"role": "system", "content": "You are AIVORA, a helpful AI assistant."}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 800
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ DeepSeek request failed: {e}"

# --- App Title and Description ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🤖 AIVORA ✨</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 1.1em;'>Your Super Intelligent Assistant powered by DeepSeek V3.</p>",
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
    "✨ Extra AI Features (AI-Super)",
    ["None", "Summarization", "Translation", "Sentiment Analysis"]
)

# For translation target languages
language_choice = st.sidebar.selectbox(
    "Translate to (if Translation selected):",
    ["French", "Spanish", "German", "Japanese"]
)
model_map = {
    "French": "Helsinki-NLP/opus-mt-en-fr",
    "Spanish": "Helsinki-NLP/opus-mt-en-es",
    "German": "Helsinki-NLP/opus-mt-en-de",
    "Japanese": "Helsinki-NLP/opus-mt-en-ja"
}

# --- Helper function for Hugging Face features ---
def run_huggingface_feature(feature, text):
    if not hf_client:
        return "⚠️ Hugging Face API not configured."

    try:
        if feature == "Summarization":
            result = hf_client.summarization(text, model="facebook/bart-large-cnn")
            if hasattr(result, "summary_text"):
                return result.summary_text
            if isinstance(result, list) and result and hasattr(result[0], "summary_text"):
                return result[0].summary_text
            return str(result)

        elif feature == "Translation":
            translation_model = model_map.get(language_choice, "Helsinki-NLP/opus-mt-en-fr")
            result = hf_client.translation(text, model=translation_model)
            if hasattr(result, "translation_text"):
                return result.translation_text
            return str(result)

        elif feature == "Sentiment Analysis":
            result = hf_client.text_classification(
                text,
                model="cardiffnlp/twitter-roberta-base-sentiment"
            )

            sentiment_map = {
                "LABEL_0": "Negative",
                "LABEL_1": "Neutral",
                "LABEL_2": "Positive"
            }

            if isinstance(result, list) and result:
                label = sentiment_map.get(result[0]['label'], result[0]['label'])
                return f"Sentiment: {label} (score: {result[0]['score']:.2f})"
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
    st.info("👋 Hello! I'm AIVORA (DeepSeek V3). How can I help you today?")
    st.markdown("Feel free to ask me anything!")

# --- Chat Input ---
if prompt := st.chat_input("Ask AIVORA anything..."):
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("AIVORA is thinking..."):
            try:
                # Step 1: Get DeepSeek response
                response = deepseek_chat(prompt, history=st.session_state.messages)

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

