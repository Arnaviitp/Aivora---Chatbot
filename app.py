import os
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import requests
import sseclient
import json

# --- Load .env for local development ---
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="AIVORA 🤖",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Access API Keys ---
openrouter_key = (
    st.secrets.get("api", {}).get("OPENROUTER_API_KEY") if "api" in st.secrets else os.getenv("OPENROUTER_API_KEY")
)
hf_key = (
    st.secrets.get("api", {}).get("HUGGINGFACE_API_KEY") if "api" in st.secrets else os.getenv("HUGGINGFACE_API_KEY")
)

if not openrouter_key:
    st.error("🚨 OPENROUTER_API_KEY not found! Please set it in Streamlit Secrets or .env")
    st.stop()

# --- Configure Hugging Face ---
hf_client = None
if hf_key:
    try:
        hf_client = InferenceClient(token=hf_key)
    except Exception as e:
        st.error(f"Failed to initialize Hugging Face client: {e}")

# --- Translation Model Map ---
model_map = {
    "French": "Helsinki-NLP/opus-mt-en-fr",
    "Spanish": "Helsinki-NLP/opus-mt-en-es",
    "German": "Helsinki-NLP/opus-mt-en-de",
    "Japanese": "Helsinki-NLP/opus-mt-en-jap"
}

# --- DeepSeek streaming function with silent HuggingFace fallback ---
def deepseek_chat_stream(prompt, history=None):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "Content-Type": "application/json"
    }

    messages = [{"role": "system", "content": (
        "You are AIVORA, a helpful AI assistant created by Arnav Anand of IIT Patna. "
        "Whenever someone asks 'Who made you?', reply: 'I was created by Arnav Anand from IIT Patna.'"
    )}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "deepseek/deepseek-chat-v3.1:free",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 800,
        "stream": True
    }

    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as r:
            # Silent fallback if daily limit exceeded
            if r.status_code == 429 and hf_client:
                fallback_text = hf_client.text_generation(prompt, model="gpt2")
                yield fallback_text[0]['generated_text']
                return

            r.raise_for_status()
            client = sseclient.SSEClient(r)
            for event in client.events():
                if event.data == "[DONE]":
                    break
                try:
                    data = json.loads(event.data)
                    token = data["choices"][0]["delta"].get("content")
                    if token:
                        yield token
                except:
                    continue

    except Exception:
        # If DeepSeek fails, fallback to HuggingFace silently
        if hf_client:
            fallback_text = hf_client.text_generation(prompt, model="gpt2")
            yield fallback_text[0]['generated_text']
        else:
            yield "⚠️ Failed to get response from DeepSeek and HuggingFace."

# --- HuggingFace feature helper ---
def run_huggingface_feature(feature, text):
    if not hf_client:
        return "⚠️ Hugging Face API not configured."

    try:
        if feature == "Summarization":
            result = hf_client.summarization(text, model="facebook/bart-large-cnn")
            return result.get("summary_text") if isinstance(result, dict) else str(result)

        elif feature == "Translation":
            model_id = model_map.get(language_choice, "Helsinki-NLP/opus-mt-en-fr")
            result = hf_client.translation(text, model=model_id)
            return result.get("translation_text") if isinstance(result, dict) else str(result)

        elif feature == "Sentiment Analysis":
            result = hf_client.text_classification(
                text, model="cardiffnlp/twitter-roberta-base-sentiment"
            )
            sentiment_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
            if isinstance(result, list) and result:
                label = sentiment_map.get(result[0]['label'], result[0]['label'])
                return f"Sentiment: {label} (score: {result[0]['score']:.2f})"
            return str(result)

    except Exception as e:
        return f"⚠️ Hugging Face request failed: {e}"

# --- Sidebar Features ---
feature_choice = st.sidebar.selectbox(
    "✨ Extra AI Features (AI-Super)",
    ["None", "Summarization", "Translation", "Sentiment Analysis"]
)
language_choice = st.sidebar.selectbox(
    "Translate to (if Translation selected):",
    ["French", "Spanish", "German", "Japanese"]
)

# --- Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Clear Chat ---
if st.sidebar.button("Clear Chat History", type="secondary"):
    st.session_state.messages = []
    st.rerun()

# --- App Title & Description ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🤖 AIVORA ✨</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 1.1em;'>Your Super Intelligent Assistant powered by AI.</p>",
    unsafe_allow_html=True
)
st.divider()

# --- Display Previous Messages ---
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- Welcome Message ---
if not st.session_state.messages:
    st.info("👋 Hello! I'm AIVORA, your personal AI assistant. How can I help you today?")
    st.markdown("Ask me anything or try the AI-Super features from the sidebar!")

# --- Chat Input with Streaming ---
if prompt := st.chat_input("Ask AIVORA anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        response_text = ""
        try:
            for token in deepseek_chat_stream(prompt, history=st.session_state.messages):
                response_text += token
                message_placeholder.markdown(response_text)
        except Exception as e:
            message_placeholder.markdown(f"⚠️ Error: {e}")

        # Apply HuggingFace Feature
        if feature_choice != "None":
            extra_output = run_huggingface_feature(feature_choice, prompt)
            response_text += f"\n\n---\n✨ **{feature_choice} Result:**\n{extra_output}"
            message_placeholder.markdown(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})

# --- Footer ---
st.divider()
st.markdown("<p style='text-align: center; color: #888;'>Created by Arnav Anand IITP.</p>", unsafe_allow_html=True)
