import base64
import re

import streamlit as st
import ollama


def parse_thinking(text: str) -> tuple[str, str]:
    """
    Split raw model output into (thinking, clean_content).
    Handles:
      - <think>...</think> tags (standard Gemma 4 native format)
      - <channel|> separator (Gemma 4 artifact: everything before last occurrence = scratchpad)
    """
    # 1. Standard <think> tags
    thinking_parts = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
    if thinking_parts:
        thinking = '\n'.join(thinking_parts).strip()
        content = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return thinking, content

    # 2. <channel|> separator — take the last segment as the final answer,
    #    everything before as scratchpad
    if '<channel|>' in text:
        parts = text.split('<channel|>')
        thinking = '<channel|>'.join(parts[:-1]).strip()
        content = parts[-1].strip()
        return thinking, content

    # 3. No recognisable thinking markers — return as-is
    return '', text.strip()
# streamlit run app.py

# # change for your local model ################################
MODEL = ""
# ############################################
#### Model Class
class Model:
    def __init__(self, model, model_name, title, blanktext, logo):
        self.model_name = model_name  # name to show in dropdown
        self.model = model  # actual model name for ollama
        self.title = title  # title to show in app when model is selected
        self.blanktext = blanktext # placeholder text for chat input
        self.logo = logo # logo url for model (not currently used, but could be added to sidebar or something in the future)


###############################################
# 2 model setup
# MODEL1 = "llama3.2:3b"
# MODEL2 = "phi4-mini"
MODEL1 = "atp-gemma4-bf16"
MODEL2 = "atp-gemma4-e4b"

MODEL1_NAME = "ATP 2-01.3 (BF16)"
MODEL2_NAME = "ATP 2-01.3 (Q4_K_M)"
blankvalue = Model("No Agent Selected", "No Agent Selected", "", "", "")
model1 = Model(MODEL1, MODEL1_NAME, "ATP 2-01.3 — BF16 (Full Quantization)", "Ask me a question about ATP 2-01.3?", "")
model2 = Model(MODEL2, MODEL2_NAME, "ATP 2-01.3 — Q4_K_M (Quantized)", "Ask me a question about ATP 2-01.3?", "")
models  = [blankvalue, model1, model2]



###################################################################################
## Logos
###################################################################################

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64_dina  = get_base64_of_bin_file('static/DINA_LOGO.png')
img_base64_dod   = get_base64_of_bin_file('static/dod_logo.png')
img_base64_1     = get_base64_of_bin_file('static/army_logo_star.png')
img_base64_2     = get_base64_of_bin_file('static/army_int_logo.png')
img_base64_3     = get_base64_of_bin_file('static/forth_logo2.png')
img_base64_4     = get_base64_of_bin_file('static/IMCOM_logo.png')

# All logos live in the sidebar — no fixed positioning, no overlap with chat
with st.sidebar:
    # DINA logo — top of sidebar, centred
    st.markdown(
        f"""
        <div style="text-align:center; padding: 12px 0 8px 0;">
            <img src="data:image/png;base64,{img_base64_dina}" width="160">
        </div>
        <hr style="margin: 4px 0 12px 0;">
        """,
        unsafe_allow_html=True,
    )

    # DoD logo — below DINA, centred
    st.markdown(
        f"""
        <div style="text-align:center; padding-bottom: 12px;">
            <img src="data:image/png;base64,{img_base64_dod}" width="80">
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Spacer so settings land below branding
    st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)

# st.markdown(
#     f"""
#     <style>
#     .logo-container {{
#         position: fixed;
#         bottom: 20px;
#         right: 20px;
#         display: flex;
#         flex-direction: column; /* Stack vertically */
#         gap: 10px;             /* Space between logos */
#         z-index: 100;
#     }}
#     </style>
#     <div class="logo-container">
#         <img src="data:image/png;base64,{img_base64_1}" width="80">
#         <img src="data:image/png;base64,{img_base64_2}" width="80">
#     </div>
#     """,
#     unsafe_allow_html=True
# )



# #Horizontal Logos
# st.markdown(
#     f"""
#     <style>
#     .logo-container {{
#         position: fixed;
#         bottom: 20px;
#         right: 20px;
#         display: flex;
#         flex-direction: row;    /* Place side-by-side */
#         align-items: center;    /* Center them vertically relative to each other */
#         gap: 15px;
#         z-index: 100;
#     }}
#     </style>
#     <div class="logo-container">
#         <img src="data:image/png;base64,{img_base64_1}" width="80">
#         <img src="data:image/png;base64,{img_base64_2}" width="80">
#     </div>
#     """,
#     unsafe_allow_html=True
# )


### Icons for chat
icons = {"user": "👤", "assistant": "✨"}


###############################################
## Model Selection
###############################################

MODEL = st.selectbox(
    "Selected Agent",
    options=models,
    format_func=lambda x: x.model_name,
    key="model_selection"
)

if st.session_state.model_selection == MODEL1_NAME:
    MODEL = MODEL1
elif st.session_state.model_selection == MODEL2_NAME:
    MODEL = MODEL2

# Clear chat history when the selected agent changes
if "active_model" not in st.session_state:
    st.session_state.active_model = MODEL.model_name
if st.session_state.active_model != MODEL.model_name:
    st.session_state.messages = []
    st.session_state.active_model = MODEL.model_name

if MODEL.model_name == "No Agent Selected":
    st.text("Please select a Agent to start chatting.")
else:
    st.title(MODEL.title)
    # --- 1. Sidebar Setup ---
    with st.sidebar:
        st.header("Settings")
        show_thinking = st.toggle("Show Thinking Process", value=False, key="show_thinking")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        # 4 unit logos — single row pinned to the very bottom of the sidebar
        st.markdown(
            f"""
            <div style="
                position: fixed;
                bottom: 16px;
                display: flex;
                flex-direction: row;
                justify-content: space-evenly;
                align-items: center;
                width: 260px;
                padding: 8px 4px;
            ">
                <img src="data:image/png;base64,{img_base64_1}" style="width:44px; height:auto;">
                <img src="data:image/png;base64,{img_base64_2}" style="width:44px; height:auto;">
                <img src="data:image/png;base64,{img_base64_3}" style="width:44px; height:auto;">
                <img src="data:image/png;base64,{img_base64_4}" style="width:44px; height:auto;">
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- 2. Session State Initialization ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- 3. Display Chat History ---
    for message in st.session_state.messages:
        if message["role"] == "assistant" and show_thinking and message.get("thinking"):
            with st.expander("💭 Thinking process"):
                st.markdown(message["thinking"])
        with st.chat_message(message["role"], avatar=icons.get(message["role"])):
            st.markdown(message["content"])

    # --- 4. Chat Logic ---
    if prompt := st.chat_input(MODEL.blanktext):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=icons.get("user")):
            st.markdown(prompt)

        # Stream the assistant response
        # thinking_placeholder is rendered first so it sits ABOVE the chat bubble
        thinking_placeholder = st.empty()

        ollama_messages = [{"role": m["role"], "content": m["content"]}
                           for m in st.session_state.messages]

        thinking_buffer = ""
        content_buffer = ""

        with st.chat_message("assistant", avatar=icons.get("assistant")):
            content_placeholder = st.empty()
            stream = ollama.chat(
                model=MODEL.model,
                messages=ollama_messages,
                think=True,   # Ollama native thinking separation (0.6+)
                stream=True,
            )
            for chunk in stream:
                msg = chunk['message']
                # Ollama native thinking field (preferred)
                thinking_buffer += msg.get('thinking') or ''
                content_buffer  += msg.get('content')  or ''
                content_placeholder.markdown(content_buffer + "▌")
            content_placeholder.markdown(content_buffer)

        # Fallback: if Ollama didn't separate thinking, parse it from content
        if not thinking_buffer and content_buffer:
            thinking_buffer, content_buffer = parse_thinking(content_buffer)

        # Render thinking expander in the placeholder ABOVE the chat bubble
        if show_thinking and thinking_buffer:
            with thinking_placeholder.expander("💭 Thinking process"):
                st.markdown(thinking_buffer)

        # Save with thinking stored separately
        st.session_state.messages.append({
            "role": "assistant",
            "content": content_buffer,
            "thinking": thinking_buffer,
        })