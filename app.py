import streamlit as st
import time

from src.pdf_reader import pdf_to_text
from src.text_splitter import text_splitterchunk
from src.embedding import get_embeddings
from src.vectorstores import vector_store
from src.rag_chain import create_rag_chain

# PAGE CONFIG
st.set_page_config(
    page_title="Multimodal GenAI ChatBot",
    page_icon="🤖",
    layout="wide"
)

# Keep simple session state ONLY for the chat logs so they don't erase on screen refresh
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

# CUSTOM CSS (Fixes Scrolling, Layout Padding & Bottom Anchoring)
st.markdown("""
<style>
/* Main App Background */
[data-testid="stAppViewContainer"] {
    background: #0f172a;
}

/* Fix main page container margins to prevent overall page vertical bars */
[data-testid="stMainBlockContainer"] {
    padding-top: 1.5rem !important;
    padding-bottom: 8rem !important; /* Safety margin so chat content isn't blocked by the input box */
    overflow: hidden !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #1e293b;
}

/* Global Typography Color Overrides */
h1, h2, h3, h4, h5, h6, p, label, span {
    color: white !important;
}

/* ===================================================
   FIXES: FILE UPLOADER VISIBILITY & WHITE CARD REMOVAL
   =================================================== */
/* Targets the container wrapper */
.stFileUploader {
    background: #1e293b;
    padding: 12px;
    border-radius: 12px;
}

/* REMOVES THE WHITE CARD BACKGROUND */
[data-testid="stFileUploaderDropzone"] {
    background-color: #111827 !important; /* Changes the white card to a sleek dark card */
    border: 1px dashed #334155 !important;
}

/* Fixes the 'Upload' button inside the dropzone */
[data-testid="stFileUploaderDropzone"] button {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
}

/* Fixes the text and info colors inside the uploader so they are readable on dark background */
[data-testid="stFileUploader"] section div span, 
[data-testid="stFileUploader"] section div small,
[data-testid="stFileUploaderDropzone"] div {
    color: #94a3b8 !important;
}

[data-testid="stFileUploader"] button div span {
    color: white !important;
}

/* ===================================================
   FIXES: YOUTUBE LINK PLACEHOLDER VISIBILITY
   =================================================== */
.stTextInput input {
    background: #1e293b !important;
    color: white !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
}

.stTextInput input::placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
}

.stSelectbox div[data-baseweb="select"] {
    background: #1e293b;
    color: white;
    border-radius: 12px;
}

.stButton button {
    width: 100%;
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px;
    font-size: 16px;
    font-weight: 600;
}

/* Clean Center Title Layout */
.main-title {
    text-align: center;
    color: white;
    font-size: 38px;
    font-weight: bold;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    color: #cbd5e1;
    font-size: 15px;
    margin-bottom: 20px;
}

/* SCROLLABLE CHAT CONTAINER (GEMINI STYLE) */
.chat-scroll-container {
    max-height: 56vh; /* Prevents the window itself from scrolling */
    overflow-y: auto;
    padding-right: 15px;
    margin-bottom: 10px;
}

.gemini-qa-block {
    margin-bottom: 30px;
    width: 100%;
}

.gemini-user-card {
    font-size: 20px;
    font-weight: 500;
    color: #f8fafc !important;
    margin-bottom: 12px;
    padding-left: 5px;
    line-height: 1.4;
}

.gemini-ai-card {
    background: #1e293b;
    border: 1px solid #23354d;
    border-radius: 16px;
    padding: 20px;
    color: #e2e8f0 !important;
    font-size: 16px;
    line-height: 1.6;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

/* PERFECT FIXED BOTTOM INPUT (NO BACKGROUND BLOCKS) */
/* Make Streamlit's native bottom block structural wrapper completely invisible */
div[data-testid="stBottom"] {
    background-color: transparent !important;
    background-image: none !important;
    bottom: 25px !important;
}

/* Ensure the sub-container block aligns perfectly without wide stretching layouts */
div[data-testid="stBottom"] > div {
    background: transparent !important;
}

/* Make chat input box floating and stylish */
[data-testid="stChatInput"] {
    background-color: #1e293b !important;
    border-radius: 30px !important;
    border: 1px solid #334155 !important;
    padding: 6px 12px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5) !important;
}

/* Custom Scrollbar */
.chat-scroll-container::-webkit-scrollbar {
    width: 6px;
}
.chat-scroll-container::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# SIDEBAR CONTENT
with st.sidebar:
    st.title("📘 GenAI Engine")

    is_context_ready = False

    st.subheader("📂 Upload Context")
    uploaded_files = st.file_uploader("Choose PDF Files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        is_context_ready = True
        
    if st.button("🚀 Process Documents"):
        message_container = st.empty()
        
        if not uploaded_files:
            message_container.error("⚠️ Upload field is empty!")
            time.sleep(3)
            message_container.empty()
            st.rerun()
        else:
            with st.spinner("Extracting text from PDF files..."):

                # 1 pdf file data 
                combain_text = pdf_to_text(uploaded_files)

                # 2 text splitter (chunk)
                document = text_splitterchunk(combain_text)

                # 3 get embeddings
                embedding = get_embeddings()

                # 4 data store vecter db
                vectorData = vector_store(document, embedding)
                
                # 5. Build RAG chain and lock it to the persistent session state
                st.session_state.qa_pair_question = create_rag_chain(vectorData)

    st.markdown("<hr style='border-color: #1e293b;'>", unsafe_allow_html=True)

    st.sidebar.header(":gear: Chat History")

    # Ensure 'history' exists in session state before looping
    if "history" in st.session_state and st.session_state.history:
        for item in st.session_state.history:
            # Create a clean white card structure for the sidebar width
            st.sidebar.markdown(f"""
            <div style="
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 12px;
                margin-bottom: 10px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            ">
                <div style="color: #1e293b !important; font-weight: 600; font-size: 13px; margin-bottom: 4px; word-wrap: break-word;">
                    👤 {item['history']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div style="text-align: center; color: #ffffff; !important; font-size: 13px; padding: 10px;">
            No conversation history yet.
        </div>
        """, unsafe_allow_html=True)


# MAIN CENTERED LAYOUT WORKSPACE
st.markdown('<div style="margin-top: 3vh;"></div>', unsafe_allow_html=True) 
st.markdown('<div class="main-title">🤖 Multimodal GenAI Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ask anything about your uploaded documents</div>', unsafe_allow_html=True)

layout_left, layout_center, layout_right = st.columns([1.5, 7, 1.5])

with layout_center:
    

    # 1. CHAT HISTORY CONTAINER (ON TOP)

    if st.session_state.messages:
        chat_html_stream = ""
        
        for qa_pair in st.session_state.messages:
            chat_html_stream += (
                f'<div class="gemini-qa-block">'
                f'<div class="gemini-user-card">👤 {qa_pair["question"]}</div>'
                f'<div class="gemini-ai-card">{qa_pair["answer"]}</div>'
                f'</div>'
            )
        
        st.markdown(f'<div class="chat-scroll-container">{chat_html_stream}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding:40px; border: 1px dashed #334155; border-radius:15px; background:#111827; margin-bottom: 20px;">
            <p style='color:#64748b !important; margin:0;'>Your conversation stream will populate right here once you type a question below.</p>
        </div>
        """, unsafe_allow_html=True)


    # 2. CHAT INPUT FIELD (ANCHORED AT THE BOTTOM)

    if prompt := st.chat_input("Ask anything..."):
        
        if not is_context_ready:
            ai_response = "⚠️ No active context file found. Please upload a PDF or paste a YouTube stream in the sidebar first."
        else:
            try:
                # Use invoke instead of run to bypass modern LangChain deprecations
                response = st.session_state.qa_pair_question.invoke({"query": prompt})
                ai_response = response["result"]
            except Exception as e:
                ai_response = f"❌ Error querying model: {str(e)}"
        
        st.session_state.history.append({
            "history":prompt
        })

        st.session_state.messages.append({
            "question": prompt,
            "answer": ai_response
        })
        
        st.rerun()