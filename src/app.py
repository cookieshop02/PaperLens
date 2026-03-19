import streamlit as st
import requests

API = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="PaperLens",
    page_icon="🔍",
    layout="wide"
)

# --- Session State Init ---
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "papers" not in st.session_state:
    st.session_state.papers = []


# ─────────────────────────────────────────────
#  SIDEBAR — Upload Panel
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 PaperLens")
    st.caption("Research paper Q&A with semantic transparency")
    st.divider()

    st.subheader("📤 Upload Papers")
    st.caption("Upload up to 4 research papers (PDF only, max 20MB each)")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("📥 Ingest Papers", use_container_width=True):
            for file in uploaded_files:
                if len(st.session_state.papers) >= 4:
                    st.warning("⚠️ Maximum 4 papers reached.")
                    break
                with st.spinner(f"Ingesting {file.name}..."):
                    resp = requests.post(
                        f"{API}/upload",
                        files={"file": (file.name, file.getvalue(), "application/pdf")}
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.papers = data["papers"]
                        st.success(f"✅ {file.name} — {data['chunks']} chunks")
                    else:
                        try:
                            st.error(f"❌ {resp.json().get('detail', 'Upload failed')}")
                        except Exception:
                            st.error(f"❌ HTTP {resp.status_code}: {resp.text}")

    st.divider()

    # --- Uploaded Papers Status ---
    st.subheader("📄 Uploaded Papers")
    if st.session_state.papers:
        for paper in st.session_state.papers:
            st.markdown(f"- `{paper}`")
        slots_left = 4 - len(st.session_state.papers)
        st.caption(f"{slots_left} slot(s) remaining")
    else:
        st.info("No papers uploaded yet.")

    if st.session_state.papers:
        if st.button("🗑️ Clear All Papers", use_container_width=True):
            resp = requests.delete(f"{API}/papers")
            if resp.status_code == 200:
                st.session_state.papers = []
                st.session_state.session_id = None
                st.session_state.messages = []
                st.success("✅ All papers cleared.")
                st.rerun()

    st.divider()

    # --- New Chat ---
    if st.button("💬 New Chat", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()


# ─────────────────────────────────────────────
#  MAIN AREA — Chat Interface
# ─────────────────────────────────────────────
st.title("🔍 PaperLens")
st.caption("Ask anything from your uploaded research papers")

if not st.session_state.papers:
    st.info("👈 Upload research papers from the sidebar to get started.")
    st.stop()

# --- Chat History Display ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input ---
query = st.chat_input("Ask a question from your papers...")

if query:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Searching papers..."):
            resp = requests.post(
                f"{API}/chat",
                json={
                    "query": query,
                    "session_id": st.session_state.session_id
                }
            )

        if resp.status_code != 200:
            st.error(f"❌ {resp.json().get('detail', 'Something went wrong.')}")
        else:
            data = resp.json()

            # Save session_id
            st.session_state.session_id = data["session_id"]

            # --- Answer ---
            confidence = data["confidence"]
            confidence_badge = {
                "high": "🟢 High confidence",
                "medium": "🟡 Medium confidence",
                "low": "🔴 Low confidence",
                "none": "⚫ No match"
            }.get(confidence, "")

            st.markdown(data["answer"])
            st.caption(confidence_badge)

            # Save to local history
            st.session_state.messages.append({
                "role": "assistant",
                "content": data["answer"]
            })

            st.divider()

            # ─── Semantic Breakdown ───
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("📊 Paper Relevance Scores")
                per_paper = data.get("per_paper_scores", {})
                if per_paper:
                    for paper, score in per_paper.items():
                        bar = "█" * int(score * 10)
                        st.markdown(f"**{paper}**")
                        st.progress(score, text=f"{score:.2f}")
                else:
                    st.info("No scores available.")

            with col2:
                st.subheader("🏆 Top 3 Chunks Overall")
                top_chunks = data.get("top_chunks_overall", [])
                for chunk in top_chunks:
                    with st.expander(
                        f"#{chunk['rank']} — {chunk['filename']} | Score: {chunk['score']:.2f}"
                    ):
                        st.markdown(chunk["text"])

            st.divider()

            # ─── Per-Paper Chunk Breakdown ───
            st.subheader("📄 Per-Paper Top Chunks")
            per_paper_chunks = data.get("top_chunks_per_paper", {})
            if per_paper_chunks:
                tabs = st.tabs(list(per_paper_chunks.keys()))
                for tab, (paper, chunks) in zip(tabs, per_paper_chunks.items()):
                    with tab:
                        for i, chunk in enumerate(chunks, 1):
                            with st.expander(f"Chunk {i} — Score: {chunk['score']:.2f}"):
                                st.markdown(chunk["text"])
            else:
                st.info("No per-paper breakdown available.")
