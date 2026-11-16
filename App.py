# import os
# import re
# import streamlit as st
# from dotenv import load_dotenv

# from src.Models.ResponseGeneration import TextGeneration
# from src.Indexing.data_extraction.DataExtractorFactory import DataExtractorFactory
# from src.Models.EmbeddingModel import MultilingualEmbeddingGenerator
# from src.Indexing.preprocessor.TextPreprocessor import TextPreprocessor
# from src.Indexing.preprocessor.VectorDB import VectorDB
# from src.RAGPipeline import Pipeline
# from src.Memory.SQLiteMemory import SQLiteChatMemory

# load_dotenv()

# st.set_page_config(
#     page_title="Production Multilingual RAG System",
#     page_icon="🤖",
#     layout="wide",
# )

# st.title("🤖 Advanced Production RAG System")
# st.markdown("Enterprise-grade Retrieval-Augmented Generation pipeline with Guardrails, Hybrid Search, and Citations.")

# # ==============================================================================
# # SIDEBAR CONFIGURATION OPTIONS
# # ==============================================================================
# st.sidebar.header("⚙️ Pipeline Configuration")

# # --- 1. LLM Settings ---
# with st.sidebar.expander("🤖 LLM Settings", expanded=True):
#     llm_provider = st.selectbox(
#         "LLM Provider",
#         options=["groq", "openai", "anthropic", "ollama"],
#         index=0,
#     )
    
#     default_models = {
#         "groq": "openai/gpt-oss-120b",
#         "openai": "gpt-4o",
#         "anthropic": "claude-3-5-sonnet-20241022",
#         "ollama": "llama3",
#     }
    
#     llm_model_name = st.text_input("Model ID", value=default_models.get(llm_provider, ""))
#     user_llm_api_key = st.text_input("Custom LLM API Key (Optional)", type="password")
#     temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
#     verify_ssl = st.checkbox("Verify SSL Certificates", value=False)

# # --- 2. Embedding & Vector DB Settings ---
# with st.sidebar.expander("📊 Embedding & Vector DB", expanded=False):
#     embed_provider = st.selectbox(
#         "Embedding Provider",
#         options=["huggingface", "huggingface_api", "openai", "cohere"],
#         index=0,
#     )
    
#     embed_model_name = st.text_input(
#         "Embedding Model", 
#         value="intfloat/multilingual-e5-small" if embed_provider == "huggingface" else ""
#     )
#     user_embed_api_key = st.text_input("Custom Embedding API Key (Optional)", type="password")
#     compute_device = st.selectbox("Compute Device (Local)", options=["cpu", "cuda", "mps"], index=0)
    
#     # ChromaDB collection names must not contain spaces
#     collection_name = st.text_input("Chroma DB Collection Name", value="youtube_video_transcripts")

# # --- 3. Retrieval & Guardrail Controls ---
# with st.sidebar.expander("🛠 Retrieval & Guardrails", expanded=False):
#     enable_streaming = st.checkbox("⚡ Enable Live Token Streaming", value=True)
#     enable_input_guardrail = st.checkbox("Enable Input Guardrail", value=True)
#     enable_output_guardrail = st.checkbox("Enable Output Guardrail", value=True)
#     enable_pre_retrieval = st.checkbox("Enable Pre-Retrieval (Rewriting)", value=True)
#     history_window = st.slider("Conversation Memory Window (Turns)", min_value=2, max_value=12, value=6, step=2)
#     enable_compression = st.checkbox("Enable Contextual Compression", value=True)
#     enable_lost_in_middle = st.checkbox("Enable Lost-In-The-Middle Reordering", value=True)
#     top_n_rerank = st.slider("Top-N Reranked Documents", min_value=1, max_value=10, value=4)
#     max_context_chars = st.number_input("Max Context Character Limit", value=4000, step=500)
#     reranker_model = st.text_input("Reranker Model", value="BAAI/bge-reranker-base")


# # ==============================================================================
# # INITIALIZATION & CACHING FUNCTIONS
# # ==============================================================================
# @st.cache_resource
# def get_llm_instance(provider, model_name, api_key, temp, ssl_verify):
#     return TextGeneration(
#         provider=provider,
#         model_name=model_name if model_name else None,
#         api_key=api_key if api_key else None,
#         temperature=temp,
#         verify_ssl=ssl_verify,
#     ).llm


# @st.cache_resource
# def get_embedder_instance(provider, model_name, api_key, device):
#     return MultilingualEmbeddingGenerator(
#         provider=provider,
#         model_name=model_name if model_name else None,
#         api_key=api_key if api_key else None,
#         device=device,
#     )


# # Instantiate Embedder & LLM
# try:
#     active_llm = get_llm_instance(
#         llm_provider, llm_model_name, user_llm_api_key, temperature, verify_ssl
#     )
#     active_embedder_service = get_embedder_instance(
#         embed_provider, embed_model_name, user_embed_api_key, compute_device
#     )
    
#     vector_db = VectorDB(
#         embedding_function=active_embedder_service.embedder,
#         collection_name=collection_name,
#         persist_directory="./my_chroma_data",
#     )
#     chat_memory = SQLiteChatMemory(db_path="conversations.db")
# except Exception as e:
#     st.error(f"Initialization Error: {str(e)}")
#     st.stop()


# # ==============================================================================
# # MAIN APPLICATION TABS
# # ==============================================================================
# tab1, tab2 = st.tabs(["📌 Document Ingestion", "💬 RAG Q&A Interface"])

# # ------------------------------------------------------------------------------
# # TAB 1: DOCUMENT INGESTION
# # ------------------------------------------------------------------------------
# with tab1:
#     st.subheader("Extract & Index New Source Documents")
    
#     col1, col2 = st.columns([1, 2])
#     with col1:
#         source_type = st.selectbox(
#             "Select Data Source Type",
#             options=["YouTube Transcript", "Web Article"],
#             index=0,
#         )
#     with col2:
#         source_url = st.text_input(
#             "Enter Source URL / Path",
#             value="https://www.youtube.com/",
#         )

#     if st.button("🚀 Process & Index Source", use_container_width=True):
#         with st.spinner("Extracting text and chunking document..."):
#             try:
#                 # 1. Dynamic Extractor Selection via Factory
#                 extractor = DataExtractorFactory.get_extractor(source_type)
                
#                 # 2. Text Preprocessing & Chunking
#                 preprocessor = TextPreprocessor(embeddings=active_embedder_service.embedder)
#                 documents = preprocessor.process(extractor, source=source_url)
                
#                 # 3. Add to Vector DB
#                 vector_db.add_documents(documents)
                
#                 # 4. Save processed documents in session state for BM25 hybrid search
#                 st.session_state["indexed_documents"] = documents
#                 st.success(f"Successfully processed & indexed {len(documents)} document chunks into '{collection_name}'!")
            
#             except Exception as e:
#                 st.error(f"Ingestion Failed: {str(e)}")

# # ------------------------------------------------------------------------------
# # TAB 2: RAG Q&A INTERFACE (Multi-Turn Chat with SQLite Memory)
# # ------------------------------------------------------------------------------
# with tab2:
#     st.subheader("Ask Questions Against Your Knowledge Base")

#     # --- Session Controls ---
#     col_session, col_clear = st.columns([3, 1])
#     with col_session:
#         if "session_id" not in st.session_state:
#             st.session_state["session_id"] = "user_session_1"
#         session_id = st.text_input(
#             "Active Conversation Session ID:",
#             value=st.session_state["session_id"],
#             help="Switch sessions or enter a new ID to start an isolated conversation thread.",
#         )
#         st.session_state["session_id"] = session_id
#     with col_clear:
#         st.write("")
#         st.write("")
#         if st.button("🗑️ Clear Session History", use_container_width=True):
#             chat_memory.clear_session(session_id)
#             st.success(f"History cleared for session: '{session_id}'")
#             st.rerun()

#     # --- Render Past Messages from SQLite ---
#     past_messages = chat_memory.get_messages(session_id=session_id, limit=10)
#     if past_messages:
#         with st.expander(f"📜 Session History ({len(past_messages)} messages)", expanded=False):
#             for msg in past_messages:
#                 role_icon = "🧑‍💻" if msg["role"] == "user" else "🤖"
#                 st.markdown(f"**{role_icon} {msg['role'].title()}:** {msg['message']}")

#     user_query = st.text_input(
#         "Enter your question or follow-up:",
#         placeholder="e.g., 'What's coming in your mind?'",
#     )

#     if st.button("Search & Generate Answer", type="primary", use_container_width=True):
#         if not user_query.strip():
#             st.warning("Please enter a valid query.")
#         elif "indexed_documents" not in st.session_state:
#             st.warning("Please index a document in the 'Document Ingestion' tab first before searching.")
#         else:
#             # 1. Instantiate Pipeline with Memory Parameters
#             pipeline = Pipeline(
#                 llm=active_llm,
#                 vector_db=vector_db,
#                 indexed_documents=st.session_state["indexed_documents"],
#                 history_window=history_window,
#                 enable_input_guardrail=enable_input_guardrail,
#                 enable_output_guardrail=enable_output_guardrail,
#                 enable_pre_retrieval=enable_pre_retrieval,
#                 enable_compression=enable_compression,
#                 enable_lost_in_middle=enable_lost_in_middle,
#                 max_context_chars=max_context_chars,
#                 top_n_rerank=top_n_rerank,
#                 reranker_model_name=reranker_model,
#             )

#             citations_to_display = []
#             logs = {}

#             # ==================================================================
#             # BRANCH 1: LIVE TOKEN STREAMING
#             # ==================================================================
#             if enable_streaming:
#                 with st.spinner("Analyzing query, resolving context & searching documents concurrently..."):
#                     is_safe, payload, logs = pipeline.prepare_context(
#                         raw_query=user_query, session_id=session_id
#                     )

#                 if not is_safe:
#                     st.error(payload.get("error", "Query rejected by Input Guardrail."))
#                 else:
#                     st.markdown("### 📝 Response")
                    
#                     # Real-time token streaming
#                     stream_generator = pipeline.stream_answer(payload)
#                     final_answer = st.write_stream(stream_generator)

#                     # Persist turn to SQLite
#                     pipeline.save_turn(
#                         session_id=session_id,
#                         user_query=user_query,
#                         assistant_answer=final_answer,
#                     )

#                     # Stage 7: Post-Stream Faithfulness Audit
#                     if enable_output_guardrail and payload.get("formatted_context"):
#                         with st.spinner("Verifying factual accuracy..."):
#                             output_check = pipeline.audit_output(
#                                 context=payload["formatted_context"],
#                                 answer=final_answer,
#                             )
#                         if output_check:
#                             logs["output_guardrail"] = output_check
#                             if not output_check.is_grounded:
#                                 st.warning("⚠️ **Output Guardrail Alert:** Some claims in the generated response could not be verified from sources:")
#                                 for claim in output_check.hallucinated_claims:
#                                     st.write(f"- {claim}")
#                             else:
#                                 st.success("✅ **Output Verified:** All statements are fully grounded in the provided source context.")

#                     # Resolve citations from streamed text
#                     if hasattr(pipeline.generator, "extract_citations_from_stream"):
#                         citations_to_display = pipeline.generator.extract_citations_from_stream(final_answer, payload)
#                     else:
#                         used_docs = payload.get("used_documents", [])
#                         cited_ids = set(map(int, re.findall(r"\[(\d+)\]", final_answer)))
#                         for cid in sorted(cited_ids):
#                             if 1 <= cid <= len(used_docs):
#                                 doc = used_docs[cid - 1]
#                                 citations_to_display.append({
#                                     "citation_id": cid,
#                                     "source": doc.metadata.get("source", f"Document {cid}"),
#                                     "quoted_snippet": doc.page_content[:250].strip() + "..."
#                                 })

#             # ==================================================================
#             # BRANCH 2: SYNCHRONOUS FULL-PASS EXECUTION (.run automatically saves turn)
#             # ==================================================================
#             else:
#                 with st.spinner("Executing End-to-End RAG Pipeline..."):
#                     result = pipeline.run(raw_query=user_query, session_id=session_id)

#                 st.markdown("### 📝 Grounded Response")
#                 if not result.get("success", False):
#                     st.error(result["answer"])
#                 else:
#                     st.info(result["answer"])

#                 citations_to_display = result.get("citations", [])
#                 logs = result.get("logs", {})

#             # ==================================================================
#             # DISPLAY CITATIONS (Shared across both modes)
#             # ==================================================================
#             if citations_to_display:
#                 st.markdown("### 📚 Source Citations")
#                 for cit in citations_to_display:
#                     cid = getattr(cit, "citation_id", None) or (cit.get("citation_id") if isinstance(cit, dict) else "Ref")
#                     src = getattr(cit, "source", None) or (cit.get("source") if isinstance(cit, dict) else "Document")
#                     snip = getattr(cit, "quoted_snippet", None) or (cit.get("quoted_snippet") if isinstance(cit, dict) else "")
                    
#                     with st.expander(f"[{cid}] Source: {src}"):
#                         st.markdown(f"**Supporting Quote:** *\"{snip}\"*")

#             # ==================================================================
#             # DISPLAY PIPELINE LOGS (Shared across both modes)
#             # ==================================================================
#             st.markdown("---")
#             st.markdown("### 🔍 Pipeline Execution Details")

#             # Stage 0: Conversational Memory Context
#             if logs.get("retrieved_history"):
#                 with st.expander("📜 Stage 0: Conversational Memory Context (From SQLite)"):
#                     st.text(logs["retrieved_history"])

#             # Stage 1: Input Guardrail Log
#             if logs.get("input_guardrail"):
#                 with st.expander("🛡 Stage 1: Input Guardrail Evaluation"):
#                     ig = logs["input_guardrail"]
#                     is_safe_val = getattr(ig, "is_safe", None) or (ig.get("is_safe") if isinstance(ig, dict) else False)
#                     reason_val = getattr(ig, "reason", None) or (ig.get("reason") if isinstance(ig, dict) else "")
#                     st.write(f"**Status:** {'Safe ✅' if is_safe_val else 'Unsafe ❌'}")
#                     st.write(f"**Reason:** {reason_val}")

#             # Stage 2: History-Aware Pre-Retrieval Log
#             if logs.get("pre_retrieval"):
#                 with st.expander("🧠 Stage 2: History-Aware Query Condensation & Multi-Queries"):
#                     pr = logs["pre_retrieval"]
#                     st.write(f"**Rewritten / Standalone Query:** `{pr.rewritten_query}`")
#                     st.write(f"**Multi-Queries:** {pr.multi_queries}")
#                     st.write(f"**Domain Route:** {pr.domain_route}")

#             # Stage 3: Retrieved Documents Log
#             if logs.get("retrieved_documents"):
#                 with st.expander(f"🔎 Stage 3: Async Parallel Retrieved Docs ({len(logs['retrieved_documents'])})"):
#                     for i, doc in enumerate(logs["retrieved_documents"], 1):
#                         st.markdown(f"**Doc Chunk [{i}]**")
#                         st.text(doc.page_content[:300] + "...")

#             # Stage 4: Compressed Documents Log
#             if logs.get("compressed_documents"):
#                 with st.expander(f"✂️ Stage 4: Contextual Compression ({len(logs['compressed_documents'])})"):
#                     for i, doc in enumerate(logs["compressed_documents"], 1):
#                         st.markdown(f"**Compressed Chunk [{i}]**")
#                         st.text(doc.page_content)

#             # Stage 7: Output Guardrail Log (if recorded)
#             if logs.get("output_guardrail"):
#                 with st.expander("🛡️ Stage 7: Output Faithfulness Evaluation"):
#                     og = logs["output_guardrail"]
#                     grounded_val = getattr(og, "is_grounded", None)
#                     if grounded_val is None and isinstance(og, dict):
#                         grounded_val = og.get("is_grounded", False)
                    
#                     claims = getattr(og, "hallucinated_claims", []) or (og.get("hallucinated_claims", []) if isinstance(og, dict) else [])
                    
#                     st.write(f"**Grounded:** {'Passed ✅' if grounded_val else 'Hallucinated ⚠️'}")
#                     if claims:
#                         st.write(f"**Flagged Claims:** {claims}")


import json
import requests
import streamlit as st
import sseclient

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Production Multilingual RAG Client",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Advanced Production RAG System")
st.caption("Streamlit Frontend connected to FastAPI Backend")

# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.header("⚙️ Query Configuration")
enable_streaming = st.sidebar.checkbox("⚡ Enable Live Streaming", value=True)
enable_input_guardrail = st.sidebar.checkbox("Enable Input Guardrail", value=True)
enable_output_guardrail = st.sidebar.checkbox("Enable Output Guardrail", value=True)
enable_pre_retrieval = st.sidebar.checkbox("Enable Pre-Retrieval", value=True)
enable_compression = st.sidebar.checkbox("Enable Context Compression", value=True)
top_n_rerank = st.sidebar.slider("Top-N Reranked Docs", 1, 10, 4)

# Backend health status check
try:
    health = requests.get(f"{API_BASE_URL}/health", timeout=2).json()
    st.sidebar.success(f"Backend Online | Chunks Indexed: {health.get('indexed_chunks', 0)}")
except Exception:
    st.sidebar.error("⚠️ FastAPI backend is offline. Start it with uvicorn.")

# ==============================================================================
# TABS
# ==============================================================================
tab1, tab2 = st.tabs(["📌 Document Ingestion", "💬 RAG Q&A Interface"])

# ------------------------------------------------------------------------------
# TAB 1: INGESTION VIA API
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Extract & Index New Source Documents")
    col1, col2 = st.columns([1, 2])
    with col1:
        source_type = st.selectbox("Source Type", ["YouTube Transcript", "Web Article"])
    with col2:
        source_url = st.text_input("Source URL / Path", value="https://www.youtube.com/")

    if st.button("🚀 Process & Index via Backend", use_container_width=True):
        with st.spinner("Backend is extracting and chunking document..."):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/api/v1/ingest",
                    json={"source_type": source_type, "source_url": source_url},
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"{data['message']} (Total Chunks: {data['total_chunks']})")
                else:
                    st.error(f"Ingestion Error: {resp.text}")
            except Exception as e:
                st.error(f"Failed to reach API: {str(e)}")

# ------------------------------------------------------------------------------
# TAB 2: CHAT & QUERY VIA API
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Chat with your Knowledge Base")

    col_session, col_clear = st.columns([3, 1])
    with col_session:
        session_id = st.text_input("Session ID:", value="user_session_1")
    with col_clear:
        st.write("")
        st.write("")
        if st.button("🗑️ Clear History", use_container_width=True):
            requests.delete(f"{API_BASE_URL}/api/v1/history/{session_id}")
            st.success("Session history cleared on server.")
            st.rerun()

    # Load session history from FastAPI / SQLite
    try:
        hist_resp = requests.get(f"{API_BASE_URL}/api/v1/history/{session_id}", timeout=3)
        if hist_resp.status_code == 200:
            history = hist_resp.json().get("messages", [])
            if history:
                with st.expander(f"📜 Session History ({len(history)} messages)", expanded=False):
                    for msg in history:
                        icon = "🧑‍💻" if msg["role"] == "user" else "🤖"
                        st.markdown(f"**{icon} {msg['role'].title()}:** {msg['message']}")
    except Exception:
        pass

    user_query = st.text_input("Enter your question:")

    if st.button("Search & Generate Answer", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("Please enter a query.")
        else:
            payload = {
                "query": user_query,
                "session_id": session_id,
                "enable_input_guardrail": enable_input_guardrail,
                "enable_output_guardrail": enable_output_guardrail,
                "enable_pre_retrieval": enable_pre_retrieval,
                "enable_compression": enable_compression,
                "top_n_rerank": top_n_rerank,
            }

            # --- BRANCH A: STREAMING VIA FASTAPI SSE ---
            if enable_streaming:
                st.markdown("### 📝 Grounded Response")
                response = requests.post(
                    f"{API_BASE_URL}/api/v1/query/stream",
                    json=payload,
                    stream=True,
                    headers={"Accept": "text/event-stream"},
                )

                if response.status_code != 200:
                    st.error(f"Server error: {response.text}")
                else:
                    client = sseclient.SSEClient(response)
                    text_placeholder = st.empty()
                    collected_tokens = []
                    metadata = {}

                    for event in client.events():
                        if event.data == "[DONE]":
                            break
                        try:
                            msg = json.loads(event.data)
                            if msg.get("type") == "token":
                                collected_tokens.append(msg["content"])
                                text_placeholder.markdown("".join(collected_tokens))
                            elif msg.get("type") == "error":
                                st.error(msg["message"])
                            elif msg.get("type") == "guardrail_alert":
                                st.warning(f"⚠️ Guardrail Alert: {msg.get('claims')}")
                            elif msg.get("type") == "metadata":
                                metadata = msg
                        except json.JSONDecodeError:
                            continue

                    # Render Citations from backend stream metadata
                    citations = metadata.get("citations", [])
                    if citations:
                        st.markdown("### 📚 Source Citations")
                        for cit in citations:
                            with st.expander(f"[{cit['citation_id']}] Source: {cit['source']}"):
                                st.markdown(f"**Supporting Quote:** *\"{cit['quoted_snippet']}\"*")

            # --- BRANCH B: ONE-SHOT QUERY VIA FASTAPI ---
            else:
                with st.spinner("Waiting for backend pipeline..."):
                    resp = requests.post(f"{API_BASE_URL}/api/v1/query", json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.markdown("### 📝 Grounded Response")
                        st.info(data["answer"])

                        if data.get("rewritten_query"):
                            st.caption(f"🔍 *Standalone Search Query:* `{data['rewritten_query']}`")

                        if data.get("citations"):
                            st.markdown("### 📚 Source Citations")
                            for cit in data["citations"]:
                                with st.expander(f"[{cit['citation_id']}] Source: {cit['source']}"):
                                    st.markdown(f"**Supporting Quote:** *\"{cit['quoted_snippet']}\"*")
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text}")

                        