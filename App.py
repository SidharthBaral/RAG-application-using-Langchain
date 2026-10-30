import os
import re
import streamlit as st
from dotenv import load_dotenv

from Models.ResponseGeneration import TextGeneration
from Models.EmbeddingModel import MultilingualEmbeddingGenerator
from Indexing.data_extraction.DataExtractorFactory import DataExtractorFactory
from Indexing.preprocessor.TextPreprocessor import TextPreprocessor
from Indexing.preprocessor.VectorDB import VectorDB
from RAGPipeline import Pipeline

load_dotenv()

st.set_page_config(
    page_title="Production Multilingual RAG System",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Advanced Production RAG System")
st.markdown("Enterprise-grade Retrieval-Augmented Generation pipeline with Guardrails, Hybrid Search, and Citations.")

# ==============================================================================
# SIDEBAR CONFIGURATION OPTIONS
# ==============================================================================
st.sidebar.header("⚙️ Pipeline Configuration")

# --- 1. LLM Settings ---
with st.sidebar.expander("🤖 LLM Settings", expanded=True):
    llm_provider = st.selectbox(
        "LLM Provider",
        options=["groq", "openai", "anthropic", "ollama"],
        index=0,
    )
    
    default_models = {
        "groq": "openai/gpt-oss-120b",
        "openai": "gpt-4o",
        "anthropic": "claude-3-5-sonnet-20241022",
        "ollama": "llama3",
    }
    
    llm_model_name = st.text_input("Model ID", value=default_models.get(llm_provider, ""))
    user_llm_api_key = st.text_input("Custom LLM API Key (Optional)", type="password")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    verify_ssl = st.checkbox("Verify SSL Certificates", value=False)

# --- 2. Embedding & Vector DB Settings ---
with st.sidebar.expander("📊 Embedding & Vector DB", expanded=False):
    embed_provider = st.selectbox(
        "Embedding Provider",
        options=["huggingface", "huggingface_api", "openai", "cohere"],
        index=0,
    )
    
    embed_model_name = st.text_input(
        "Embedding Model", 
        value="intfloat/multilingual-e5-small" if embed_provider == "huggingface" else ""
    )
    user_embed_api_key = st.text_input("Custom Embedding API Key (Optional)", type="password")
    compute_device = st.selectbox("Compute Device (Local)", options=["cpu", "cuda", "mps"], index=0)
    
    # ChromaDB collection names must not contain spaces
    collection_name = st.text_input("Chroma DB Collection Name", value="youtube_video_transcripts")

# --- 3. Retrieval & Guardrail Controls ---
with st.sidebar.expander("🛠 Retrieval & Guardrails", expanded=False):
    enable_streaming = st.checkbox("⚡ Enable Live Token Streaming", value=True)
    enable_input_guardrail = st.checkbox("Enable Input Guardrail", value=True)
    enable_output_guardrail = st.checkbox("Enable Output Guardrail", value=True)
    enable_pre_retrieval = st.checkbox("Enable Pre-Retrieval (Rewriting)", value=True)
    enable_compression = st.checkbox("Enable Contextual Compression", value=True)
    enable_lost_in_middle = st.checkbox("Enable Lost-In-The-Middle Reordering", value=True)
    top_n_rerank = st.slider("Top-N Reranked Documents", min_value=1, max_value=10, value=4)
    max_context_chars = st.number_input("Max Context Character Limit", value=4000, step=500)
    reranker_model = st.text_input("Reranker Model", value="BAAI/bge-reranker-base")


# ==============================================================================
# INITIALIZATION & CACHING FUNCTIONS
# ==============================================================================
@st.cache_resource
def get_llm_instance(provider, model_name, api_key, temp, ssl_verify):
    return TextGeneration(
        provider=provider,
        model_name=model_name if model_name else None,
        api_key=api_key if api_key else None,
        temperature=temp,
        verify_ssl=ssl_verify,
    ).llm


@st.cache_resource
def get_embedder_instance(provider, model_name, api_key, device):
    return MultilingualEmbeddingGenerator(
        provider=provider,
        model_name=model_name if model_name else None,
        api_key=api_key if api_key else None,
        device=device,
    )


# Instantiate Embedder & LLM
try:
    active_llm = get_llm_instance(
        llm_provider, llm_model_name, user_llm_api_key, temperature, verify_ssl
    )
    active_embedder_service = get_embedder_instance(
        embed_provider, embed_model_name, user_embed_api_key, compute_device
    )
    
    vector_db = VectorDB(
        embedding_function=active_embedder_service.embedder,
        collection_name=collection_name,
        persist_directory="./my_chroma_data",
    )
except Exception as e:
    st.error(f"Initialization Error: {str(e)}")
    st.stop()


# ==============================================================================
# MAIN APPLICATION TABS
# ==============================================================================
tab1, tab2 = st.tabs(["📌 Document Ingestion", "💬 RAG Q&A Interface"])

# ------------------------------------------------------------------------------
# TAB 1: DOCUMENT INGESTION
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Extract & Index New Source Documents")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        source_type = st.selectbox(
            "Select Data Source Type",
            options=["YouTube Transcript", "Web Article"],
            index=0,
        )
    with col2:
        source_url = st.text_input(
            "Enter Source URL / Path",
            value="https://www.youtube.com/",
        )

    if st.button("🚀 Process & Index Source", use_container_width=True):
        with st.spinner("Extracting text and chunking document..."):
            try:
                # 1. Dynamic Extractor Selection via Factory
                extractor = DataExtractorFactory.get_extractor(source_type)
                
                # 2. Text Preprocessing & Chunking
                preprocessor = TextPreprocessor(embeddings=active_embedder_service.embedder)
                documents = preprocessor.process(extractor, source=source_url)
                
                # 3. Add to Vector DB
                vector_db.add_documents(documents)
                
                # 4. Save processed documents in session state for BM25 hybrid search
                st.session_state["indexed_documents"] = documents
                st.success(f"Successfully processed & indexed {len(documents)} document chunks into '{collection_name}'!")
            
            except Exception as e:
                st.error(f"Ingestion Failed: {str(e)}")

# ------------------------------------------------------------------------------
# TAB 2: RAG Q&A INTERFACE
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Ask Questions Against Your Knowledge Base")

    user_query = st.text_input(
        "Enter your question:",
        placeholder="What did he say about building agents from scratch?",
    )

    if st.button("Search & Generate Answer", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("Please enter a valid query.")
        elif "indexed_documents" not in st.session_state:
            st.warning("Please index a document in the 'Document Ingestion' tab first before searching.")
        else:
            # 1. Instantiate Pipeline
            pipeline = Pipeline(
                llm=active_llm,
                vector_db=vector_db,
                indexed_documents=st.session_state["indexed_documents"],
                enable_input_guardrail=enable_input_guardrail,
                enable_output_guardrail=enable_output_guardrail,
                enable_pre_retrieval=enable_pre_retrieval,
                enable_compression=enable_compression,
                enable_lost_in_middle=enable_lost_in_middle,
                max_context_chars=max_context_chars,
                top_n_rerank=top_n_rerank,
                reranker_model_name=reranker_model,
            )

            citations_to_display = []
            logs = {}

            # ==================================================================
            # BRANCH 1: LIVE TOKEN STREAMING
            # ==================================================================
            if enable_streaming:
                # Run Stages 1 through 5 inside the spinner
                with st.spinner("Analyzing query and searching documents concurrently..."):
                    is_safe, payload, logs = pipeline.prepare_context(user_query)

                if not is_safe:
                    st.error(payload.get("error", "Query rejected by Input Guardrail."))
                else:
                    st.markdown("### 📝 Grounded Response")
                    
                    # Real-time token streaming (First token renders in ~0.3-0.5s)
                    stream_generator = pipeline.stream_answer(payload)
                    final_answer = st.write_stream(stream_generator)

                    # Stage 7: Post-Stream Faithfulness Audit
                    if enable_output_guardrail and payload.get("formatted_context"):
                        with st.spinner("Verifying factual accuracy..."):
                            output_check = pipeline.audit_output(
                                context=payload["formatted_context"],
                                answer=final_answer
                            )
                        if output_check:
                            logs["output_guardrail"] = output_check
                            if not output_check.is_grounded:
                                st.warning("⚠️ **Output Guardrail Alert:** Some claims in the generated response could not be verified from sources:")
                                for claim in output_check.hallucinated_claims:
                                    st.write(f"- {claim}")
                            else:
                                st.success("✅ **Output Verified:** All statements are fully grounded in the provided source context.")

                    # Resolve citations from streamed text
                    if hasattr(pipeline.generator, "extract_citations_from_stream"):
                        citations_to_display = pipeline.generator.extract_citations_from_stream(final_answer, payload)
                    else:
                        used_docs = payload.get("used_documents", [])
                        cited_ids = set(map(int, re.findall(r"\[(\d+)\]", final_answer)))
                        for cid in sorted(cited_ids):
                            if 1 <= cid <= len(used_docs):
                                doc = used_docs[cid - 1]
                                citations_to_display.append({
                                    "citation_id": cid,
                                    "source": doc.metadata.get("source", f"Document {cid}"),
                                    "quoted_snippet": doc.page_content[:250].strip() + "..."
                                })

            # ==================================================================
            # BRANCH 2: SYNCHRONOUS FULL-PASS EXECUTION (Original Behavior)
            # ==================================================================
            else:
                with st.spinner("Executing End-to-End RAG Pipeline..."):
                    result = pipeline.run(user_query)

                st.markdown("### 📝 Grounded Response")
                if not result.get("success", False):
                    st.error(result["answer"])
                else:
                    st.info(result["answer"])

                citations_to_display = result.get("citations", [])
                logs = result.get("logs", {})

            # ==================================================================
            # DISPLAY CITATIONS (Shared across both modes)
            # ==================================================================
            if citations_to_display:
                st.markdown("### 📚 Source Citations")
                for cit in citations_to_display:
                    cid = getattr(cit, "citation_id", None) or (cit.get("citation_id") if isinstance(cit, dict) else "Ref")
                    src = getattr(cit, "source", None) or (cit.get("source") if isinstance(cit, dict) else "Document")
                    snip = getattr(cit, "quoted_snippet", None) or (cit.get("quoted_snippet") if isinstance(cit, dict) else "")
                    
                    with st.expander(f"[{cid}] Source: {src}"):
                        st.markdown(f"**Supporting Quote:** *\"{snip}\"*")

            # ==================================================================
            # DISPLAY PIPELINE LOGS (Shared across both modes)
            # ==================================================================
            st.markdown("---")
            st.markdown("### 🔍 Pipeline Execution Details")

            # Stage 1: Input Guardrail Log
            if logs.get("input_guardrail"):
                with st.expander("🛡 Stage 1: Input Guardrail Evaluation"):
                    ig = logs["input_guardrail"]
                    is_safe_val = getattr(ig, "is_safe", None) or (ig.get("is_safe") if isinstance(ig, dict) else False)
                    reason_val = getattr(ig, "reason", None) or (ig.get("reason") if isinstance(ig, dict) else "")
                    st.write(f"**Status:** {'Safe ✅' if is_safe_val else 'Unsafe ❌'}")
                    st.write(f"**Reason:** {reason_val}")

            # Stage 2: Pre-Retrieval Log
            if logs.get("pre_retrieval"):
                with st.expander("🧠 Stage 2: Pre-Retrieval Rewriting & Multi-Queries"):
                    pr = logs["pre_retrieval"]
                    st.write(f"**Rewritten Query:** {pr.rewritten_query}")
                    st.write(f"**Multi-Queries:** {pr.multi_queries}")
                    st.write(f"**Domain Route:** {pr.domain_route}")

            # Stage 3: Retrieved Documents Log
            if logs.get("retrieved_documents"):
                with st.expander(f"🔎 Stage 3: Async Parallel Retrieved Docs ({len(logs['retrieved_documents'])})"):
                    for i, doc in enumerate(logs["retrieved_documents"], 1):
                        st.markdown(f"**Doc Chunk [{i}]**")
                        st.text(doc.page_content[:300] + "...")

            # Stage 4: Compressed Documents Log
            if logs.get("compressed_documents"):
                with st.expander(f"✂️ Stage 4: Contextual Compression ({len(logs['compressed_documents'])})"):
                    for i, doc in enumerate(logs["compressed_documents"], 1):
                        st.markdown(f"**Compressed Chunk [{i}]**")
                        st.text(doc.page_content)

            # Stage 7: Output Guardrail Log (if recorded)
            if logs.get("output_guardrail"):
                with st.expander("🛡️ Stage 7: Output Faithfulness Evaluation"):
                    og = logs["output_guardrail"]
                    grounded_val = getattr(og, "is_grounded", None)
                    if grounded_val is None and isinstance(og, dict):
                        grounded_val = og.get("is_grounded", False)
                    
                    claims = getattr(og, "hallucinated_claims", []) or (og.get("hallucinated_claims", []) if isinstance(og, dict) else [])
                    
                    st.write(f"**Grounded:** {'Passed ✅' if grounded_val else 'Hallucinated ⚠️'}")
                    if claims:
                        st.write(f"**Flagged Claims:** {claims}")