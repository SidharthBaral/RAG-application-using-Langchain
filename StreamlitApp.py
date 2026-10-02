import os
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
    collection_name = st.text_input("Chroma DB Collection Name", value="youtube-video-transcripts")

# --- 3. Retrieval & Guardrail Controls ---
with st.sidebar.expander("🛠️️ Retrieval & Guardrails", expanded=False):
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
            value="https://www.youtube.com",
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
            with st.spinner("Executing End-to-End RAG Pipeline..."):
                # 1. Instantiate Unified Master RAG Pipeline
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

                # 2. Execute RAG Chain
                result = pipeline.run(user_query)

            # 3. Render Final Grounded Answer
            st.markdown("### 📝 Grounded Response")
            st.info(result["answer"])

            # 4. Render Structured Citations
            if result.get("citations"):
                st.markdown("### 📚 Source Citations")
                for cit in result["citations"]:
                    with st.expander(f"[{cit.citation_id}] Source: {cit.source}"):
                        st.markdown(f"**Supporting Quote:** *\"{cit.quoted_snippet}\"*")

            # 5. Render Pipeline Logs & Intermediate Stage Expanders
            st.markdown("---")
            st.markdown("### 🔍 Pipeline Execution Details")
            logs = result.get("logs", {})

            if logs.get("input_guardrail"):
                with st.expander("🛡️️ Stage 1: Input Guardrail Evaluation"):
                    ig = logs["input_guardrail"]
                    st.write(f"**Status:** {'Safe ✅' if ig.is_safe else 'Unsafe ❌'}")
                    st.write(f"**Reason:** {ig.reason}")

            if logs.get("pre_retrieval"):
                with st.expander("🧠 Stage 2: Pre-Retrieval Rewriting & Multi-Queries"):
                    pr = logs["pre_retrieval"]
                    st.write(f"**Rewritten Query:** {pr.rewritten_query}")
                    st.write(f"**Multi-Queries:** {pr.multi_queries}")
                    st.write(f"**Domain Route:** {pr.domain_route}")

            if logs.get("retrieved_documents"):
                with st.expander(f"🔎 Stage 3: Hybrid Search & Reranked Docs ({len(logs['retrieved_documents'])})"):
                    for i, doc in enumerate(logs["retrieved_documents"], 1):
                        st.markdown(f"**Doc Chunk [{i}]**")
                        st.text(doc.page_content[:300] + "...")

            if logs.get("compressed_documents"):
                with st.expander(f"✂️ Stage 4: Post-Retrieval Compressed Context ({len(logs['compressed_documents'])})"):
                    for i, doc in enumerate(logs["compressed_documents"], 1):
                        st.markdown(f"**Compressed Chunk [{i}]**")
                        st.text(doc.page_content)

            if logs.get("output_guardrail"):
                with st.expander("🛡️ Stage 7: Output Faithfulness Evaluation"):
                    og = logs["output_guardrail"]
                    st.write(f"**Grounded:** {'Passed ✅' if og.is_grounded else 'Hallucinated ⚠️'}")
                    if og.hallucinated_claims:
                        st.write(f"**Flagged Claims:** {og.hallucinated_claims}")