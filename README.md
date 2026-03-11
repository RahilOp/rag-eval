# RAG Benchmark Toolkit

## Overview

A comprehensive toolkit for evaluating and benchmarking Retrieval-Augmented Generation (RAG) pipelines. This project systematically tests combinations of document processors, chunking strategies, embedding models, retrieval methods, rerankers, and LLM generators to identify the optimal RAG configuration for your data.

## Architecture

```
Documents (PDF, DOCX, etc.)
    |
    v
[Document Processing] ---- Marker / Tika / Docling / Unstructured
    |
    v
[Chunking Strategy] ------- Markdown / Recursive / Character / Token / HTML / Fixed / Semantic
    |
    v
[Embedding Generation] ---- BGE-m3 / E5 / sentence-transformers / OpenAI
    |
    v
[Elasticsearch Indexing] --- Multiple indices with different configs
    |
    v
[Retrieval Testing] ------- BM25 / KNN / Hybrid / Hybrid-CC
    |                        + optional reranking (mMiniLM cross-encoder)
    v
[Evaluation Metrics] ------ MRR / Context Precision (RAGAS) / ROUGE / Semantic Similarity
    |
    v
[Generation Testing] ------ LLM response generation + quality scoring
    |
    v
[Best Config Output] ------ JSON with optimal parameters per index
```

## Features

- **4 Document Processors**: Marker, Tika, Docling, Unstructured
- **9 Chunking Strategies**: Markdown (3-level), Markdown + Fixed Size, Character, Recursive, Token, Fixed, Semantic, HTML DOM, Page
- **4 Search Methods**: BM25, KNN, Hybrid (weighted RRF), Hybrid Combined Query
- **Reranking**: Cross-encoder reranking with mMiniLM
- **Multiple Metrics**: MRR, Context Precision, ROUGE, Semantic Similarity
- **Generation Evaluation**: LLM response quality scoring via E5 embeddings
- **YAML-driven Configuration**: Easy to configure and extend

## Installation

### Prerequisites

- Python 3.10+
- CUDA >= 11.7 (for GPU-accelerated models)
- Docker (for Tika/LLMSherpa service)
- Elasticsearch 8.x

### Setup

```bash
git clone https://github.com/yourusername/rag-benchmark-toolkit.git
cd rag-benchmark-toolkit
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your credentials

# Start Elasticsearch (if not already running)
docker run -d --name elasticsearch -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" elasticsearch:8.14.0

# Start NLM Ingestor (required for Tika processor)
docker pull ghcr.io/nlmatics/nlm-ingestor:v0.1.4
docker run -p 5009:5001 ghcr.io/nlmatics/nlm-ingestor:v0.1.4
```

## Usage

### Configuration

Edit `config.yaml` to define your test parameters:

```yaml
search_methods:
  - bm25
  - hybrid
  - knn

retriever_weights:
  - [0.7, 0.3]
  - [0.5, 0.5]

chunk_method:
  - mdChunking
  - recursiveChunking

chunk_size:
  - 300
  - 500
  - 1000

embedder_name: "BAAI/bge-m3"
```

### Running

```bash
# Run the full pipeline (indexing + retrieval testing)
python main.py

# Or test existing indices only (set only_test_index: true in config)
python main.py
```

### Results

Results are saved to:
- `best_configs.json` - Best performing configuration per index
- `results/` - Detailed per-query metrics and LLM responses

## Project Structure

```
.
├── main.py                             # Pipeline orchestrator
├── config.yaml                         # Test configuration
├── config_logger.py                    # Logging setup
├── requirements.txt
│
├── processors/                         # Document processing & indexing
│   ├── chunkers.py                     # 9 chunking strategies
│   ├── html_chunking.py               # HTML DOM-based chunking
│   ├── embedder.py                     # HuggingFace embedding handler
│   ├── elasticsearch_indexer.py        # Elasticsearch indexing
│   ├── marker/                         # PDF -> Markdown + VLM image captions
│   │   ├── pipeline.py                 # Marker processing pipeline
│   │   ├── pdf_converter.py            # Marker PDF conversion
│   │   ├── vlm.py                      # VLM image captioning
│   │   ├── vlm_loader.py              # VLM model loader
│   │   └── image_utils.py             # Image encoding utilities
│   └── tika/                           # Multi-format document processing
│       ├── processor.py                # Tika/LLMSherpa processor
│       ├── file_processor.py           # File processing orchestrator
│       ├── image_extractor.py          # PDF image extraction
│       ├── image_captioner.py          # VLM image captioning
│       └── ocr.py                      # OCR pipeline
│
├── retrieval/                          # Search & retrieval
│   ├── elasticsearch_retriever.py      # BM25 / KNN / Hybrid / RRF search
│   ├── search.py                       # Search orchestration + reranking
│   ├── result_collector.py             # Batch search & CSV output
│   └── embeddings.py                   # OpenAI-compatible embedding client
│
├── evaluation/                         # Metrics & evaluation
│   ├── pipeline.py                     # End-to-end evaluation pipeline
│   ├── mrr.py                          # Mean Reciprocal Rank
│   ├── context_precision.py            # Context Precision (RAGAS)
│   ├── generation_metric.py            # E5 semantic similarity scoring
│   └── custom_metrics.py              # Custom weighted metrics
│
└── data/
    ├── documents/                      # Source PDFs for indexing
    └── ground_truth/                   # Q&A datasets for evaluation
        └── qa_dataset.csv
```

## Environment Variables

See `.env.example` for all required variables. Key ones:

| Variable | Description |
|----------|-------------|
| `ES_URL` | Elasticsearch URL |
| `ES_USER` / `ES_PASSWORD` | Elasticsearch credentials |
| `EMBEDDING_BASE_URL` | OpenAI-compatible embedding service |
| `GPT4O_MINI_OPENAI_*` | LLM for generation testing |
| `VLM_BASE_URL` | VLM service for image captioning |
| `AZURE_OPENAI_*` | Azure OpenAI for RAGAS evaluation |

## License

MIT
