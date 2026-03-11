# Tika Document Processor

## Overview

This module processes various document formats (PDF, DOCX, PPTX, etc.) using Apache Tika via LLMSherpa. It extracts text, handles OCR for scanned pages, extracts images, and generates captions using a VLM.

## Features

- Multi-format document processing (PDF, DOCX, PPTX, etc.)
- OCR support for scanned pages
- Image extraction and VLM-based captioning
- Elasticsearch indexing with hybrid search support
- Metadata extraction via ExifTool

## Prerequisites

- Python 3.8+
- Docker (for LLMSherpa/NLM Ingestor)
- ExifTool (for metadata extraction)

## Setup

```bash
# Start the NLM Ingestor service
docker pull ghcr.io/nlmatics/nlm-ingestor:v0.1.4
docker run -p 5009:5001 ghcr.io/nlmatics/nlm-ingestor:v0.1.4
```
