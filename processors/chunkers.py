from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from processors.html_chunking import get_html_chunks


class Chunker:
    def __init__(self, chunker: str = "mdChunks", chunk_size: int = 300, file_path: str = None):
        self.chunker = chunker
        self.file_path = file_path
        self.chunk_size = int(chunk_size)

    def create_chunks(self):
        if self.chunker == "mdChunking":
            return self.create_mdchunks()
        elif self.chunker == "charChunking":
            return self.create_char_chunks()
        elif self.chunker == "recursiveChunking":
            return self.create_recursive_chunks()
        elif self.chunker == "tokenChunking":
            return self.create_token_chunks()
        elif self.chunker == "mdFixedSizeChunking":
            return self.create_mdfixedsizechunks()
        elif self.chunker == "fixedChunking":
            return self.create_fixedchunks()
        elif self.chunker == "pageChunking":
            return self.create_pagechunks()
        elif self.chunker == "semanticChunking":
            return self.create_semanticchunks()
        elif self.chunker == "htmlChunking":
            return self.create_htmlchunks()

    def create_mdchunks(self):    
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        with open(self.file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        chunks = markdown_splitter.split_text(content)
        return chunks

    def create_char_chunks(self):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_size*0.1,
            length_function=len,
            is_separator_regex=False,
        )

        with open(self.file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        chunks = text_splitter.split_text(content)
        return chunks

    def create_recursive_chunks(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_size*0.1,
            length_function=len,
            is_separator_regex=False,
        )

        with open(self.file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        chunks = text_splitter.split_text(content)
        return chunks

    def create_token_chunks(self):
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=self.chunk_size, chunk_overlap=self.chunk_size*0.1
        )

        with open(self.file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        chunks = text_splitter.split_text(content)
        return chunks

    def create_mdfixedsizechunks(self):
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        with open(self.file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        md_header_splits = markdown_splitter.split_text(content)

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_size*0.1,
            length_function=len,
            is_separator_regex=False,
        )

        splits = text_splitter.split_documents(md_header_splits)
        return splits

    def create_fixedchunks(self):
        text_splitter = CharacterTextSplitter(
            separator="",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_size*0.1,
            length_function=len,
            is_separator_regex=False,
        )

        with open(self.file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        chunks = text_splitter.split_text(content)
        return chunks

    def create_pagechunks(self):
        return None

    def create_semanticchunks(self):
        return None
    
    def create_htmlchunks(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        return get_html_chunks(content, max_tokens=200, is_clean_html=True, attr_cutoff_len=25)