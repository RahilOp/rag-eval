import os
import pandas as pd
from .processor import TikaProcessor
from .embedding import EmbeddingHandler
from .ocr import OCRPipeline
import json
from flask import Flask
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch
from .image_extractor import extract_images_from_pdf
from .image_captioner import generate_response_qwen
from .vlm_loader import load_qwen_vl
from tqdm import tqdm


class FileProcessor:
    def __init__(self, folder_path, output_type, index_name=None, use_vlm=False):
        self.folder_path = folder_path
        self.output_type = output_type
        self.index_name = index_name
        self.use_vlm = use_vlm
        self.chunk_list = []
        self.link_list = []
        self.file_name = []
        self.metadata_list = []

    def process_file(self):
        tika_processor = TikaProcessor(apply_ocr=True)
        ocr_pipeline = OCRPipeline(self.folder_path)
        
        link_data_path = self.folder_path + '/link_data.json'
        link_data = tika_processor.load_json(link_data_path)
        files = tika_processor.get_files(self.folder_path)

        for file_name in files:
            file_path = os.path.join(self.folder_path, file_name)

            csv_file = tika_processor.generate_metadata_csv(file_path)
            csv_data = pd.read_csv(csv_file)
            doc = tika_processor.read_pdf_with_llmsherpa(file_path)
            
            if self.output_type == "csv":
                self.convert_to_csv(doc, file_name, file_path, csv_data, link_data, ocr_pipeline, tika_processor)
            elif self.output_type == "json":
                self.convert_to_json(doc, file_name, file_path, csv_data, link_data, ocr_pipeline, tika_processor)
            elif self.output_type == "markdown":
                self.convert_to_markdown()
            elif self.output_type == "txt":
                self.convert_to_txt(doc)
            elif self.output_type == "html":
                self.convert_to_html(doc)
            elif self.output_type == "embedding":
                self.convert_to_embedding(doc, file_name, file_path, csv_data, link_data, ocr_pipeline, tika_processor)
            else:
                raise ValueError("Invalid output type")
        
            if os.path.exists(csv_file):
                os.remove(csv_file)

    def update_metadata(self, content, link=None, metadatas=None, file_name=None):
        self.chunk_list.append(content)
        self.link_list.append(link)
        self.file_name.append(file_name)
        self.metadata_list.append(metadatas)

    def convert_to_csv(self, doc, file_name, file_path, csv_data, link_data, ocr_pipeline, tika_processor):

        # extract images from pdf
        if self.use_vlm:
            image_list = extract_images_from_pdf(file_path)

        sum_text = ''
        page_flag = 0
        for chunk in doc.chunks():
            # count length of characters on a page to check if page needs ocr
            page_len = tika_processor.get_page_length(doc.json, chunk.page_idx)
            
            # initiate ocr pipeline if total characters in page less than threshold (in future threshold to be
            # calculated based on average length of characters of entire doc)
            if page_len < 100:
                content, link, metadatas = ocr_pipeline.perform_ocr_chunk(file_name, file_path, link_data, chunk)
                self.update_metadata(content, link, metadatas, file_name)
            else:
                text = chunk.to_context_text()
                length = tika_processor.length_after_first_newline(text)
                if length < 40 or len(sum_text) < 300:
                    if len(sum_text) == 0:
                        sum_text += text
                    else:
                        sum_text = sum_text + " " + chunk.to_text()
                
                if len(sum_text) >= 300:
                    link, metadatas = tika_processor.create_metadata(csv_data, file_name, chunk, link_data)
                    self.update_metadata(sum_text, link, metadatas, file_name)
                    sum_text = ''

            # generate captions and insert chunks
            if self.use_vlm:
                for image_file in image_list:
                    # print("Went here", image_file[0][0], chunk.page_idx)
                    if int(image_file[0][0]) == int(chunk.page_idx):
                        # print("Yooooooooooooo")
                        caption = generate_response_qwen(image_file[1]) # try adding heading hierarchy info
                        print("generated caption for", image_file[0])
                        self.update_metadata(caption)
                        image_list.remove(image_file)


        df = pd.DataFrame({
            'chunks': self.chunk_list,
            'links': self.link_list,
            'file_name': self.file_name,
            'metadatas': self.metadata_list
        })
        df.to_excel(self.folder_path + '/output.xlsx')

        return None

    # def convert_to_csv(self, doc, file_name, file_path, csv_data, link_data, ocr_pipeline, tika_processor):
    #     sum_text = ''
    #     for i in range(len(doc.chunks())):
    #         chunk = doc.chunks()[i]

    #         # count length of characters on a page to check if page needs ocr
    #         page_len = tika_processor.get_page_length(doc.json, chunk.page_idx)

    #         # initiate ocr pipeline if total characters in page less than threshold (in future threshold to be
    #         # calculated based on average length of characters of entire doc)
    #         if page_len < 100:
    #             content, link, metadatas = ocr_pipeline.perform_ocr_chunk(file_name, file_path, link_data, chunk)
    #             self.update_metadata(content, link, metadatas, file_name)
    #         else:
    #             text = chunk.to_context_text()
    #             length = tika_processor.length_after_first_newline(text)
    #             if length < 40 or len(sum_text) < 300:
    #                 if len(sum_text) == 0:
    #                     sum_text += text
    #                 else:
    #                     sum_text = sum_text + " " + chunk.to_text()
                
    #             if len(sum_text) >= 300:
    #                 link, metadatas = tika_processor.create_metadata(csv_data, file_name, chunk, link_data)
    #                 self.update_metadata(sum_text, link, metadatas, file_name)
    #                 sum_text = ''            

    def convert_to_json(self, doc, file_name, file_path, csv_data, link_data, ocr_pipeline, tika_processor):
        json_out = doc.json
        less_than_threshold_page = -1
        for i in range(len(json_out)):
            if json_out[i]['page_idx'] == less_than_threshold_page:
                continue
            page_len = tika_processor.get_page_length(json_out, json_out[i]['page_idx'])
            if page_len < 100:
                less_than_threshold_page = json_out[i]['page_idx']
                content, link, metadatas = ocr_pipeline.perform_ocr_json(file_name, file_path, link_data, json_out[i])
                if content:
                    if 'sentences' in json_out[i].keys():
                        del json_out[i]['sentences']
                        json_out[i]['sentences'] = [content]
                        json_out[i]['link'] = link
                        json_out[i]['metadatas'] = metadatas
                    else:
                        json_out[i]['sentences'] = [content]
                        json_out[i]['link'] = link
                        json_out[i]['metadatas'] = metadatas
        
        with open(self.folder_path + '/output.json', 'w', encoding='utf-8') as f:
            json.dump(json_out, f, indent=4, ensure_ascii=False)

        # later modify
        return None


    def to_markdown(self, section, include_children=False, recurse=False):
        """
        Converts a section to markdown format. If include_children is True, then the markdown of the children is also included.
        If recurse is True, then the markdown of the children's children are also included.

        Parameters
        ----------
        section: Section
            The section to be converted to markdown
        include_children: bool
            If True, include the markdown of children sections
        recurse: bool
            If True, include the markdown of children recursively
        """
        # Create Markdown header based on the section's level
        markdown = f"{'#' * (section.level + 1)} {section.title.strip()} \n"

        # Add the text of the section
        markdown += section.to_text(include_children=recurse, recurse=recurse)

        # if include_children:
        #     for child in section.children:
        #         markdown += self.to_markdown(child, include_children=recurse, recurse=recurse)
        
        # later modify
        return markdown
    
    def convert_to_markdown(self, doc):
        """
        Returns markdown of a document by iterating through all the sections.
        Parameters
        ----------
        doc: Document
            The document object to convert to markdown
        """
        markdown = ""
        for section in doc.sections():
            markdown += self.to_markdown(section, include_children=True, recurse=True) + "\n"

        # Save the Markdown content to the specified file
        with open(self.folder_path + '/output.md', 'w', encoding='utf-8') as md_file:
            md_file.write(markdown)
        

    def convert_to_txt(self, doc):
        # TODO: add vlm support
        return doc.to_text()

    def convert_to_html(self, doc):
        # TODO: add vlm support
        html_out = doc.to_html()

        file_path = self.folder_path + '/output.html'
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_out)
        

    def convert_to_embedding(self, doc, file_name, file_path, csv_data, link_data, ocr_pipeline, tika_processor):
        # elasticsearch indexing
        index_name = self.index_name
        es_url = os.getenv("ES_URL", "http://localhost:9200")

        # Configure Elasticsearch client with a custom timeout
        es_client = Elasticsearch(
            hosts=[es_url], 
            timeout=60,  # Timeout set to 30 seconds
        )

        es_user = os.getenv("ES_USER", "elastic")
        es_password = os.getenv("ES_PASSWORD", "")

        # huggingface embedder model
        embeddings = EmbeddingHandler(
            model_name="BAAI/bge-m3",
            device="cuda"
        )

        # extract images from pdf
        if self.use_vlm:
            image_list = extract_images_from_pdf(file_path)        

        sum_text = ''
        for chunk in tqdm(doc.chunks()):
            # count length of characters on a page to check if page needs ocr
            page_len = tika_processor.get_page_length(doc.json, chunk.page_idx)
            
            # initiate ocr pipeline if total characters in page less than threshold (in future threshold to be
            # calculated based on average length of characters of entire doc)
            if page_len < 100:
                content, link, metadatas = ocr_pipeline.perform_ocr_chunk(file_name, file_path, link_data, chunk)
                self.update_metadata(content, link, metadatas, file_name)
                db = ElasticsearchStore.from_texts(
                    es_connection=es_client,
                    # es_url=es_url,
                    index_name=index_name,
                    embedding=embeddings.get_embeddings(),
                    texts=[content],
                    metadatas=[{'file_name': file_name, 'page_num': chunk.page_idx, 'bbox': chunk.bbox, 'link': link}],
                    # es_user=es_user,
                    # es_password=es_password,
                    es_api_key=os.getenv("ES_API_KEY"),
                    # query_field='content',
                    # distance_strategy="COSINE",
                    strategy=ElasticsearchStore.ApproxRetrievalStrategy(hybrid=True),
                    # vector_query_field='vector'
                    )
            else:
                text = chunk.to_context_text()
                length = tika_processor.length_after_first_newline(text)
                link = 'none'
                # if length < 40 or len(sum_text) < 2000:
                #     if len(sum_text) == 0:
                #         sum_text += text
                #     else:
                #         sum_text = sum_text + " " + chunk.to_text()
                if len(sum_text) < 2000:
                    if len(sum_text) == 0:
                        sum_text += text
                    else:
                        sum_text = sum_text + " " + chunk.to_text()

            if len(sum_text) >= 2000:
                link, metadatas = tika_processor.create_metadata(csv_data, file_name, chunk, link_data)
                self.update_metadata(sum_text, link, metadatas, file_name)
                db = ElasticsearchStore.from_texts(
                    es_connection=es_client,
                    # es_url=es_url,
                    index_name=index_name,
                    embedding=embeddings.get_embeddings(),
                    texts=[sum_text],
                    metadatas=[{'file_name': file_name, 'page_num': chunk.page_idx, 'bbox': chunk.bbox, 'link': link}],
                    # es_user=es_user,
                    # es_password=es_password,
                    es_api_key=os.getenv("ES_API_KEY"),
                    # query_field='content',
                    # distance_strategy="COSINE",
                    strategy=ElasticsearchStore.ApproxRetrievalStrategy(hybrid=True),
                    # vector_query_field='vector'
                    )
                sum_text = ''

            # generate captions and insert chunks
            if self.use_vlm:
                for image_file in image_list:
                    if int(image_file[0].split('_', 1)[0]) == int(chunk.page_idx):
                        caption = generate_response_qwen(image_file[1]) # try adding heading hierarchy info
                        print(caption, image_file[0])
                        self.update_metadata(caption)
                        image_list.remove(image_file)
                        db = ElasticsearchStore.from_texts(
                            es_connection=es_client,
                            # es_url=es_url,
                            index_name=index_name,
                            embedding=embeddings.get_embeddings(),
                            texts=[caption],
                            metadatas=[{'file_name': file_name, 'page_num': chunk.page_idx, 'bbox': chunk.bbox, 'link': link}],
                            # es_user=es_user,
                            # es_password=es_password,
                            es_api_key=os.getenv("ES_API_KEY"),
                            # query_field='content',
                            # distance_strategy="COSINE",
                            strategy=ElasticsearchStore.ApproxRetrievalStrategy(hybrid=True),
                            # vector_query_field='vector'
                        )




if __name__ == "__main__":
    converter = FileProcessor("./data/sample", "embedding", "sample_index", use_vlm=True)
    converter.process_file()

