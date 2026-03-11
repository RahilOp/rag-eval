import os
import shutil
import glob
from spire.pdf.common import *
from spire.pdf import *
import subprocess


class OCRPipeline:
    def __init__(self, folder_path):
        self.root_folder = folder_path + "/temp-ocr"
        self.marker_command = "marker_single"
        self.batch_multiplier = 2
        self.langs = "English,Japanese"

    def perform_ocr_chunk(self, file_name, file_path, link_data, chunk):
        pdf_doc = PdfDocument()
        pdf_doc.LoadFromFile(file_path)
        
        new_doc = PdfDocument()
        new_doc.InsertPage(pdf_doc, chunk.page_idx)
        
        file_path_temp = self.root_folder + f"/{chunk.page_idx}.pdf"
        print(file_path_temp)
        new_doc.SaveToFile(file_path_temp)
        
        ocr_command = f"{self.marker_command} {file_path_temp} {self.root_folder} --batch_multiplier {self.batch_multiplier} --langs {self.langs}"
        os.system(ocr_command)

        markdown_content = self.read_markdown_file(self.root_folder)
        link = link_data[file_name] + '#page=' + str(chunk.page_idx + 1)
        metadatas = {'title': file_name, 'page_num': chunk.page_idx + 1, 'bbox': chunk.bbox, 'link': link}
        
        new_doc.Close()
        shutil.rmtree(self.root_folder)
        
        return markdown_content, link, metadatas
    
    def perform_ocr_json(self, file_name, file_path, link_data, json_out):
        pdf_doc = PdfDocument()
        pdf_doc.LoadFromFile(file_path)
        
        new_doc = PdfDocument()
        new_doc.InsertPage(pdf_doc, json_out['page_idx'])
        
        file_path_temp = self.root_folder + f"/{json_out['page_idx']}.pdf"
        print(file_path_temp)
        new_doc.SaveToFile(file_path_temp)
        
        ocr_command = f"{self.marker_command} {file_path_temp} {self.root_folder} --batch_multiplier {self.batch_multiplier} --langs {self.langs}"
        os.system(ocr_command)

        markdown_content = self.read_markdown_file(self.root_folder)
        link = link_data[file_name] + '#page=' + str(json_out['page_idx'] + 1)
        metadatas = {'title': file_name, 'page_num': json_out['page_idx'] + 1, 'bbox': json_out['bbox'], 'link': link}
        
        new_doc.Close()
        shutil.rmtree(self.root_folder)
        
        return markdown_content, link, metadatas

    def read_markdown_file(self, root_folder):
        markdown_files = []
        for subdir, _, _ in os.walk(root_folder):
            markdown_files.extend(glob.glob(os.path.join(subdir, '*.md')))
        
        # print(markdown_files)

        for fi in markdown_files:
            # Reading the Markdown file
            with open(fi, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
  
        return markdown_content