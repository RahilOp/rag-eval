import os
import re
from json import load
from llmsherpa.readers import LayoutPDFReader

class TikaProcessor:
    def __init__(self, apply_ocr=False):
        self.apply_ocr = apply_ocr
        if apply_ocr:
            self.llmsherpa_api_url = "http://localhost:5009/api/parseDocument?renderFormat=all&useNewIndentParser=yes&applyOcr=yes"
        else:
            self.llmsherpa_api_url = "http://localhost:5009/api/parseDocument?renderFormat=all&useNewIndentParser=yes"

    def read_pdf_with_llmsherpa(self, file_path):
        pdf_reader = LayoutPDFReader(self.llmsherpa_api_url)
        return pdf_reader.read_pdf(file_path)

    def generate_metadata_csv(self, file_path):
        csv_file = re.sub('pdf$', 'csv', file_path)
        command = f"exiftool -csv '{file_path}' > '{csv_file}'"
        os.system(command)
        return csv_file

    def get_page_length(self, json_out, page_idx):
        page_len = 0
        for sentence in json_out:
            if sentence['page_idx'] == page_idx and 'sentences' in sentence.keys():
                for j in sentence['sentences']:
                    page_len += len(j)
        return page_len

    def create_metadata(self, csv_data, file_name, chunk, link_data):
        page_num = chunk.page_idx + 1
        link = link_data[file_name] + '#page=' + str(page_num)
        bbox = chunk.bbox
        metadatas = {'title': file_name, 'page_num': page_num, 'bbox': bbox}
        for i in range(len(csv_data.iloc[0])):
            metadatas.update({csv_data.keys()[i]: csv_data.iloc[0][i]})
        return link, metadatas

    def length_after_first_newline(self, text):
        first_newline_index = text.find('\n')
        return len(text[first_newline_index + 1:]) if first_newline_index != -1 else 0

    def load_json(self, json_path):
        with open(json_path, 'r') as f:
            return load(f)

    def get_files(self, folder_path):
        return [f for f in os.listdir(folder_path) if f.endswith('.pdf')]