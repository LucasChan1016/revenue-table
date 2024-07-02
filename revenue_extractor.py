import json
import logging
import os
import re
import time
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
import json

import cv2
import fitz
import nltk
import numpy as np
import pytesseract
from natsort import natsorted
from openai import OpenAI
from pdf2image import convert_from_path
from PIL import Image
from ultralytics.yolo.engine.results import Boxes
from ultralyticsplus import YOLO

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

try:
    # Check if 'punkt' tokenizer is available
    nltk.data.find("tokenizers/punkt")
    # print("Punkt tokenizer is already downloaded.")
except LookupError:
    # print("Punkt tokenizer not found. Downloading now...")
    nltk.download("punkt")


def load_table_model() -> YOLO:
    # load model
    model = YOLO("keremberke/yolov8n-table-extraction")

    # set model parameters
    model.overrides["conf"] = 0.25  # NMS confidence threshold
    model.overrides["iou"] = 0.45  # NMS IoU threshold
    model.overrides["agnostic_nms"] = False  # NMS class-agnostic
    model.overrides["max_det"] = 10  # maximum number of detections per image

    return model


table_model = load_table_model()


def convert_json(text: str) -> dict:
    # Use regular expression to extract the JSON string inside the Markdown code block
    json_match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if not json_match:
        return {"Not Matched": None}

    json_str = json_match.group(1)

    # Load the JSON string into a dictionary
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        # print(e)
        return {"Not Loaded": None}

    return data


def pdf_page_to_image(pdf_path: str, page_num: int) -> Image.Image:
    images = convert_from_path(
        pdf_path, dpi=300, first_page=page_num, last_page=page_num
    )
    return images[0]


def preprocess_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Repair horizontal table lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 2))
    detect_horizontal = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )
    cnts = cv2.findContours(
        detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 4)

    # denoise the image
    image = cv2.fastNlMeansDenoisingColored(image, None, 30, 30, 7, 21)

    return image


def filter_pages(pdf_path: str) -> Dict[int, Boxes]:
    def check_line(text: str) -> bool:
        return (
            "Key Components of Results of Operations" in text
            or "Revenue recognition" in text
            or (
                re.search(r"(revenue|net revenue|total revenue)", text, re.IGNORECASE)
                # and "the following table" in text
                and re.search(
                    r"(for the years|for the periods|years? ended december 31)",
                    text,
                    re.IGNORECASE,
                )
                # and re.search(r"(as a percentage|in absolute amounts)", text)
            )
        ) and "deferred tax" not in text.lower() and "within the scope of ASC 606" not in text

    document = fitz.open(pdf_path)
    filtered_pages = {}

    # get all the filtered pages
    for page_num, page in enumerate(document):
        page_num += 1
        text = page.get_textpage().extractText()

        # if "revenue" not in text.lower():
        #     continue

        if check_line(text):
            tables = table_model(pdf_page_to_image(pdf_path, page_num), verbose=False)
            if len(tables[0]) == 0:
                continue
            if "Key Components of Results of Operations" in text:
                filtered_pages.clear()
            filtered_pages[page_num] = tables

    return filtered_pages


def get_segment_info_pages(document) -> list:
    segment_infos_pages = []

    for page_num, page in enumerate(document):
        page_num += 1
        text = page.get_textpage().extractText()

        if "segment information" in text.lower():
            segment_infos_pages.append(page_num)

    return segment_infos_pages


def merge_dicts(data: Dict[str, List[Dict[str, Optional[list]]]]) -> Dict[str, Dict[str, list]]:
    '''
    data: dictionary with a list of dictionaries
    
    Remove the duplicates and merge the dictionaries in the list
    '''
    def is_valid(value):
        return value is not None and isinstance(value, list) and len(value) > 0
    
    def can_convert_float(x):
        try:
            float(x.replace(",", ""))
            return True
        except ValueError:
            return False
    
    merged = {}
    for key, list_of_dicts in data.items():
        merged[key] = {}
        for d in list_of_dicts:
            for k, v in d.items():
                if not is_valid(v):
                    continue
                if k in merged[key]:
                    merged[key][k] += v
                else:
                    merged[key][k] = v
        
        # remove duplicates
        for k, v in merged[key].items():
            v = list(filter(lambda x: not can_convert_float(x), set(v)))
            merged[key][k] = v
    return merged


class RevenueExtractor:
    def __init__(self, pdf_path: str, llm_api_key: str):
        self.pdf_path = pdf_path
        self._llm = OpenAI(api_key=llm_api_key, base_url="https://api.deepseek.com")
        self.document = fitz.open(pdf_path)

        # for revenue table extraction
        self.crop_offset = 30
        self.tesseract_config = r"--oem 3 --psm 6"
        self.table_prompt = "You are a finance expert. You are given a table in a text format extracted from an annual report. Your task is to determine if the table is a revenue table or not. If the table contains other information such as expenses and dividend, it is not considered as a revenue table. The table should include the total value of the most recent year. You MUST output the confidence level of the table being a revenue table in a json format: {'confidence': value in float}. You will be penalized for wrong result. I’m going to tip $10000 for a better solution!"

        self.max_table_confidence = 0
        self.table_threshold = 0.5
        self.table_results = {}
        self.all_tables_text = None

        # for total revenue extraction
        self.total_revenue_prompt = "You are a finance expert. You are given a revenue table in a text format extracted from an annual report. Your task is to return the total revenue for the most recent year, with the corresponding unit, such as US or RMB, and exact value. You also need to output the corresponding scale, such as thousands or millions. You MUST output the total revenue in a json format: {'total_revenue': value in string}. You will be penalized for wrong result. I’m going to tip $10000 for a better solution!"

        self.total_revenue = ""

        # for segments extraction
        self.segment_prompt = "You are a finance expert. You are given a revenue table in a text format extracted from an annual report. Your task is to return the segments that form the revenue and the corresponding values. Please extract the information from only one table, without combining the results of all the tables. You MUST output all the segments in a valid json format: {'segment_name': {'year 1': 'value in string', 'year 2': 'value in string'}}. No need to output other information. You will be penalized for wrong result. I’m going to tip $10000 for a better solution!"

        self.segments = None

        # for segment sentences extraction
        self.segment_sentence_prompt = (
            lambda text: "You are a finance expert. You are given a page of segment information extracted from an annual report, and a particular segment name. Your task is to return the sentences that describe the particular segments. You MUST output the sentences without the values of that particular segments. You MUST extract the information from only one page. You MUST output all the sentences in a json format: {'segment_name': sentences in list format}. No need to output other information. You will be penalized for wrong result. I’m going to tip $10000 for a better solution!"
            + f"\n\n the segment information page: \n{text}"
        )

        self.segment_sentences = None

    
    def revenue_table_backup(self):
        self.table_results = {}
        self.max_table_confidence = 0
    

    def extract_revenue_table(self) -> Tuple[Dict[int, str], float]:
        filtered_pages = filter_pages(self.pdf_path)

        confidences_map = defaultdict(lambda: -1)
        table_text_map = {}

        for page_num, tables in filtered_pages.items():
            print(f"Page {page_num}:")
            doc_text = self.document[page_num - 1].get_textpage().extractText()
            doc_sentences = nltk.tokenize.sent_tokenize(doc_text)
            doc_sentences = list(map(lambda x: x.replace("\n", " "), doc_sentences))
            doc_sentences = list(filter(lambda x: "revenue" in x.lower(), doc_sentences))
            
            image = pdf_page_to_image(pdf_path, page_num)

            image_array = np.array(image)

            for box in tables[0].boxes.data.numpy().astype(int):
                x1, y1, x2, y2, _, _ = box

                cropped_image = image_array[
                    y1 - self.crop_offset : y2 + self.crop_offset,
                    x1 - self.crop_offset : x2 + self.crop_offset,
                ]
                if cropped_image.size == 0:
                    continue
                preprocessed_image = preprocess_image(cropped_image)

                table_text = pytesseract.image_to_string(
                    preprocessed_image, config=self.tesseract_config
                )

                response = self._llm.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": self.table_prompt},
                        {"role": "user", "content": table_text},
                    ],
                    temperature=0,
                    stream=False,
                )

                output = response.choices[0].message.content

                output_json = convert_json(output)

                confidence_level = float(output_json.get("confidence", 0))
                
                # discard the table if the confidence level is below the threshold
                if confidence_level < self.table_threshold:
                    continue

                if confidence_level > confidences_map[page_num]:
                    confidences_map[page_num] = confidence_level
                    table_text_map[page_num] = table_text

                print(table_text)
                print("Confidence level: ", confidence_level)
                print("=" * 50)

        confidences_values = list(confidences_map.values())
        self.max_table_confidence = max(confidences_values) if len(confidences_values) > 0 else 0
        
        if self.max_table_confidence < self.table_threshold:
            self.revenue_table_backup()
            return self.table_results, self.max_table_confidence

        selected_pages = [
            page_num
            for page_num, confidence in confidences_map.items()
            if confidence == self.max_table_confidence
        ]

        self.table_results = {}
        self.all_tables_text = ""
        for selected_page_num in selected_pages:
            self.table_results[selected_page_num] = table_text_map[selected_page_num]
            self.all_tables_text += table_text_map[selected_page_num] + f"\n{'-'*80}\n"

        return self.table_results, self.max_table_confidence


    def extract_total_revenue(self) -> str:
        if not self.all_tables_text:
            self.extract_revenue_table()

        response = self._llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": self.total_revenue_prompt},
                {"role": "user", "content": self.all_tables_text},
            ],
            temperature=0,
            stream=False,
        )

        output = response.choices[0].message.content

        output_json = convert_json(output)

        self.total_revenue = output_json.get("total_revenue", "Not found")

        return self.total_revenue


    def extract_segments(self) -> Dict[str, Dict[str, str]]:
        if not self.all_tables_text:
            self.extract_revenue_table()

        response = self._llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": self.segment_prompt},
                {"role": "user", "content": self.all_tables_text},
            ],
            temperature=0,
            stream=False,
        )

        output = response.choices[0].message.content

        print(output)

        self.segments = convert_json(output)

        return self.segments


    def extract_sentences(self) -> dict:
        if not self.segments:
            self.extract_segments()
        segment_names = list(self.segments.keys())

        segment_infos_pages = get_segment_info_pages(self.document)

        self.segment_sentences = defaultdict(list)

        segment_details_pages = segment_infos_pages + list(self.table_results.keys())

        for page_num in segment_details_pages:
            page = self.document[page_num - 1].get_textpage().extractText()
            idx = page.lower().find("segment information")
            if idx != -1:
                page = page[idx:]

            for name in segment_names:
                # print(name)
                response = self._llm.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {
                            "role": "system",
                            "content": self.segment_sentence_prompt(page),
                        },
                        {"role": "user", "content": name},
                    ],
                    temperature=0,
                    stream=False,
                )

                output = response.choices[0].message.content
                output_json = convert_json(output)

                self.segment_sentences[name].append(output_json)

        self.segment_sentences = merge_dicts(self.segment_sentences)

        return self.segment_sentences


if __name__ == "__main__":
    pdf_path = "pdf_sample/4a1641f0-241a-33fe-9f1c-ec0687ad4d29.pdf"
    api_key = "sk-436808c023b34f4185c96d8d438aa4a3"
    extractor = RevenueExtractor(pdf_path, api_key)

    one_result = {}

    start = time.time()

    table_results, confidence = extractor.extract_revenue_table()

    print(pdf_path)
    print(table_results)
    for page_num, text in table_results.items():
        print("-" * 50)
        print(f"Page {page_num}:")
        print(text)
        print("-" * 50)

    total_revenue = extractor.extract_total_revenue()
    print(f"Total revenue : {total_revenue}")

    segments = extractor.extract_segments()
    print(f"Segments : {segments}")

    segment_sentences = extractor.extract_sentences()
    print(f"Segment sentences : {segment_sentences}")

    print(f"Final confidence : {confidence}")

    time_taken = time.time() - start

    one_result["pdf_path"] = pdf_path
    one_result["tables"] = table_results
    one_result["total_revenue"] = total_revenue
    one_result["confidence"] = confidence
    one_result["segments"] = segments
    one_result["segment_sentences"] = segment_sentences
    one_result["time_taken"] = time_taken

    with open("one_result.json", "w") as f:
        json.dump(one_result, f)

    ######################################################################################

    # final_results = {}
    # for pdf in natsorted(os.listdir("pdf_sample")):
    #     pdf_path = f"pdf_sample/{pdf}"
    #     pdf_name = pdf.split(".")[0]

    #     extractor = RevenueExtractor(pdf_path, api_key)
    #     results, confidence = extractor.extract_revenue_table()
    #     total_revenue = extractor.extract_total_revenue()
    #     segments = extractor.extract_segments()

    #     final_results[pdf_name] = {"tables": results, "total_revenue": total_revenue, "confidence": confidence, "segments": segments}
    #     print("Finished processing: ", pdf)

    # time_taken = time.time() - start
    # final_results["time_taken"] = time_taken

    # print("Time taken: ", time_taken)

    # with open("final_results.json", "w") as f:
    #     json.dump(final_results, f)
