import json
import logging
import os
import re
import time
from collections import defaultdict
from typing import Dict, Tuple, List
import json

import cv2
import fitz
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


class RevenueExtractorHelper:
    def __init__(self):
        self._table_model = self._load_table_model()

    def _load_table_model(self) -> YOLO:
        # load model
        model = YOLO("keremberke/yolov8n-table-extraction")

        # set model parameters
        model.overrides["conf"] = 0.25  # NMS confidence threshold
        model.overrides["iou"] = 0.45  # NMS IoU threshold
        model.overrides["agnostic_nms"] = False  # NMS class-agnostic
        model.overrides["max_det"] = 10  # maximum number of detections per image

        return model

    def pdf_page_to_image(self, pdf_path: str, page_num: int) -> Image.Image:
        images = convert_from_path(
            pdf_path, dpi=300, first_page=page_num, last_page=page_num
        )
        return images[0]

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
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

    def filter_pages(self, pdf_path: str) -> Dict[int, Boxes]:
        def check_line(text: str) -> bool:
            return (
                "Key Components of Results of Operations" in text
                or "Revenue recognition" in text
                or (
                    re.search(
                        r"(revenue|net revenue|total revenue)", text, re.IGNORECASE
                    )
                    # and "the following table" in text
                    and re.search(
                        r"(for the years|for the periods|years? ended december 31)",
                        text,
                        re.IGNORECASE,
                    )
                    # and re.search(r"(as a percentage|in absolute amounts)", text)
                )
            )

        self.document = fitz.open(pdf_path)
        filtered_pages = {}

        # get all the filtered pages
        for page_num, page in enumerate(self.document):
            page_num += 1
            text = page.get_textpage().extractText()

            if "revenue" not in text.lower():
                continue

            if check_line(text):
                tables = self._table_model(
                    self.pdf_page_to_image(pdf_path, page_num), verbose=False
                )
                if len(tables[0]) == 0:
                    continue
                if "Key Components of Results of Operations" in text:
                    filtered_pages.clear()
                filtered_pages[page_num] = tables

        return filtered_pages


class RevenueExtractor:
    def __init__(self, llm_api_key: str):
        self._helper = RevenueExtractorHelper()
        self._llm = OpenAI(api_key=llm_api_key, base_url="https://api.deepseek.com")

        # for revenue table extraction
        self.crop_offset = 30
        self.tesseract_config = r"--oem 3 --psm 6"
        self.confidence_pattern = re.compile(r"confidence level: (\d+(\.\d+)?)")
        self.table_prompt = "You are a finance expert. You are given a table in a text format extracted from an annual report. Your task is to determine if the table is a revenue table or not. If the table contains other information such as expenses and dividend, it is not considered as a revenue table. The table should include the total value of the most recent year. You just need to output the confidence level of the table being a revenue table in the format of 'confidence level: {your output}'"

        # for total revenue extraction
        self.total_revenue_pattern = re.compile(r"total revenue:\s(.+)")
        self.total_revenue_prompt = "You are a finance expert. You are given a revenue table in a text format extracted from an annual report. Your task is to return the total revenue for the most recent year, with the corresponding unit, such as US or RMB, and exact value. You also need to output the corresponding scale, such as thousands or million. You just need to output the total revenue in the format of 'total revenue: {your output}'"

        # for segments extraction
        # self.segment_pattern = re.compile()
        self.segment_prompt = "You are a finance expert. You are given a revenue table in a text format extracted from an annual report. Your task is to return the segments that form the revenue. You must output all the segments in point form. No need to output other information."

    def extract_revenue_table(self, pdf_path: str) -> Tuple[Dict[int, str], float]:
        self.pdf_path = pdf_path
        filtered_pages = self._helper.filter_pages(pdf_path)

        confidences_map = defaultdict(lambda: -1)
        table_text_map = {}

        for page_num, tables in filtered_pages.items():
            image = self._helper.pdf_page_to_image(pdf_path, page_num)
            # print(image, page_num)

            image_array = np.array(image)

            for box in tables[0].boxes.data.numpy().astype(int):
                x1, y1, x2, y2, _, _ = box

                cropped_image = image_array[
                    y1 - self.crop_offset : y2 + self.crop_offset,
                    x1 - self.crop_offset : x2 + self.crop_offset,
                ]
                if cropped_image.size == 0:
                    continue
                preprocessed_image = self._helper.preprocess_image(cropped_image)

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

                match = self.confidence_pattern.search(output)

                confidence_level = float(match.group(1)) if match else 0

                if confidence_level > confidences_map[page_num]:
                    confidences_map[page_num] = confidence_level
                    table_text_map[page_num] = table_text

                # print(table_text)
                # print("Confidence level: ", confidence_level)
                # print("=" * 50)

        max_confidence = max(confidences_map.values())
        selected_pages = [
            page_num
            for page_num, confidence in confidences_map.items()
            if confidence == max_confidence
        ]

        table_results = {}
        self.all_table_text = ""
        for selected_page_num in selected_pages:
            table_results[selected_page_num] = table_text_map[selected_page_num]
            self.all_table_text += table_text_map[selected_page_num]

        return table_results, max_confidence

    def extract_total_revenue(self, text: str) -> str:
        response = self._llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": self.total_revenue_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
            stream=False,
        )

        output = response.choices[0].message.content

        match = self.total_revenue_pattern.search(output)
        total_revenue = match.group(1) if match else "Not found"

        return total_revenue

    def extract_segments(self, text) -> List[str]:
        response = self._llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": self.segment_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
            stream=False,
        )
        
        output = response.choices[0].message.content
        print(output)
        
        segments = list(map(lambda x: x[2:], output.split("\n")))
        
        return segments


if __name__ == "__main__":
    pdf_path = "pdf_sample/3e7da988-b969-3703-bf91-596d105bffb7.pdf"
    api_key = "sk-436808c023b34f4185c96d8d438aa4a3"
    extractor = RevenueExtractor(api_key)

    start = time.time()

    results, confidence = extractor.extract_revenue_table(pdf_path)

    print(pdf_path)
    print(results)
    for page_num, text in results.items():
        print("-" * 50)
        print(f"Page {page_num}:")
        print(text)
        print("-" * 50)

    # total_revenue = extractor.extract_total_revenue(extractor.all_table_text)
    # print(f"Total revenue : {total_revenue}")

    segments = extractor.extract_segments(extractor.all_table_text)
    print(f"Segments : {segments}")

    print(f"Final confidence : {confidence}")

    # final_results = {}
    # for pdf in natsorted(os.listdir("pdf_sample")):
    #     pdf_path = f"pdf_sample/{pdf}"
    #     results, confidence = extractor.extract_revenue_table(pdf_path)
    #     total_revenue = extractor.extract_total_revenue(extractor.all_table_text)
    #     final_results[pdf.split(".")[0]] = {"tables": results, "total_revenue": total_revenue, "confidence": confidence}
    #     print("Finished processing: ", pdf)

    # with open("final_results.json", "w") as f:
    #     json.dump(final_results, f)

    print("Time taken: ", time.time() - start)
