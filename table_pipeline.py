import json
import logging
import os
import re
import time
from collections import defaultdict
from typing import Dict, Tuple
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


def load_table_model() -> YOLO:
    # load model
    model = YOLO("keremberke/yolov8n-table-extraction")

    # set model parameters
    model.overrides["conf"] = 0.25  # NMS confidence threshold
    model.overrides["iou"] = 0.45  # NMS IoU threshold
    model.overrides["agnostic_nms"] = False  # NMS class-agnostic
    model.overrides["max_det"] = 10  # maximum number of detections per image

    return model


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
    )


def filter_pages(pdf_path: str) -> Dict[int, Boxes]:
    file = fitz.open(pdf_path)
    filtered_pages = {}
    table_model = load_table_model()

    # get all the filtered pages
    for page_num, page in enumerate(file):
        page_num += 1
        text = page.get_textpage().extractText()

        if "revenue" not in text.lower():
            continue

        if check_line(text):
            tables = table_model(pdf_page_to_image(pdf_path, page_num), verbose=False)
            if len(tables[0]) == 0:
                continue
            if (
                "Key Components of Results of Operations" in text
            ):
                filtered_pages.clear()
            filtered_pages[page_num] = tables

    return filtered_pages


def extract_revenue_table(pdf_path: str, api_key: str) -> Tuple[Dict[int, str], float]:
    filtered_pages = filter_pages(pdf_path)

    offset = 30
    custom_config = r"--oem 3 --psm 6"
    confidence_pattern = re.compile(r"confidence level: (\d+(\.\d+)?)")

    llm = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt = "You are a finance expert. You are given a table in a text format extracted from an annual report. Your task is to determine if the table is a revenue table or not. If the table contains other information such as expenses and dividend, it is not considered as a revenue table. The table should include the total value of the most recent year. You just need to output the confidence level of the table being a revenue table in the format of 'confidence level: {your output}'"

    confidences_map = defaultdict(lambda: -1)
    table_text_map = {}

    for page_num, tables in filtered_pages.items():
        image = pdf_page_to_image(pdf_path, page_num)
        # print(image, page_num)

        image_array = np.array(image)

        for box in tables[0].boxes.data.numpy().astype(int):
            x1, y1, x2, y2, _, _ = box

            cropped_image = image_array[
                y1 - offset : y2 + offset, x1 - offset : x2 + offset
            ]
            if cropped_image.size == 0:
                continue
            preprocessed_image = preprocess_image(cropped_image)

            table_text = pytesseract.image_to_string(
                preprocessed_image, config=custom_config
            )

            response = llm.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": table_text},
                ],
                temperature=0,
                stream=False,
            )

            output = response.choices[0].message.content

            match = confidence_pattern.search(output)

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

    return {
        selected_page_num: table_text_map[selected_page_num] for selected_page_num in selected_pages
    }, max_confidence


if __name__ == "__main__":
    pdf_path = "pdf_sample/6194e704-82cf-338b-a9a2-36093fc50938.pdf"
    api_key = "sk-436808c023b34f4185c96d8d438aa4a3"

    start = time.time()

    # results, confidence = extract_revenue_table(pdf_path, api_key)

    # print(pdf_path)
    # print(results)
    # for page_num, text in results.items():
    #     print("-" * 50)
    #     print(f"Page {page_num}:")
    #     print(text)
    #     print("-" * 50)
        
    # print(f"Final confidence : {confidence}")
    # print("Time taken: ", time.time() - start)

    final_results = {}
    for pdf in natsorted(os.listdir("pdf_sample")):
        pdf_path = f"pdf_sample/{pdf}"
        results, confidence = extract_revenue_table(pdf_path, api_key)
        final_results[pdf.split(".")[0]] = results
        print("Finished processing: ", pdf)

    with open("final_results.json", "w") as f:
        json.dump(final_results, f)

    print("Time taken: ", time.time() - start)
    
    
    



"""
You are a finance expert. You are given a revenue table in a text format extracted from an annual report. Your task is to return the total revenue for the most recent year, with the corresponding unit and exact value. You just need to output the total revenue in the format of 'total revenue: {your output}'
"""
