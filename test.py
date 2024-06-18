import os
import re
import shutil

import cv2
import fitz
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from ultralyticsplus import YOLO, render_result

filtered_pages_manual = [160, 162, 169, 190, 263, 266, 281, 283, 306, 309, 314, 345]
pdf_name = "pdf_sample/0bce53d3-5a06-32f7-8154-98dc65ebab39.pdf"

# load model
model = YOLO("keremberke/yolov8n-table-extraction")

# set model parameters
model.overrides["conf"] = 0.25  # NMS confidence threshold
model.overrides["iou"] = 0.45  # NMS IoU threshold
model.overrides["agnostic_nms"] = False  # NMS class-agnostic
model.overrides["max_det"] = 10  # maximum number of detections per image


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


def get_tables(img, test_idx):

    img.save(f"fitz_pix_images/{test_idx}.png")
    tables = model(img, verbose=False)
    render = render_result(model=model, image=img, result=tables[0])
    render.save(f"fitz_pix_images/{test_idx}_table.png")

    return tables


def filter_pages(pdf_path: str):
    file = fitz.open(pdf_path)
    filtered_pages = {}
    test_idx = 1

    # get all the filtered pages
    for page_num, page in enumerate(file):
        page_num += 1
        text = page.get_textpage().extractText()

        if "revenue" not in text.lower():
            continue

        if check_line(text):
            tables = get_tables(pdf_page_to_image(pdf_path, page_num), test_idx)
            test_idx += 1
            if len(tables[0]) == 0:
                continue
            if (
                "Key Components of Results of Operations" in text
                or "Revenue recognition" in text
            ):
                filtered_pages.clear()
            filtered_pages[page_num] = tables[0]

    return filtered_pages


offset = 30
custom_config = r"--oem 3 --psm 6"


def pdf_page_to_image(pdf_path: str, page_num: int):
    images = convert_from_path(
        pdf_path, dpi=300, first_page=page_num, last_page=page_num
    )
    return images[0]


def preprocess_image(image: np.ndarray):
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


if os.path.exists("filtered_tables"):
    shutil.rmtree("filtered_tables")

os.makedirs("filtered_tables")

filtered_pages = filter_pages(pdf_name)

for page_num, tables in filtered_pages.items():
    image = pdf_page_to_image(pdf_name, page_num)
    image_array = np.array(image)
    # tables = model(image, verbose=False)

    file_offset = 0

    for box in tables.boxes.data.numpy().astype(int):
        x1, y1, x2, y2, _, _ = box

        cropped_image = image_array[
            y1 - offset : y2 + offset, x1 - offset : x2 + offset
        ]
        preprocessed_image = preprocess_image(cropped_image)

        text = pytesseract.image_to_string(preprocessed_image, config=custom_config)

        cv2.imwrite(
            f"filtered_tables/{page_num}_table_cropped_{file_offset}.png",
            preprocessed_image,
        )
        with open(
            f"filtered_tables/{page_num}_table_text_{file_offset}.txt", "w"
        ) as file:
            file.write(text)

        file_offset += 1
