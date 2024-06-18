import io
import json
import os
import re
import shutil
import time
from collections import defaultdict
from typing import Dict

import fitz
import numpy as np
import pytesseract
import ultralytics
from natsort import natsorted
from PIL import Image
from ultralyticsplus import YOLO


def check_line(line):
    return (
        re.search(r"(revenue|net revenue|total revenue)", line)
        # and "the following table" in line
        and re.search(r"(for the years|for the periods|years? ended december 31)", line)
        # and re.search(r"(as a percentage|in absolute amounts)", line)
    )


data = {
    "0bce53d3-5a06-32f7-8154-98dc65ebab39": 160,
    "000d93bd-8f91-30cf-97ab-e612ba04b3f5": 219,
    "2e69d284-6dac-36ef-87b5-5e9d8a8973e5": 113,
    "3b08464e-9978-3c38-8c60-7cd239050855": 142,
    "3d059130-fc20-3f1d-95c9-5778a5a58b21": 145,
    "3e7da988-b969-3703-bf91-596d105bffb7": 154,
    "4a1641f0-241a-33fe-9f1c-ec0687ad4d29": 136,
    "4e9de22e-57e5-3701-845c-e9b08256bbd1": 85,
    "4fa37fab-5790-3d21-8718-407333f287b3": 131,
    "5d42fc53-e7f0-36ca-b446-a6bcc16c8eac": 136,
    "6f02f942-3e1d-3053-b12b-e1082a9356db": 81,
    "7d280936-6d08-35ae-bdf9-fa3fb2791828": 186,
    "08de6fb2-42a5-3c04-92cc-41df09db462a": 91,
    "9a3cb882-bf64-3506-a4f5-01e244e07bbe": 114,
    "9c21c57a-35c7-3cb4-82ae-126bc7ecf641": 141,
    "9d4d8e53-3874-3cb4-b1df-18d9218c7bb2": 114,
    "24ddebec-566d-341c-8920-d7a5847499d6": 174,
    "27ce0eda-2547-34ee-9806-5d8be21b5c9b": 221,
    "28a5cc35-84f1-3271-8be2-ccd208e3082a": 158,
    "43fa50ca-6dd7-3133-8752-9bde56f72770": 139,
    "44f83fbe-7dd9-3ed6-88ca-990a0349db70": 114,
    "78cf4857-3fcd-36f5-b9da-adac8c338194": 132,
    "81cafc38-7920-36f3-bdd7-1007ee26be52": 167,
    "90d130a1-2225-3e07-b73d-7794d066edaa": 85,
    "0129c5e0-dacb-32d4-92ca-fc5f7d5c8a6b": 163,
    "2177e713-b2bd-3783-b782-b31db7d5695a": 157,
    "4618f882-ab07-38cf-ad70-6479d8c7668a": 156,
    "4843ab8c-fb7b-3870-9137-1b4da5b0a182": 175,
    "6194e704-82cf-338b-a9a2-36093fc50938": 150,
    "7630c2fc-d6e8-3122-b563-9e721c34d800": 237,
    "35704fd1-007d-368c-8d62-ce344ce833fe": 108,
    "43532eaf-f163-3233-b514-8774f0c61751": 77,
    "755349e7-c35f-3031-9857-b2cfd70bf6b7": 248,
    "5594850a-36e2-3be2-9970-c8a51744eb19": 141,
    "26274672-34cc-3dbd-a8b4-48933f1b189d": 116,
    "a9ce0cd7-2166-3317-af23-5dd3c11d3e53": 130,
    "a377ab26-0a90-389e-aa71-175bb68c0984": 115,
    "a325100f-0011-3869-b40e-bd66891c9339": 161,
    "ab418d5e-c8f0-30b8-bbc2-e948702b71ac": 141,
    "c0e7960f-57eb-3e38-820b-316c530dfccd": 194,
    "c4c0e881-7735-3005-b23b-3a90aa39294d": 77,
    "c22d1270-ca35-3430-9b02-b67fc98ce5bc": 219,
    "cb23fa37-4b14-386b-ba0d-6b154174ff5b": 126,
    "cc0a0543-a4fd-3821-a047-39e6a023be8f": 278,
    "e73300b6-d21f-3cdb-88e8-cd2671ade899": 166,
    "edb4cfee-ad54-33ee-b766-a991aeec7fe9": None,
    "ef440f04-a4e7-3462-a9f0-1394d8dfebc9": 159,
    "f29b8d26-64e9-38a3-ada1-bc457361b355": None,
    "f90fff09-95a4-3a6d-b7e9-7e5f5e493ec7": None,
}

new_data = {
    "0bce53d3-5a06-32f7-8154-98dc65ebab39": 160,
    "000d93bd-8f91-30cf-97ab-e612ba04b3f5": 219,
    "2e69d284-6dac-36ef-87b5-5e9d8a8973e5": 113,
    "3b08464e-9978-3c38-8c60-7cd239050855": 142,
    "3d059130-fc20-3f1d-95c9-5778a5a58b21": 145,
    "3e7da988-b969-3703-bf91-596d105bffb7": 154,
    "4a1641f0-241a-33fe-9f1c-ec0687ad4d29": 136,
    "4e9de22e-57e5-3701-845c-e9b08256bbd1": 136,
    "4fa37fab-5790-3d21-8718-407333f287b3": 230,
    "5d42fc53-e7f0-36ca-b446-a6bcc16c8eac": 263,
    "6f02f942-3e1d-3053-b12b-e1082a9356db": 183,
    "7d280936-6d08-35ae-bdf9-fa3fb2791828": 314,
    "08de6fb2-42a5-3c04-92cc-41df09db462a": 183,
    "9a3cb882-bf64-3506-a4f5-01e244e07bbe": 114,
    "9c21c57a-35c7-3cb4-82ae-126bc7ecf641": 272,
    "9d4d8e53-3874-3cb4-b1df-18d9218c7bb2": 114,
    "24ddebec-566d-341c-8920-d7a5847499d6": 294,
    "27ce0eda-2547-34ee-9806-5d8be21b5c9b": 378,
    "28a5cc35-84f1-3271-8be2-ccd208e3082a": 158,
    "43fa50ca-6dd7-3133-8752-9bde56f72770": 266,
    "44f83fbe-7dd9-3ed6-88ca-990a0349db70": 114,
    "78cf4857-3fcd-36f5-b9da-adac8c338194": 281,
    "81cafc38-7920-36f3-bdd7-1007ee26be52": 167,
    "90d130a1-2225-3e07-b73d-7794d066edaa": 85,
    "0129c5e0-dacb-32d4-92ca-fc5f7d5c8a6b": 336,
    "2177e713-b2bd-3783-b782-b31db7d5695a": 157,
    "4618f882-ab07-38cf-ad70-6479d8c7668a": 156,
    "4843ab8c-fb7b-3870-9137-1b4da5b0a182": 73,
    "6194e704-82cf-338b-a9a2-36093fc50938": 249,
    "7630c2fc-d6e8-3122-b563-9e721c34d800": 237,
    "35704fd1-007d-368c-8d62-ce344ce833fe": 202,
    "43532eaf-f163-3233-b514-8774f0c61751": 77,
    "755349e7-c35f-3031-9857-b2cfd70bf6b7": 248,
    "5594850a-36e2-3be2-9970-c8a51744eb19": 248,
    "26274672-34cc-3dbd-a8b4-48933f1b189d": 116,
    "a9ce0cd7-2166-3317-af23-5dd3c11d3e53": 244,
    "a377ab26-0a90-389e-aa71-175bb68c0984": 115,
    "a325100f-0011-3869-b40e-bd66891c9339": 161,
    "ab418d5e-c8f0-30b8-bbc2-e948702b71ac": 250,
    "c0e7960f-57eb-3e38-820b-316c530dfccd": 194,
    "c4c0e881-7735-3005-b23b-3a90aa39294d": 77,
    "c22d1270-ca35-3430-9b02-b67fc98ce5bc": 219,
    "cb23fa37-4b14-386b-ba0d-6b154174ff5b": 126,
    "cc0a0543-a4fd-3821-a047-39e6a023be8f": 278,
    "e73300b6-d21f-3cdb-88e8-cd2671ade899": 283,
    "edb4cfee-ad54-33ee-b766-a991aeec7fe9": None,
    "ef440f04-a4e7-3462-a9f0-1394d8dfebc9": 159,
    "f29b8d26-64e9-38a3-ada1-bc457361b355": None,
    "f90fff09-95a4-3a6d-b7e9-7e5f5e493ec7": None,
}

details = defaultdict(list)
model = YOLO("keremberke/yolov8n-table-extraction")

# set model parameters
model.overrides["conf"] = 0.25  # NMS confidence threshold
model.overrides["iou"] = 0.45  # NMS IoU threshold
model.overrides["agnostic_nms"] = False  # NMS class-agnostic
model.overrides["max_det"] = 10  # maximum number of detections per image


def get_tables(page: fitz.Page) -> ultralytics.yolo.engine.results.Boxes:
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples_mv)
    tables = model(img, verbose=False)

    return tables


def filter_pages(pdf_path: str) -> Dict[int, ultralytics.yolo.engine.results.Boxes]:
    file = fitz.open(pdf_path)
    filtered_pages = {}

    # get all the filtered pages
    for page_num, page in enumerate(file):
        page_num += 1
        text = page.get_textpage().extractText()

        if "revenue" not in text.lower():
            continue

        if (
            "Key Components of Results of Operations" in text
            or "Revenue recognition" in text
            or check_line(text.lower())
        ):
            tables = get_tables(page)
            if len(tables[0]) == 0:
                continue
            if (
                "Key Components of Results of Operations" in text
                or "Revenue recognition" in text
            ):
                filtered_pages.clear()
            filtered_pages[page_num] = tables

    return filtered_pages


count = set()
start = time.time()

for pdf in natsorted(os.listdir("pdf_sample")):
    filtered_pages = filter_pages(f"pdf_sample/{pdf}")
    details[pdf.split(".")[0]] = list(filtered_pages.keys())
    # file = fitz.open(f"pdf_sample/{pdf}")
    # pdf_name = pdf.split(".")[0]

    # # get all the filtered pages
    # for page_num, page in enumerate(file):
    #     page_num += 1
    #     text = page.get_textpage().extractText()
    #     if "revenue" not in text.lower():
    #         continue

    #     if "Key Components of Results of Operations" in text:
    #         # the revenue table must be located after or at this page
    #         details[pdf_name] = [page_num]
    #         count.add(pdf_name)
    #         continue

    #     if "Revenue recognition" in text or check_line(text.lower()):
    #         tables = check_tables(page)
    #         if len(tables[0]) == 0:
    #             continue
    #         if "Revenue recognition" in text:
    #             details[pdf_name].clear()
    #         details[pdf_name].append(page_num)
    #     # print(page.get_textpage().extractText())

# save details
with open("details.json", "w") as f:
    json.dump(details, f)


correct = {"True": 0, "False": 0}
for pdf_name, filtered_pages in details.items():
    label = new_data[pdf_name]
    if label is None:
        print(f"{pdf_name} : None")
        continue
    if label in filtered_pages:
        print(f"{pdf_name} : True - {filtered_pages}")
        correct["True"] += 1
    else:
        print(f"{pdf_name} : False - {filtered_pages}")
        correct["False"] += 1

print("=" * 50)
print(correct)
print("Total #reports with key components: ", len(count))
print("Accuracy: ", correct["True"] / (correct["True"] + correct["False"]))
print("Time taken: ", time.time() - start)


# count = 0
# for page_num, page in enumerate(file):
#     text = page.get_textpage().extractText()
#     if "revenue" not in text.lower():
#         continue
#     if check_line(text.lower()):
#         print(page_num)
#     count += 1
#     # print(page.get_textpage().extractText())

# print("Total pages with revenue: ", count)
# print("Total pages: ", len(file))
