import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from ultralyticsplus import YOLO, render_result


def pdf_page_to_image(pdf_path: str, page_num: int, image_path: str):
    images = convert_from_path(
        pdf_path, dpi=300, first_page=page_num, last_page=page_num
    )
    images[0].save(image_path, image_path.split(".")[-1].upper())


# Path to the PDF file
pdf_path = "pdf_sample/ef440f04-a4e7-3462-a9f0-1394d8dfebc9.pdf"
page_number = 159

image_name = pdf_path.split("/")[1].split("-")[0]  # 9c21c57a
image_folder = "table_images"
image_path = f"{image_folder}/{image_name}_table.png"
pdf_page_to_image(pdf_path, page_number, image_path)

# load model
model = YOLO("keremberke/yolov8n-table-extraction")

# set model parameters
model.overrides["conf"] = 0.25  # NMS confidence threshold
model.overrides["iou"] = 0.45  # NMS IoU threshold
model.overrides["agnostic_nms"] = False  # NMS class-agnostic
model.overrides["max_det"] = 10  # maximum number of detections per image

# perform inference
results = model(image_path, verbose=False)

# exit if no table is detected
if len(results[0]) == 0:
    print("No tables detected")
    exit()

print("Length of results: ", len(results[0]))
print("=================")
print(results)
for result in results[0].boxes.data.numpy().astype(int):
    x1, y1, x2, y2, _, _ = result
    print(x1, y1, x2, y2)
print("=================")
# observe results
print(results[0].boxes)
x1, y1, x2, y2, _, _ = tuple(int(item) for item in results[0].boxes.data.numpy()[0])
print(x1, y1, x2, y2)

# crop the table
img = np.array(Image.open(image_path))
offset = 30
cropped_image = img[y1 - offset : y2 + offset, x1 - offset : x2 + offset]
Image.fromarray(cropped_image).save(f"{image_folder}/{image_name}_table_cropped.png")


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

    # # Remove vertical lines
    # vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 60))
    # detect_vertical = cv2.morphologyEx(
    #     thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2
    # )
    # cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # for c in cnts:
    #     cv2.drawContours(image, [c], -1, (255, 255, 255), 4)

    # denoise the image
    image = cv2.fastNlMeansDenoisingColored(image, None, 30, 30, 7, 21)

    return image


preprocessed_image = preprocess_image(cropped_image)
Image.fromarray(preprocessed_image).save(
    f"{image_folder}/{image_name}_preprocessed_table_cropped.png"
)

# extract text from the image
custom_config = r"--oem 3 --psm 6"
text = pytesseract.image_to_string(preprocessed_image, config=custom_config)
print(text)

with open(f"{image_folder}/{image_name}_table_result.txt", "w") as file:
    file.write(text)


# lines = text.strip().split("\n")
# lines = list(filter(lambda x: x.strip() != "", lines))
# print(lines)
# cleaned_result = "\n".join(["\t".join(line.split()) for line in lines])
# cleaned_result_list = [line.split() for line in lines]
# print(cleaned_result)

# csv_file = "revenue_table_test.csv"
# file = open(csv_file, "w", newline="")
# writer = csv.writer(file)

# for line in cleaned_result_list:
#     print(line)
#     writer.writerow()

# file.close()

# print(df)

render = render_result(model=model, image=image_path, result=results[0])
render.save(f"{image_folder}/{image_name}_table_result.png")
