import json
import os
from openai import OpenAI
import re
import fitz
from collections import defaultdict

pdf_path = "pdf_sample/000d93bd-8f91-30cf-97ab-e612ba04b3f5.pdf"
pdf_name = os.path.basename(pdf_path).split(".")[0]

with open("final_results.json", "r") as f:
    data = json.load(f)

llm = OpenAI(api_key="sk-436808c023b34f4185c96d8d438aa4a3", base_url="https://api.deepseek.com")

tables = data[pdf_name]["tables"]

all_tables_text = ""
for page_num, text in tables.items():
    all_tables_text += text + f"\n{'-'*80}\n"

segment_prompt = "You are a finance expert. You are given a revenue table in a text format extracted from an annual report. Your task is to return the segments that form the revenue, and the corresponding values with the exact value and unit such as RMB or US, and in thousands or millions. Please extract the information from only one table, without combining the results of all the tables. You must output all the segments in a json format: {'segment_name': {'year 1': value, 'year 2': 'value'}}. No need to output other information."

def convert_json(text: str) -> dict:
    # Use regular expression to extract the JSON string inside the Markdown code block
    json_match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if not json_match:
        return {}

    json_str = json_match.group(1)

    # Load the JSON string into a dictionary
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return {}

    return data

def extract_segments():
    response = llm.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": segment_prompt},
            {"role": "user", "content": all_tables_text},
        ],
        temperature=0,
        stream=False,
    )

    output = response.choices[0].message.content

    segments = convert_json(output)
    segments = segments

    return segments

segments = extract_segments()

segment_names = list(segments.keys())
print(segment_names)

document = fitz.open(pdf_path)

found_pages = 0
segments_pages = defaultdict(list)

for page_num, page in enumerate(document):
    page_num += 1
    text = page.get_textpage().extractText()
    
    for name in segment_names:
        if name in text:
            print(f"Segment '{name}' found on page {page_num}")
            found_pages += 1
            segments_pages[name].append(page_num)

print(f"Found {found_pages} pages with segments")
print(segments_pages)