import json
import os
import re

import pandas as pd
from natsort import natsorted
from openai import OpenAI

client = OpenAI(
    api_key="sk-436808c023b34f4185c96d8d438aa4a3", base_url="https://api.deepseek.com"
)
prompt = "You are a finance expert. You are given a table in a text format extracted from an annual report. Your task is to determine if the table is a revenue table or not. If the table contains other information such as expenses and dividend, it is not considered as a revenue table. The table should include the total value of the most recent year. You just need to output the confidence level of the table being a revenue table in the format of 'confidence level: {your output}'"

confidences = {}

for text_file in natsorted(os.listdir("filtered_tables")):
    if ".txt" not in text_file or "_AIoutput" in text_file:
        continue
    with open(f"filtered_tables/{text_file}", "rb") as f:
        data = str(f.read())

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": data},
        ],
        temperature=0,
        stream=False,
    )
    # print("usage : ", response.usage.total_tokens)
    # print("=" * 50)

    output = response.choices[0].message.content
    # print(output)
    with open(f"filtered_tables/{text_file.split('.')[0]}_AIoutput.txt", "w") as f:
        f.write(output)

    pattern = r"confidence level: (\d+(\.\d+)?)"
    match = re.search(pattern, output)

    confidence_level = float(match.group(1)) if match else 0

    confidences[text_file] = confidence_level

print(confidences)
# find the max confidence level
max_confidence = max(confidences.values())
max_files = [
    file for file, confidence in confidences.items() if confidence == max_confidence
]
print(max_files)
