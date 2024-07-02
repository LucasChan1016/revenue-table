import json
import re

with open("one_result.json", "r") as f:
    data = json.load(f)

sentences = data["segment_sentences"]

# print(sentences)

def merge_dicts(sentences):
    def can_convert_float(x):
        try:
            float(x.replace(",", ""))
            return True
        except ValueError:
            return False
    merged = {}
    for key, list_of_dicts in sentences.items():
        merged[key] = {}
        for d in list_of_dicts:
            for k, v in d.items():
                if v is None or len(v) == 0:
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

merged = merge_dicts(sentences)

with open("merged.json", "w") as f:
    json.dump(merged, f, indent=4)
