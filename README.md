# revenue-table

## Instruction
The whole pipelines are implemented in [revenue_extractor.py](revenue_extractor.py). The usage is as follows:   

```python
pdf_path = ...
api_key = ...
extractor = RevenueExtractor(pdf_path, api_key)

# extract revenue table
table_results, confidence = extractor.extract_revenue_table()

# extract total revenue
total_revenue = extractor.extract_total_revenue()

# extract segments with the corresponding values
segments = extractor.extract_segments()

# extract sentences related to other segments
segment_sentences = extractor.extract_sentences()
```

To change the LLM, please change the `self._llm` in the constructor. Also, change the LLM calling process in all the extract function, i.e.
```python
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
```
The sample outputs are shown in [one_result.json](one_result.json)
