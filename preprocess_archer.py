import json

from pathlib import Path

import lxml.etree as etree

from spacy.lang.en import English


tokenizer = English().tokenizer

root = Path("data/ARCHER3/ARCHER_3-2_TXT")
records = []
for folder in root.iterdir():
    if folder.is_dir():
        for fn in folder.iterdir():
            if fn.suffix == ".txt":
                with fn.open(encoding="latin-1") as f:
                    record = {}
                    for line in f:
                        if line.startswith("<"):
                            record[line[1:line.index("=")].strip()] = line[line.index("=") + 1: -2].strip()
                        else:
                            break
                    text = f.read()
                    tokens = tokenizer(text)
                    record["text"] = [token.text for token in tokens]
                    records.append(record)

with open("data/archer.json", "w") as f:
    json.dump(records, f, indent=4)
                    

