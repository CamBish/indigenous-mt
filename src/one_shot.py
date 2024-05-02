import re
import sys

import pandas as pd
from dotenv import load_dotenv
from IPython import display
from nltk.translate.bleu_score import sentence_bleu

from utils import (
    chat_completion_openai_api,
    check_environment_variables,
    load_parallel_corpus,
)

load_dotenv()

# constants
GPT_MODEL = "gpt-3.5-turbo-0613"
N = 1000
XML_FILE_PATH = "gold-standard/annotator1/Hansard_19990401.en.xml"
SRC = "Inuktitut"  # source language
TGT = "English"
DOM = "Hearings from a Legislative Assembly"
PATH = "inuk_data/norm/test"
OUTFILE = "res.csv"

try:
    check_environment_variables()
except KeyError:
    print(
        "Error: Required environment variables are not set"
    )  # TODO include which variables are missing
    sys.exit()

df = load_parallel_corpus(PATH)

subset = df.sample(n=N)
display.display(subset)


cols = ["src_txt", "tgt_txt", "rom_txt", "trans_txt"]
results = []

for _, row in subset.iterrows():
    src_txt = f'[{row["source_text"]}]'
    tgt_txt = row["target_text"]
    message = [
        {
            "role": "system",
            "content": f'You are a machine translation system that operates in two steps. Step 1 - The user will provide {SRC} text within square brackets. Romanize the text for use in the next step with a prefix that says "Romanization: ". Step 2 - Translate the romanized text from step 1 into {TGT} with a prefix that says "Translation: "',
        },
        {
            "role": "user",
            "content": f"Please provide the {TGT} translation for the following sentences: {src_txt}",
        },
    ]
    res = chat_completion_openai_api(messages=message)

    pred_txt = res["choices"][0]["message"]["content"]

    rom_match = re.search(r"Romanization: (.+?)\n", pred_txt)
    trans_match = re.search(r"Translation: (.+?)$", pred_txt)

    rom_txt = rom_match[1].strip("[]") if rom_match else ""
    trans_txt = trans_match[1].strip("[]") if trans_match else ""

    src_txt = src_txt.strip("[]")

    print("Romanized Text:", rom_txt)
    print("Translated Text:", trans_txt)
    print("Actual translation: ", tgt_txt)

    results.append(
        {
            "src_txt": src_txt,
            "tgt_txt": tgt_txt,
            "rom_txt": rom_txt,
            "trans_txt": trans_txt,
        }
    )

rdf = pd.DataFrame(results, columns=cols)
display(rdf)
rdf.to_pickle("results1.pkl")


res_df = pd.read_pickle("results1.pkl")
bleu_scores = []
for _, row in res_df.iterrows():
    reference = row["tgt_txt"].split()
    prediction = row["trans_txt"].split()

    bleu = sentence_bleu([reference], prediction)
    bleu_scores.append(bleu)


res_df["bleu_scores"] = bleu_scores

display(res_df)
avg_bleu = sum(bleu_scores) / len(bleu_scores)
print(avg_bleu)
max_bleu = max(bleu_scores)
print(max_bleu)
