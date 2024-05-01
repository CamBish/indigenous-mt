import os
import re
import time

import dotenv
import pandas as pd
import pyarrow.parquet as pq
from IPython.display import display

from utils import (
    LanguageMode,
    chat_completion_ollama_api,
    eval_results,
    serialize_parallel_corpus,
)

project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
dotenv_path = os.path.join(project_dir, ".env")
dotenv.load_dotenv(dotenv_path)

# constants
MODEL = ""
N_SAMPLES = 1000

TEST_INUKTITUT_SYLLABIC_PATH = os.path.join(
    project_dir,
    "data",
    "external",
    "Nunavut-Hansard-Inuktitut-English-Parallel-Corpus-3.0",
    "split",
    "test",
)

TEST_INUKTITUT_ROMAN_PATH = os.path.join(
    project_dir,
    "data",
    "external",
    "Nunavut-Hansard-Inuktitut-English-Parallel-Corpus-3.0",
    "split",
    "test",
)

TEST_SERIALIZED_INUKTITUT_SYLLABIC_PATH = os.path.join(
    project_dir, "data", "serialized", "test_syllabic_parallel_corpus.parquet"
)
TEST_SERIALIZED_INUKTITUT_ROMAN_PATH = os.path.join(
    project_dir, "data", "serialized", "test_roman_parallel_corpus.parquet"
)
SERIALIZED_GOLD_STANDARD_PATH = os.path.join(
    project_dir, "data", "serialized", "gold_standard.parquet"
)
SERIALIZED_CREE_PATH = os.path.join(
    project_dir, "data", "serialized", "cree_corpus.parquet"
)

SOURCE_LANGUAGE = os.environ.get("SOURCE_LANGUAGE")
TARGET_LANGUAGE = os.environ.get("TARGET_LANGUAGE")
TEXT_DOMAIN = os.environ.get("TEXT_DOMAIN")

if not os.path.exists(f"results/{MODEL}"):
    os.makedirs(f"results/{MODEL}")

serialize_parallel_corpus(
    input_path=TEST_INUKTITUT_SYLLABIC_PATH,
    output_path=TEST_SERIALIZED_INUKTITUT_SYLLABIC_PATH,
    mode=LanguageMode.INUKTITUT,
)
serialize_parallel_corpus(
    input_path=TEST_INUKTITUT_ROMAN_PATH,
    output_path=TEST_SERIALIZED_INUKTITUT_ROMAN_PATH,
    mode=LanguageMode.INUKTITUT,
)
serialize_parallel_corpus(
    input_path=SERIALIZED_CREE_PATH,
    output_path=SERIALIZED_CREE_PATH,
    mode=LanguageMode.CREE,
)

df = pq.read_table(TEST_SERIALIZED_INUKTITUT_SYLLABIC_PATH).to_pandas()
print("Serialized Data Loaded")

display(df)

df_subset = df.sample(n=N_SAMPLES, random_state=42)

print("Test Sample")
display(df_subset)

cols = ["src_txt", "tgt_txt", "rom_txt", "trans_txt"]
results = []

sys_msg = f"""You are a machine translation system that operates in two steps.

Step 1 - The user will provide {SOURCE_LANGUAGE} text denoted by "Text: ".
Transliterate the text to roman characters with a prefix that says "Romanization: ".

Step 2 - Translate the romanized text from step 1 into {TARGET_LANGUAGE} with a prefix that says "Translation: " ###
"""


for _, row in df_subset.iterrows():
    src_txt = f'[{row["source_text"]}]'
    tgt_txt = row["target_text"]
    usr_msg = f"""Provide the {TARGET_LANGUAGE} transliteration and translation for the following text: {src_txt}"""
    message = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": usr_msg},
    ]
    start = time.time()
    res = chat_completion_ollama_api(messages=message, model=MODEL)

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
rdf = eval_results(rdf)
display(rdf)
out_path = f"results/{MODEL}/{N_SAMPLES}-zero_shot.parquet"
rdf.to_parquet(out_path)
