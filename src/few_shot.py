# %%
import os
import time

import dotenv
import pyarrow.parquet as pq
from IPython.display import display

from utils import (
    LanguageMode,
    eval_results,
    n_shot_prompting_openai,
    serialize_gold_standards,
    serialize_parallel_corpus,
)

# How many samples to test
N_SAMPLES = 1000
MODEL = "ollama_chat/llama3"

num_shots = [1, 5, 10, 15, 20]

project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
dotenv_path = os.path.join(project_dir, ".env")
dotenv.load_dotenv(dotenv_path)

# Load constants from environment variabless
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

# Load constants from environment variables - this is for raw data
RAW_INUKTITUT_SYLLABIC_PATH = os.path.join(
    project_dir,
    "data",
    "external",
    "Nunavut-Hansard-Inuktitut-English-Parallel-Corpus-3.0",
    "split",
    "test",
)
RAW_INUKTITUT_ROMAN_PATH = os.path.join(
    project_dir,
    "data",
    "external",
    "Nunavut-Hansard-Inuktitut-English-Parallel-Corpus-3.0",
    "split",
    "test",
)

GOLD_STANDARD_PATH = os.path.join(
    project_dir,
    "data",
    "external",
    "Nunavut-Hansard-Inuktitut-English-Parallel-Corpus-3.0",
    "gold-standard",
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

# Check if required environment variables are set
# Not needed anymore since models are hosted locally
# try:
#     check_environment_variables()
# except KeyError:
#     print(
#         "Error: Required environment variables are not set"
#     )
#     sys.exit()

# Load constants from environment variables
# openai.api_key = os.environ.get("OPENAI_API_KEY")
# MODEL = os.environ.get("MODEL")
SOURCE_LANGUAGE = os.environ.get("SOURCE_LANGUAGE")
TARGET_LANGUAGE = os.environ.get("TARGET_LANGUAGE")
TEXT_DOMAIN = os.environ.get("TEXT_DOMAIN")

# Check if OUTFILE directory does not exist, make it
if not os.path.exists(f"results/{MODEL}"):
    os.makedirs(f"results/{MODEL}")

# if gold standard has not been serialized, do so using parquet
serialize_gold_standards(
    input_path=GOLD_STANDARD_PATH, output_path=SERIALIZED_GOLD_STANDARD_PATH
)

# if parallel corpus has not been serialized, do so using parquet
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

# Load serialized data
df = pq.read_table(TEST_SERIALIZED_INUKTITUT_SYLLABIC_PATH).to_pandas()
gold_standard_df = pq.read_table(SERIALIZED_GOLD_STANDARD_PATH).to_pandas()
print("Serialized Data loaded")


display(df)
display(gold_standard_df)
# %%

df_subset = df.sample(n=N_SAMPLES, random_state=42)

# System message to prime assisstant for translation
sys_msg = f"""You are a machine translation system that operates in two steps.

Step 1 - The user will provide {SOURCE_LANGUAGE} text denoted by "Text: ".
Transliterate the text to roman characters with a prefix that says "Romanization: ".

Step 2 - Translate the romanized text from step 1 into {TARGET_LANGUAGE} with a prefix that says "Translation: " ###
"""

# Perform n-shot promptings with varied number of examples
for n_shots in num_shots:
    print(f"Performing {n_shots} shot experiment")
    # measure time taken
    start = time.time()
    print(f"Start time: {start}")
    # prompt LLM
    rdf = n_shot_prompting_openai(
        sys_msg, gold_standard_df, df_subset, n_shots, N_SAMPLES
    )
    end = time.time()
    print(f"Time taken for {n_shots} experiment: {end - start}")
    # estimate average time per sample
    avg_time = (end - start) / N_SAMPLES
    print(f"Average time per sample: {avg_time}")
    print(rdf)
    rdf = eval_results(rdf)
    out_path = f"results/{MODEL}/{N_SAMPLES}-few_shot-{n_shots}.parquet"
    rdf.to_parquet(out_path)

# %%
