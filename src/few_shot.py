# %%
import os
import sys
import time

import dotenv
import openai
from IPython.display import display

from utils import (
    check_environment_variables,
    eval_results,
    load_gold_standards,
    load_parallel_corpus,
    n_shot_prompting,
)

# How many samples to test
N_SAMPLES = 1000

num_shots = [1, 5, 10, 15, 20]

project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
dotenv_path = os.path.join(project_dir, ".env")
dotenv.load_dotenv(dotenv_path)

# Load constants from environment variabless
INUKTITUT_SYLLABIC_PATH = (
    "/Users/cambish/indigenous-mt/data/preprocessed/inuktitut-syllabic/tc/test"
)
GOLD_STANDARD_PATH = "/Users/cambish/indigenous-mt/data/external/Nunavut-Hansard-Inuktitut-English-Parallel-Corpus-3.0/gold-standard/"

# Check if required environment variables are set
try:
    check_environment_variables()
except KeyError:
    print(
        "Error: Required environment variables are not set"
    )  # TODO include which variables are missing
    sys.exit()

# Load constants from environment variables
openai.api_key = os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("MODEL")
SOURCE_LANGUAGE = os.environ.get("SOURCE_LANGUAGE")
TARGET_LANGUAGE = os.environ.get("TARGET_LANGUAGE")
TEXT_DOMAIN = os.environ.get("TEXT_DOMAIN")

# Check if OUTFILE directory does not exist, make it
if not os.path.exists(f"results/{MODEL}"):
    os.makedirs(f"results/{MODEL}")

# Load all relevant data, such as gold standard and parallel corpus data
df = load_parallel_corpus(INUKTITUT_SYLLABIC_PATH)
gold_standard_df = load_gold_standards(GOLD_STANDARD_PATH)

display(df)
display(gold_standard_df)
# %%

df_subset = df.sample(n=N_SAMPLES)

# System message to prime assisstant for translation
sys_msg = f"""You are a machine translation system that operates in two steps.

Step 1 - The user will provide {SOURCE_LANGUAGE} text denoted by "Text: ".
Transliterate the text to roman characters with a prefix that says "Romanization: ".

Step 2 - Translate the romanized text from step 1 into {TARGET_LANGUAGE} with a prefix that says "Translation: " ###
"""

# Perform n-shot promptings with varied number of examples
for n_shots in num_shots:
    # measure time taken
    start = time.time()
    # prompt LLM
    rdf = n_shot_prompting(sys_msg, gold_standard_df, df_subset, n_shots, N_SAMPLES)
    end = time.time()
    print(f"Time taken for {n_shots} experiment: {end - start}")
    # estimate average time per sample
    avg_time = (end - start) / N_SAMPLES
    print(f"Average time per sample: {avg_time}")
    print(rdf)
    rdf = eval_results(rdf)
    out_path = f"results/{MODEL}/{N_SAMPLES}-few_shot-{n_shots}.pkl"
    rdf.to_pickle(out_path)

# %%
