# Set imports, define constants, and load OpenAI API key
import os
import re
import time
import openai
import pandas as pd
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu
from scripts.utils import extract_and_align_gold_standard, load_parallel_corpus, eval_results
from scripts.chat import n_shot_prompting, chat_completion_request_API

load_dotenv()

#Constants
GPT_MODEL = "gpt-3.5-turbo" #which GPT model to call
SRC = "Inuktitut" #source language
TGT = "English" #taget language
DOM = "Hearings from a Legislative Assembly" #domain of text
PATH = "inuk_data/norm/test"
GS_PATH_PREFIX = 'IU-EN-Parallel-Corpus/gold-standard/annotator1-consensus/Hansard_19990401'
N_SAMPLES = 1000 #number of samples
MAX_N_SHOTS = 3 #number of shots

# Check if OUTFILE directory does not exist, make it
if not os.path.exists(f'results/{GPT_MODEL}'):
    os.makedirs(f'results/{GPT_MODEL}')

try:
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    if openai.api_key is None:
        raise Exception
    else:
        print("API Key Obtained Successfully!")
except Exception:
    print("Error reading OpenAI API key from environment variable")
    exit(1)

#%%
# Load all relevant data, such as gold standard and parallel corpus data
df = load_parallel_corpus(PATH)

gs_df = extract_and_align_gold_standard(GS_PATH_PREFIX)


df_subset = df.sample(n=N_SAMPLES)

# System message to prime assisstant for translation
sys_msg = f'''You are a machine translation system that operates in two steps. 

Step 1 - The user will provide {SRC} text denoted by "Text: ". 
Transliterate the text to roman characters with a prefix that says "Romanization: ". 

Step 2 - Translate the romanized text from step 1 into {TGT} with a prefix that says "Translation: " ###
'''

#%%

# Perform n-shot promptings with varied number of examples
for n_shots in range(1, MAX_N_SHOTS):
    #random subset of gold standard for few-shot examples
    gs_subset = gs_df.sample(n=n_shots)
    
    #measure time taken
    start = time.time()
    
    #prompt GPT
    rdf = n_shot_prompting(sys_msg, gs_subset, df_subset, n_shots, N_SAMPLES)
    
    end = time.time()
    print(f'Time taken for {n_shots} experiment: {end - start}')
    
    #estimate average time per sample
    avg_time = (end - start) / N_SAMPLES
    
    print(f'Average time per sample: {avg_time}')
    
    display(rdf)
    rdf = eval_results(rdf)
    out_path = f'results/{GPT_MODEL}/{N_SAMPLES}-few_shot-{n_shots}.pkl'
    
    rdf.to_pickle(out_path)

# %%
