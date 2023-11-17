import os
import re
import xml.etree.ElementTree as ET

import openai
import pandas as pd
from dotenv import load_dotenv
from litellm import completion
from nltk.translate.bleu_score import sentence_bleu
from tenacity import retry, stop_after_attempt, wait_random_exponential

project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
dotenv_path = os.path.join(project_dir, ".env")
load_dotenv(dotenv_path)


def check_environment_variables():
    """
    Checks if all the required environment variables are set.
    This function checks if the following environment variables are set:
    - OPENAI_API_KEY: The API key for the OpenAI service.
    - MODEL: The name of the model to be used.
    - SOURCE_LANGUAGE: The source language for translation.
    - TARGET_LANGUAGE: The target language for translation.
    - N_SAMPLES: The number of samples to generate during translation.
    - MAX_N_SHOTS: The maximum number of shots to take during translation.
    - TEXT_DOMAIN: The domain of the text to be translated.
    If any of these environment variables are not set, a KeyError is raised.
    """

    if not all(
        [
            os.environ.get("OPENAI_API_KEY"),
            os.environ.get("MODEL"),
            os.environ.get("SOURCE_LANGUAGE"),
            os.environ.get("TARGET_LANGUAGE"),
            # os.environ.get("N_SAMPLES"),
            # os.environ.get("MAX_N_SHOTS"),
            os.environ.get("TEXT_DOMAIN"),
        ]
    ):
        raise KeyError("Error: Required environment variables are not set")


def extract_and_align_gold_standard(file_prefix: str):
    """
    Extracts and aligns the text stored within the gold standards using a specified file prefix.

    Args:
        file_prefix (str): The relative file path to the gold standard, without any of the suffixes (e.g., IU-EN-Parallel-Corpus/gold-standard/annotator1-consensus/Hansard_19990401)
    """

    # Define filepaths
    # metadata_file = f"{file_prefix}.en.iu.conf"
    alignment_file = f"{file_prefix}.en.iu.xml"
    english_file = f"{file_prefix}.en.xml"
    inuktitut_file = f"{file_prefix}.iu.xml"

    # Parse the alignment file to get alignment information
    alignment_tree = ET.parse(alignment_file)
    alignment_root = alignment_tree.getroot()

    # Parse the English and Inuktitut files
    english_tree = ET.parse(english_file)
    inuktitut_tree = ET.parse(inuktitut_file)
    english_root = english_tree.getroot()
    inuktitut_root = inuktitut_tree.getroot()

    # Remove any 0-Many or Many-0 links
    zeroes = re.compile(r"0")  # regex to find zeroes in the alignment file
    links = [
        link
        for link in alignment_root.iter("link")
        if not zeroes.search(link.get("type"))
    ]

    # Return the linked gold standard as a DataFrame
    return link_gold_standard(links, inuktitut_root, english_root)


def link_gold_standard(links, inuktitut_root, english_root):
    """
    Extracts data from the given links and returns a pandas DataFrame.

    Parameters:
    - links (list): A list of links in the alignment file.
    - inuktitut_root (ElementTree.Element): The root element of the Inuktitut XML file.
    - english_root (ElementTree.Element): The root element of the English XML file.

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the extracted data.
    """
    # Create a dictionary to store the extracted data
    data = {
        "src_lang": [],
        "tgt_lang": [],
    }

    # Iterate through the links in the alignment file
    for link in links:
        # Get the xtargets and link type
        xtargets = link.get("xtargets")

        # Split the xtargets into source and target ids
        src_ids, tgt_ids = xtargets.split(";")

        # Split IDs into parts based on the link type
        src_ids = src_ids.split(" ")
        tgt_ids = tgt_ids.split(" ")
        src_phrases = [
            inuktitut_root.find(f"./p/s[@id='{sid}']").text for sid in src_ids
        ]
        tgt_phrases = [english_root.find(f"./p/s[@id='{tid}']").text for tid in tgt_ids]

        # Convert src_phrases list into one string
        src_text = " ".join(src_phrases)
        tgt_text = " ".join(tgt_phrases)

        # Add the extracted data to the data dictionary
        data["src_lang"].append(src_text)
        data["tgt_lang"].append(tgt_text)

    return pd.DataFrame(data)


def load_parallel_corpus(path: str):
    """Loads data from parallel corpus files specified by

    Args:
        path (str): Filepath to load without file extension

    Returns:
        pd.DataFrame: Dataframe with data from parallel corpus
    """
    data: dict = {"source_text": [], "target_text": []}
    source_filename = f"{path}.en"
    target_filename = f"{path}.iu"

    with open(source_filename, "r", encoding="utf-8") as f1, open(
        target_filename, "r", encoding="utf-8"
    ) as f2:
        for line1, line2 in zip(f1, f2):
            line1 = line1.strip()
            line2 = line2.strip()

            if line1 and line2:  # Only add lines with content
                data["target_text"].append(line1)
                data["source_text"].append(line2)
    return pd.DataFrame(data)


def eval_results(res_df: pd.DataFrame):
    """
    Calculates the BLEU scores for each translation in the given DataFrame and adds the scores as a new column.
    Args:
        res_df (pd.DataFrame): The DataFrame containing the translation results. It should have 'tgt_txt' and 'trans_txt' columns.
    Returns:
        pd.DataFrame: The input DataFrame with an additional 'bleu_scores' column containing the BLEU scores for each translation.
    """
    bleu_scores = []
    for row in res_df.iterrows():
        reference = row["tgt_txt"].split()
        prediction = row["trans_txt"].split()

        bleu = sentence_bleu([reference], prediction)
        bleu_scores.append(bleu)

    res_df["bleu_scores"] = bleu_scores
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"Average BLEU score: {avg_bleu}")
    max_bleu = max(bleu_scores)
    print(f"Max BLEU score: {max_bleu}")

    return res_df


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def chat_completion_request_api(
    messages,
    functions=None,
    function_call=None,
    temperature=0,
    max_tokens=None,
    stop=None,
    n=None,
    model=None,
):
    """
    Wrapper function for creating chat completion request through OpenAI.

    Args:
        messages (list): A list of messages for the chat completion.
        functions (str, optional): A string representing the functions to be used.
        function_call (str, optional): A string representing the function call to be used.
        temperature (float, optional): A float representing the temperature for generating text. Defaults to 0.
        max_tokens (int, optional): An integer representing the maximum number of tokens. Defaults to None.
        stop (str, optional): A string representing the stopping condition for generating text. Defaults to None.
        n (int, optional): An integer representing the number of completions to generate. Defaults to None.
        model (str, optional): A string representing the model to be used. Defaults to "MODEL".

    Returns:
        ChatCompletion: An instance of the ChatCompletion class.
    """
    model = os.environ.get("MODEL", "gpt-3.5-turbo")
    json_data = {"model": model, "messages": messages}
    if functions is not None:
        json_data["functions"] = functions
    if function_call is not None:
        json_data["function_call"] = function_call
    if temperature is not None:
        json_data["temperature"] = temperature
    if max_tokens is not None:
        json_data["max_tokens"] = max_tokens
    if stop is not None:
        json_data["stop"] = stop
    if n is not None:
        json_data["n"] = n

    try:
        return completion(**json_data)
    except openai.OpenAIError as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


def n_shot_prompting(sys_msg, gold_std, pll_corpus, n_shots, n_samples):
    """
    Perform n-shot prompting using a system message, gold standard, and parallel corpus.

    Args:
        sys_msg (str): The system message to include in the chat conversation.
        gold_std (pandas.DataFrame): The gold standard dataframe.
        pll_corpus (pandas.DataFrame): The parallel corpus dataframe.
        n_shots (int): The number of examples to include from the gold standard.
        n_samples (int): The number of examples to include from the PLL corpus.

    Returns:
        pandas.DataFrame: A dataframe containing the results of the n-shot prompting.
    """

    TARGET_LANGUAGE = os.environ.get("TARGET_LANGUAGE")
    # Select a random subset of examples from the gold standard and PLL corpus
    gs_subset = gold_std.sample(n=n_shots, replace=False)
    pc_subset = pll_corpus.sample(n=n_samples, replace=False)

    cols = ["src_txt", "tgt_txt", "rom_txt", "trans_txt"]
    results = []

    # Empty string to store examples for few-shot learning
    examples = ""

    # TODO move into separate function
    # Parse gold standard dataframe to get examples
    for row in gs_subset.iterrows():
        src_example = row["src_lang"]
        tgt_example = row["tgt_lang"]

        examples += f"Text: {src_example} | Translation: {tgt_example} ###\n"

    # Iterate over the dataframe subset
    for row in pc_subset.iterrows():
        src_txt = row["source_text"]
        tgt_txt = row["target_text"]

        # Create the chat conversation messages
        message = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": examples},
            {
                "role": "user",
                "content": f"Provide the {TARGET_LANGUAGE} transliteration and translation for the following text: {src_txt} ###",
            },
        ]
        max_attempts = 5
        attempts = 0

        while attempts < max_attempts:
            try:
                # TODO should be made into a function
                res = chat_completion_request_api(messages=message)
                pred_txt = res["choices"][0]["message"]["content"]
                rom_txt = (
                    re.search(r"Romanization: (.+?)\n", pred_txt).group(1).strip('[]""')
                )
                trans_txt = (
                    re.search(r"Translation: (.+?)$", pred_txt).group(1).strip('[]""')
                )

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

                break  # Exit the loop if the API call is successful

            # TODO change exeception to be more specific
            except Exception as err:
                print(f"Exception: {err}")
                attempts += 1

        if attempts == max_attempts:
            print("Max number of attempts reached. Skipping this example.")

    return pd.DataFrame(results, columns=cols)