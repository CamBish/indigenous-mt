import os
import re
from enum import Enum
from typing import Dict, List, Set, Tuple

import openai
import pandas as pd

# import pyarrow.parquet as pq
from defusedxml import ElementTree as ET
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu

from chat import ollama_chat_completion, openai_chat_completion

project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
dotenv_path = os.path.join(project_dir, ".env")
load_dotenv(dotenv_path)


class GoldStandardMode(Enum):
    CONSENSUS = "consensus"
    INDIVIDUAL = "individual"


class LanguageMode(Enum):
    INUKTITUT = "inuktitut"
    CREE = "cree"


def n_shot_prompting_openai(sys_msg, gold_std, pll_corpus, n_shots, n_samples):
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
    pc_subset = pll_corpus.sample(n=n_samples, replace=False)

    cols = ["source_text", "target_text", "romanized_text", "translated_text"]
    results = []

    examples = generate_n_shot_examples(gold_std, n_shots)

    max_attempts = 5
    # Iterate over the dataframe subset
    for _, row in pc_subset.iterrows():
        src_txt = row["source_text"]
        # Create the chat conversation messages
        message = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": examples},
            {
                "role": "user",
                "content": f"Provide the {TARGET_LANGUAGE} transliteration and translation for the following text: {src_txt} ###",
            },
        ]
        attempts = 0

        tgt_txt = row["target_text"]
        # Loop until the API call is successful or the maximum number of attempts is reached
        while attempts < max_attempts:
            try:
                # TODO should be made into a function
                res = openai_chat_completion(messages=message)
                pred_txt = res["choices"][0]["message"]["content"]
                if pred_txt is None or pred_txt == "":
                    attempts += 1
                    openai.OpenAIError("Empty response. Trying again...")

                rom_match = re.search(r"Romanization: (.+?)\n", pred_txt)
                trans_match = re.search(r"Translation: (.+?)$", pred_txt)

                if rom_match is None or trans_match is None:
                    attempts += 1
                    openai.OpenAIError("Invalid output from LLM. Trying again...")
                    continue

                rom_txt = rom_match[1].strip("[]")
                trans_txt = trans_match[1].strip("[]")

                src_txt = src_txt.strip("[]")

                print("Romanized Text:", rom_txt)
                print("Translated Text:", trans_txt)
                print("Actual translation: ", tgt_txt)

                results.append(
                    {
                        "source_text": src_txt,
                        "target_text": tgt_txt,
                        "romanized_text": rom_txt,
                        "translated_text": trans_txt,
                    }
                )

                break  # Exit the loop if the API call is successful

            except OpenAIError as err:
                print(f"Exception: {err}")
                attempts += 1

        if attempts == max_attempts:
            print("Max number of attempts reached. Skipping this example.")

    return pd.DataFrame(results, columns=cols)


def eval_results(res_df: pd.DataFrame):
    """
    Calculates the BLEU scores for each translation in the given DataFrame and adds the scores as a new column.
    Args:
        res_df (pd.DataFrame): The DataFrame containing the translation results. It should have 'tgt_txt' and 'trans_txt' columns.
    Returns:
        pd.DataFrame: The input DataFrame with an additional 'bleu_scores' column containing the BLEU scores for each translation.
    """
    bleu_scores = []
    for _, row in res_df.iterrows():
        reference = row["target_text"].split()
        prediction = row["translated_text"].split()

        bleu = sentence_bleu([reference], prediction)
        bleu_scores.append(bleu)

    res_df["bleu_scores"] = bleu_scores
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"Average BLEU score: {avg_bleu}")
    max_bleu = max(bleu_scores)
    print(f"Max BLEU score: {max_bleu}")

    return res_df


def has_parallel_cree_english_data(filename: str, filenames: List[str]) -> bool:
    """
    Check if a file has both a Cree and English version.

    Args:
        filename (str): The filename to check.
        filenames (List[str]): The list of filenames in the directory.

    Returns:
        bool: True if both versions exist, False otherwise.
    """
    if filename.endswith("_cr.txt"):
        english_filename = filename.replace("_cr.txt", "_en.txt")
        return english_filename in filenames
    return False


def get_filenames(input_directory: str) -> List[str]:
    """
    Get a list of filenames in the specified input directory.

    Args:
        input_directory (str): The path to the input directory.

    Returns:
        List[str]: A list of filenames in the input directory.
    """
    filenames = []
    for _, _, files in os.walk(input_directory):
        filenames.extend(files)
    return filenames


def get_file_prefixes(gs_1_path: str, gs_2_path: str) -> Set[str]:
    """
    Get unique file prefixes from the gold standard paths.

    Args:
        gs_1_path (str): The path to the first gold standard directory.
        gs_2_path (str): The path to the second gold standard directory.

    Returns:
        Set[str]: A set of unique file prefixes.
    """
    gs_1_files = os.listdir(gs_1_path)
    gs_1_prefixes = {
        os.path.join(gs_1_path, filename.split(".")[0]) for filename in gs_1_files
    }

    gs_2_files = os.listdir(gs_2_path)
    gs_2_prefixes = {
        os.path.join(gs_2_path, filename.split(".")[0]) for filename in gs_2_files
    }
    return gs_1_prefixes.union(gs_2_prefixes)


def load_parallel_text_data(
    source_directory: str,
    target_directory: str,
) -> Dict[str, List[str]]:
    """
    Load parallel text data from the source and target paths.

    Args:
        source_path (str): The path to the source text file.
        target_path (str): The path to the target text file.

    Returns:
        pd.DataFrame: The loaded parallel text data.
    """
    temp_data: Dict[str, List[str]] = {"source_text": [], "target_text": []}
    with open(source_directory, "r", encoding="utf-8") as source_file, open(
        target_directory, "r", encoding="utf-8"
    ) as target_file:
        for source_line, target_line in zip(source_file, target_file):
            source_line = source_line.strip()
            target_line = target_line.strip()
            if source_line and target_line:
                temp_data["source_text"].append(source_line)
                temp_data["target_text"].append(target_line)

    return temp_data


def load_cree_parallel_data(input_directory: str) -> pd.DataFrame:
    """load Cree data from specified directory into a Dataframe

    Args:
        input_directory (str): string containing input path

    Returns:
        pd.DataFrame: Dataframe with contents from parallel data
    """
    cree_text = []
    english_text = []
    filenames = get_filenames(input_directory)
    for filename in filenames:
        if has_parallel_cree_english_data(filename, filenames):
            source_path, target_path = get_cree_and_english_data_paths(
                input_directory, filename
            )
            temp_data = load_parallel_text_data(source_path, target_path)
            cree_text.extend(temp_data["source_text"])
            english_text.extend(temp_data["target_text"])
    return pd.DataFrame({"cree_text": cree_text, "english_text": english_text})


def get_cree_and_english_data_paths(root: str, filename: str) -> Tuple[str, str]:
    """
    Get the paths of the Cree and English files.

    Args:
        root (str): The root directory.
        filename (str): The filename.

    Returns:
        Tuple[str, str]: The paths of the Cree and English files.
    """
    source_path = os.path.join(root, filename)
    target_path = os.path.join(root, filename.replace("_cr.txt", "_en.txt"))
    return source_path, target_path


def serialize_gold_standards(
    input_path: str,
    output_path: str,
    mode: GoldStandardMode = GoldStandardMode.CONSENSUS,
):
    """
    Loads the gold standard data for a specified mode and file prefix.

    Args:
        gs_dir (str): The directory containing the gold standard files.
        mode (GoldStandardMode): Specifies which gold standard files to load.

    Returns:
        pd.DataFrame: The loaded gold standard data.
    """
    print("Loading Inuktitut gold standard")
    gold_standard_df = load_gold_standards(input_path, mode)
    gold_standard_df.to_parquet(output_path)


def serialize_parallel_corpus(
    input_path: str,
    output_path: str,
    mode: LanguageMode = LanguageMode.INUKTITUT,
):
    """
    Serializes the parallel corpus to a parquet file. Does not run if the file already exists.

    Args:
        path (str, optional): Filepath to save the serialized parallel corpus. Defaults to '/data/serialized/syllabic_parallel_corpus.parquet'.
    """
    if mode == LanguageMode.INUKTITUT:
        print("Serializing Inuktitut parallel corpus")
        parallel_corpus_df = load_inuktitut_parallel_corpus(input_path)
        parallel_corpus_df.to_parquet(output_path)
        return
    if mode == LanguageMode.CREE:
        print("Serializing Cree parallel corpus")
        parallel_corpus_df = load_cree_parallel_data(input_path)
        parallel_corpus_df.to_parquet(output_path)
        return


def extract_and_align_gold_standard(file_prefix: str):
    """
    Extracts and aligns the text stored within the gold standards using a specified file prefix.

    Args:
        file_prefix (str): The relative file path to the gold standard, without any of the suffixes (e.g., IU-EN-Parallel-Corpus/gold-standard/annotator1-consensus/Hansard_19990401)
    """

    # Define filepaths
    # metadata_file = f"{file_prefix}.en.iu.conf" # not needed
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


def load_consensus_gold_standards(gs_dir: str):
    """
    Load consensus gold standards from the given directory.

    Args:
        gs_dir (str): The directory path where the gold standards are located.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the concatenated gold standards.
    """
    gs_1_path = os.path.join(gs_dir, "annotator1-consensus")
    gs_2_path = os.path.join(gs_dir, "annotator2-consensus")
    print("loading consensus gold standard")

    file_prefixes = get_file_prefixes(gs_1_path, gs_2_path)

    gs_dfs = []

    for file_prefix in file_prefixes:
        df = extract_and_align_gold_standard(file_prefix)
        gs_dfs.append(df)

    return pd.concat(gs_dfs, ignore_index=True)


def load_individual_gold_standards(gs_dir: str):
    """
    Load individually annotated gold standards from the given directory.

    Args:
        gs_dir (str): The directory path where the gold standards are located.

    Returns:
        pd.DataFrame: A concatenated DataFrame of the gold standards.
    """
    gs_1_path = os.path.join(gs_dir, "annotator1")
    gs_2_path = os.path.join(gs_dir, "annotator2")
    print("loading individually annotated gold standard")

    file_prefixes = get_file_prefixes(gs_1_path, gs_2_path)

    gs_dfs = []

    for file_prefix in file_prefixes:
        df = extract_and_align_gold_standard(file_prefix)
        gs_dfs.append(df)

    return pd.concat(gs_dfs, ignore_index=True)


def load_gold_standards(
    gs_dir: str, mode: GoldStandardMode = GoldStandardMode.CONSENSUS
):
    """
    Loads the gold standard data for a specified mode and file prefix.

    Args:
        mode (GoldStandardMode): Specifies which gold standard files to load.
        gs_dir (str): filepath to gold standard directory (
            i.e. data/external/Nunavut-Hansard-Inuktitut-English-Parallel-Corpus-3.0/gold-standard/)
    """
    if mode == GoldStandardMode.CONSENSUS:
        return load_consensus_gold_standards(gs_dir)
    # else if mode == GoldStandardMode.INDIVIDUAL:
    return load_individual_gold_standards(gs_dir)


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
        "source_text": [],
        "target_text": [],
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
        data["source_text"].append(src_text)
        data["target_text"].append(tgt_text)

    return pd.DataFrame(data)


def load_inuktitut_parallel_corpus(path: str):
    """Loads data from parallel corpus files specified by

    Args:
        path (str): Filepath to load without file extension

    Returns:
        pd.DataFrame: Dataframe with data from parallel corpus
    """
    data: dict = {"source_text": [], "target_text": []}
    source_filename = f"{path}.iu"
    target_filename = f"{path}.en"

    # load data from source and target files using load_parallel_text_data
    temp_data = load_parallel_text_data(source_filename, target_filename)
    data["source_text"].extend(temp_data["source_text"])
    data["target_text"].extend(temp_data["target_text"])
    return pd.DataFrame(data)


def generate_n_shot_examples(gold_standard, n_shots):
    """
    Selects a random subset of examples from the gold standard.

    Args:
        gold_std (pandas.DataFrame): The gold standard dataframe.
        n_shots (int): The number of examples to include from the gold standard.

    Returns:
        str: A string containing the examples for few-shot learning.
    """
    # Select a random subset of examples from the gold standard
    gold_standard_subset = gold_standard.sample(n=n_shots, replace=False)

    # Empty string to store examples for few-shot learning
    examples = ""

    # Parse gold standard dataframe to get examples
    for _, row in gold_standard_subset.iterrows():
        source_example = row["source_text"]
        target_example = row["target_text"]

        examples += f"Text: {source_example} | Translation: {target_example} ###\n"

    return examples


# def check_environment_variables():
#     """
#     Checks if all the required environment variables are set.
#     This function checks if the following environment variables are set:
#     - OPENAI_API_KEY: The API key for the OpenAI service.
#     - MODEL: The name of the model to be used.
#     - SOURCE_LANGUAGE: The source language for translation.
#     - TARGET_LANGUAGE: The target language for translation.
#     - N_SAMPLES: The number of samples to generate during translation.
#     - MAX_N_SHOTS: The maximum number of shots to take during translation.
#     - TEXT_DOMAIN: The domain of the text to be translated.
#     If any of these environment variables are not set, a KeyError is raised.
#     """

#     if not all(
#         [
#             os.environ.get("OPENAI_API_KEY"),
#             os.environ.get("MODEL"),
#             os.environ.get("SOURCE_LANGUAGE"),
#             os.environ.get("TARGET_LANGUAGE"),
#             os.environ.get("TEXT_DOMAIN"),
#         ]
#     ):
#         raise KeyError("Error: Required environment variables are not set")
