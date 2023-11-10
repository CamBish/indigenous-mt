import contextlib
import re
import pandas as pd
import xml.etree.ElementTree as ET
from tenacity import retry, wait_random_exponential, stop_after_attempt
from nltk.translate.bleu_score import sentence_bleu
import openai


def extract_and_align_gold_standard(file_prefix: str):
    """
    Extracts and aligns the text stored within the gold standards using a specified file prefix.

    Args:
        file_prefix (str): The relative file path to the gold standard, without any of the suffixes (e.g., IU-EN-Parallel-Corpus/gold-standard/annotator1-consensus/Hansard_19990401)
    """

    # Define filepaths
    metadata_file = f"{file_prefix}.en.iu.conf"
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
    zeroes = re.compile(r'0')  # regex to find zeroes in the alignment file
    links = [link for link in alignment_root.iter('link') if not zeroes.search(link.get('type'))]

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
        'src_lang': [],
        'tgt_lang': [],
    }
    
    # Iterate through the links in the alignment file
    for link in links:
        # Get the xtargets and link type
        xtargets = link.get('xtargets')
        
        # Split the xtargets into source and target ids
        src_ids, tgt_ids = xtargets.split(';')

        # Split IDs into parts based on the link type
        src_ids = src_ids.split(' ')
        tgt_ids = tgt_ids.split(' ')
        src_phrases = [inuktitut_root.find(f"./p/s[@id='{sid}']").text for sid in src_ids]
        tgt_phrases = [english_root.find(f"./p/s[@id='{tid}']").text for tid in tgt_ids]

        # Convert src_phrases list into one string
        src_text = ' '.join(src_phrases)
        tgt_text = ' '.join(tgt_phrases)
        
        # Add the extracted data to the data dictionary
        data['src_lang'].append(src_text)
        data['tgt_lang'].append(tgt_text)
    
    return pd.DataFrame(data)

def load_parallel_corpus(path:str):
    """Loads data from parallel corpus files specified by 

    Args:
        path (str): Filepath to load without file extension

    Returns:
        pd.DataFrame: Dataframe with data from parallel corpus
    """    
    data = {
        'source_text': [],
        'target_text': []
    }
    source_filename = f"{path}.en"
    target_filename = f"{path}.iu"

    with open(source_filename, 'r', encoding='utf-8') as f1, open(target_filename, 'r', encoding='utf-8') as f2:
        for line1, line2 in zip(f1, f2):
            line1 = line1.strip()
            line2 = line2.strip()

            if line1 and line2:  # Only add lines with content
                data['target_text'].append(line1)
                data['source_text'].append(line2)
    return pd.DataFrame(data)

def eval_results(res_df:pd.DataFrame):
    """
    Calculates the BLEU scores for each translation in the given DataFrame and adds the scores as a new column.
    
    Args:
        res_df (pd.DataFrame): The DataFrame containing the translation results. It should have 'tgt_txt' and 'trans_txt' columns.
        
    Returns:
        pd.DataFrame: The input DataFrame with an additional 'bleu_scores' column containing the BLEU scores for each translation.
    """
    bleu_scores = []
    for idx, row in res_df.iterrows():
        reference = row['tgt_txt'].split()
        prediction = row['trans_txt'].split()
    
        bleu = sentence_bleu([reference], prediction)
        bleu_scores.append(bleu)

    res_df['bleu_scores'] = bleu_scores
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f'Average BLEU score: {avg_bleu}') 
    max_bleu = max(bleu_scores)
    print(f'Max BLEU score: {max_bleu}') 
    
    return res_df
