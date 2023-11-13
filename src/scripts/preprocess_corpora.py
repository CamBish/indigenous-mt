"""
This module provides functions for preprocessing Cree and English Corpora text files.

Functions:
- preprocess: Preprocesses a text file and writes the preprocessed lines to a new file.
- preprocess_cree: Preprocesses the content of a file in Cree language.
- preprocess_english: Preprocesses an English text file.
"""
import argparse
import re

CREE = "cr"
ENGLISH = "en"

def preprocess(input_file_path: str, output_file_path: str, file_language: str) -> None:
    """
    Preprocesses a text file and writes the preprocessed lines to a new file.
    ...
    """
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines: list[str] = file.readlines()
    if file_language == CREE:
        preprocessed_lines = preprocess_cree(lines)
    elif file_language == ENGLISH:
        preprocessed_lines = preprocess_english(input_file_path)
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for line in preprocessed_lines:
            file.write(line + '\n')
    print(f"""
          Preprocessed {input_file_path} with 
          {'Cree' if file_language == 'cr' else 'English'} 
          and saved to {output_file_path}.
          """)

def preprocess_cree(lines: list[str]) -> list[str]:
    """
    Preprocesses the content of a file in Cree language.
    
    Reads the content of the file specified by `filename` 
    and performs a series of preprocessing steps:
    - Lowercases all text.
    - Converts all accents to hat accents instead of bar.
    - Removes ellipsis.
    - Puts a space before punctuation marks.
    - Removes double quotes.
    - Inserts an 'h' between vowels on either side of a hyphen.
    - Removes hyphens.
    - Removes numbers.
    - Removes extra spaces.
    - Removes anything other than essential characters.
    
    Parameters:
    - filename (str): The name of the file to be preprocessed.
    
    Returns:
    - new_lines (list): A list of preprocessed lines from the file.
    """
    for line in lines:
        line = line.strip()
        # Lowercase everything
        line = line.lower()
        # Convert all accents to hat accents instead of bar
        line = re.sub(r"ā", r"â", line)
        line = re.sub(r"ē", r"ê", line)
        line = re.sub(r"ī", r"î", line)
        line = re.sub(r"ō", r"ô", line)
        line = re.sub(r"á", r"â", line)
        line = re.sub(r"é", r"ê", line)
        line = re.sub(r"í", r"î", line)
        line = re.sub(r"ó", r"ô", line)
        # Remove ellipsis
        line = re.sub(r'(\.){2}', '', line)
        # Put a space before punctuation
        line = re.sub(r"(['‘’:,!.?])", r" \1", line)   
        # Remove double quotes
        line = re.sub(r'“', '', line)  
        line = re.sub(r'”', '', line)  
        line = re.sub(r'"', '', line)  
        # If vowel on either side of the hyphen, insert an 'h'
        line = re.sub(r'([aioâêîô])\s*[-—]\s*([aioâêîô])', r'\1h\2', line)
        # Remove hyphen and replace with nothing because two parts should be together
        line = re.sub(r'\s*[-—]\s*', r'', line)
        # Remove numbers
        line = re.sub(r'[0-9]', '', line)
        # Remove extra spaces
        line = re.sub(r'\s+', r' ', line)
        # Remove anything other than essential characters
        line = re.sub(r"[^a-zA-Zâêîô.!?:,'‘’]+", r' ', line)
    return lines

def preprocess_english(lines: list[str]) -> list[str]:
    """
    Preprocesses an English text file by performing the following operations:
    1. Lowercases all text.
    2. Removes ellipsis.
    3. Inserts spaces before punctuation.
    4. Removes double quotes.
    5. Replaces hyphens with spaces.
    6. Removes numbers.
    7. Removes extra spaces.
    8. Removes any characters that are not essential.
    
    Parameters:
        filename (str): The name of the text file to preprocess.
        
    Returns:
        List[str]: A list of preprocessed lines from the text file.
    """
    for line in lines:
        line = line.strip()
        # Lowercase everything
        line = line.lower()
        # Remove ellipsis
        line = re.sub(r'(\.){2,}', '', line)
        # Put spaces before punctuation
        line = re.sub(r"(['‘’:,!.?])", r" \1", line)   
        # Remove double quotes
        line = re.sub(r'“', '', line)  
        line = re.sub(r'”', '', line)
        line = re.sub(r'"', '', line)  
        # Remove hyphen and replace with space because two words can standalone mostly in English
        line = re.sub(r'\s*[‐—]\s*', r' ', line)
        # Remove numbers
        line = re.sub(r'[0-9]', '', line)
        # Remove extra spaces
        line = re.sub(r'\s+', r' ', line)
        # Remove anything other than essential characters
        line = re.sub(r"[^a-zA-Z.!?:,'‘’]+", r' ', line)
    return lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-filename", "-f",help="Name of text file to preprocess.")
    parser.add_argument("-output_path", "-o",help="Name of output processed file.")
    parser.add_argument("-mode", "-m", help="Cree (cr) or English (en).")

    args = vars(parser.parse_args())
    filename = args['filename']
    output_path = args['output_path']
    mode = args['mode']
    preprocess(filename, output_path, mode)
