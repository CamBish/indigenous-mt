"""
This module provides functions for preprocessing Cree and English Corpora text files.

Functions:
- preprocess: Preprocesses a text file and writes the preprocessed lines to a new file.
- preprocess_cree: Preprocesses the content of a file in Cree language.
- preprocess_english: Preprocesses an English text file.
"""
import os
import re

CREE = "cr"
ENGLISH = "en"


def preprocess(input_file_path: str, output_file_path: str) -> None:
    """
    Preprocesses an input file based on the provided file language and saves the preprocessed content to an output file.

    Parameters:
        input_file_path (str): The path to the input file.
        output_file_path (str): The path to the output file.
        file_language (str): The language of the input file.

    Returns:
        None
    """
    file_language = "en" if input_file_path.endswith("_en.txt") else "cr"
    with open(input_file_path, "r", encoding="utf-8") as file:
        lines: list[str] = file.readlines()
    if file_language == CREE:
        preprocessed_lines = preprocess_cree(lines)
    elif file_language == ENGLISH:
        preprocessed_lines = preprocess_english(lines)
    with open(output_file_path, "w", encoding="utf-8") as file:
        for line in preprocessed_lines:
            file.write(line + "\n")
    print(
        f"""
        Preprocessed {input_file_path} with
        {'Cree' if file_language == 'cr' else 'English'}
        and saved to {output_file_path}.
        """
    )


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
        line = re.sub(r"(\.){2}", "", line)
        # Put a space before punctuation
        line = re.sub(r"(['‘’:,!.?])", r" \1", line)
        # Remove double quotes
        line = re.sub(r"“", "", line)
        line = re.sub(r"”", "", line)
        line = re.sub(r'"', "", line)
        # If vowel on either side of the hyphen, insert an 'h'
        line = re.sub(r"([aioâêîô])\s*[-—]\s*([aioâêîô])", r"\1h\2", line)
        # Remove hyphen and replace with nothing because two parts should be together
        line = re.sub(r"\s*[-—]\s*", r"", line)
        # Remove numbers
        line = re.sub(r"[0-9]", "", line)
        # Remove extra spaces
        line = re.sub(r"\s+", r" ", line)
        # Remove anything other than essential characters
        line = re.sub(r"[^a-zA-Zâêîô.!?:,'‘’]+", r" ", line)
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
        line = re.sub(r"(\.){2,}", "", line)
        # Put spaces before punctuation
        line = re.sub(r"(['‘’:,!.?])", r" \1", line)
        # Remove double quotes
        line = re.sub(r"“", "", line)
        line = re.sub(r"”", "", line)
        line = re.sub(r'"', "", line)
        # Remove hyphen and replace with space because two words can standalone mostly in English
        line = re.sub(r"\s*[‐—]\s*", r" ", line)
        # Remove numbers
        line = re.sub(r"[0-9]", "", line)
        # Remove extra spaces
        line = re.sub(r"\s+", r" ", line)
        # Remove anything other than essential characters
        line = re.sub(r"[^a-zA-Z.!?:,'‘’]+", r" ", line)
    return lines


if __name__ == "__main__":
    # recursively iterate through all files in the current working directory
    # and run preprocess() on each file if it ends with _cr.txt or _en.txt
    for root, _, filenames in os.walk(os.getcwd()):
        for filename in filenames:
            if filename.endswith("_cr.txt") or filename.endswith("_en.txt"):
                in_file_path = os.path.join(root, filename)
                # set output_file_path to mirror input folder structure but in data/preprocessed/plains-cree
                out_file_path = in_file_path.replace(
                    "external/Plains-Cree-Corpora/PlainsCree",
                    "preprocessed/plains-cree",
                )
                # check if output_file_path directory exists, if not, create it
                if not os.path.exists(os.path.dirname(out_file_path)):
                    os.makedirs(os.path.dirname(out_file_path))

                # print(output_file_path)
                preprocess(in_file_path, out_file_path)
