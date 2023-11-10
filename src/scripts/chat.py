import re
import pandas as pd
from random import randint, uniform
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import ChatCompletion, OpenAIError

GPT_MODEL = "gpt-3.5-turbo"
SRC = "Inuktitut" #source language
TGT = "English" #taget language


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def chat_completion_request_API(
    messages,
    functions=None,
    function_call=None,
    temperature=0,
    max_tokens=None,
    stop=None,
    n=None,
    model=GPT_MODEL
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
        model (str, optional): A string representing the model to be used. Defaults to "GPT_MODEL".

    Returns:
        ChatCompletion: An instance of the ChatCompletion class.
    """
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
        json_data['n'] = n

    try:
        return ChatCompletion.create(**json_data)

    except OpenAIError as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

#TODO Fix AttributeError
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
    
    # Select a random subset of examples from the gold standard and PLL corpus
    gs_subset = gold_std.sample(n=n_shots, replace=False)
    pc_subset = pll_corpus.sample(n=n_samples, replace=False)
    
    cols = ['src_txt', 'tgt_txt', 'rom_txt', 'trans_txt']
    results = []
    
    # Empty string to store examples for few-shot learning
    examples = ''

    # Parse gold standard dataframe to get examples
    for idx, row in gs_subset.iterrows():
        src_example = row['src_lang']
        tgt_example = row['tgt_lang']

        examples += f'Text: {src_example} | Translation: {tgt_example} ###\n'
    
    # Iterate over the dataframe subset
    for idx, row in pc_subset.iterrows():
        src_txt = row["source_text"]
        tgt_txt = row['target_text']
        
        # Create the chat conversation messages
        message = [
            {"role": "system", "content": sys_msg},
            {"role": "user", 'content': examples},
            {'role': "user", 'content': f'Provide the {TGT} transliteration and translation for the following text: {src_txt} ###'},
        ]
        
        try:
            # Send the chat conversation messages to the API
            res = chat_completion_request_API(messages=message)

            # Extract the predicted text, romanization, and translation from the API response
            pred_txt = res["choices"][0]['message']['content']
            rom_txt = re.search(r'Romanization: (.+?)\n',pred_txt).group(1).strip('[]""')
            trans_txt = re.search(r'Translation: (.+?)$', pred_txt).group(1).strip('[]""')

            src_txt = src_txt.strip('[]')

            print("Romanized Text:", rom_txt)
            print("Translated Text:", trans_txt)
            print('Actual translation: ', tgt_txt)

            results.append({'src_txt': src_txt, 'tgt_txt': tgt_txt, 'rom_txt': rom_txt, 'trans_txt': trans_txt})

        except Exception as e:
            # Print exception message if an error occurs\
            print(f"Exception: {e}")
            try:
                # Send the chat conversation messages to the API
                res = chat_completion_request_API(messages=message)
            
                # Extract the predicted text, romanization, and translation from the API response
                pred_txt = res["choices"][0]['message']['content']
                rom_txt = re.search(r'Romanization: (.+?)\n',pred_txt).group(1).strip('[]""')
                trans_txt = re.search(r'Translation: (.+?)$', pred_txt).group(1).strip('[]""')
            
                src_txt = src_txt.strip('[]')
            
                print("Romanized Text:", rom_txt)
                print("Translated Text:", trans_txt)
                print('Actual translation: ', tgt_txt)
            
                results.append({'src_txt': src_txt, 'tgt_txt': tgt_txt, 'rom_txt': rom_txt, 'trans_txt': trans_txt})
            
            except Exception as e:
                # Print exception message if an error occurs
                print(f"Exception: {e}")
                
                # Retry the API call until it succeeds or a maximum number of attempts is reached
                max_attempts = 5
                attempts = 0
            
                while attempts < max_attempts:
                    try:
                        res = chat_completion_request_API(messages=message)
                        pred_txt = res["choices"][0]['message']['content']
                        rom_txt = re.search(r'Romanization: (.+?)\n',pred_txt).group(1).strip('[]""')
                        trans_txt = re.search(r'Translation: (.+?)$', pred_txt).group(1).strip('[]""')
            
                        src_txt = src_txt.strip('[]')
            
                        print("Romanized Text:", rom_txt)
                        print("Translated Text:", trans_txt)
                        print('Actual translation: ', tgt_txt)
            
                        results.append({'src_txt': src_txt, 'tgt_txt': tgt_txt, 'rom_txt': rom_txt, 'trans_txt': trans_txt})
            
                        break  # Exit the loop if the API call is successful
            
                    except Exception as e:
                        print(f"Exception: {e}")
                        attempts += 1
            
                if attempts == max_attempts:
                    print("Max number of attempts reached. Skipping this example.")
                    
    return pd.DataFrame(results, columns=cols)
