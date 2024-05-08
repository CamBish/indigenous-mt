import os

import openai
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_random_exponential


# TODO label inputs and outputs
def ollama_chat_completion(
    messages,
    model,
    temperature=0,  # temperature of 0 was best as shown in Jiao et al., 2023
):
    """Make a chat request to Ollama models"""
    api_base = "http://localhost:11434"
    json_data = {
        "model": model,
        "messages": messages,
        "api_base": api_base,
        "temperature": temperature,
    }

    try:
        return completion(**json_data)
    except openai.APIConnectionError as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def openai_chat_completion(
    messages,
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
