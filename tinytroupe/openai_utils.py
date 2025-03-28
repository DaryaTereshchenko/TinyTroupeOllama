import os
import openai
from openai import OpenAI, AzureOpenAI
import time
import json
import pickle
import logging
import configparser
import tiktoken
from tinytroupe import utils
from tinytroupe.utils import compose_prompt_for_api # Added import to allow for Ollama usage
import requests

logger = logging.getLogger("tinytroupe")

# We'll use various configuration elements below
config = utils.read_config_file()

###########################################################################
# Default parameter values
###########################################################################
default = {}
default["model"] = config["OpenAI"].get("MODEL", "gpt-4")
default["max_tokens"] = int(config["OpenAI"].get("MAX_TOKENS", "1024"))
default["temperature"] = float(config["OpenAI"].get("TEMPERATURE", "0.3"))
default["top_p"] = int(config["OpenAI"].get("TOP_P", "0"))
default["frequency_penalty"] = float(config["OpenAI"].get("FREQ_PENALTY", "0.0"))
default["presence_penalty"] = float(
    config["OpenAI"].get("PRESENCE_PENALTY", "0.0"))
default["timeout"] = float(config["OpenAI"].get("TIMEOUT", "30.0"))
default["max_attempts"] = float(config["OpenAI"].get("MAX_ATTEMPTS", "0.0"))
default["waiting_time"] = float(config["OpenAI"].get("WAITING_TIME", "0.5"))
default["exponential_backoff_factor"] = float(config["OpenAI"].get("EXPONENTIAL_BACKOFF_FACTOR", "5"))

default["embedding_model"] = config["OpenAI"].get("EMBEDDING_MODEL", "text-embedding-3-small")

default["cache_api_calls"] = config["OpenAI"].getboolean("CACHE_API_CALLS", False)
default["cache_file_name"] = config["OpenAI"].get("CACHE_FILE_NAME", "openai_api_cache.pickle")

###########################################################################
# Model calling helpers
###########################################################################

# TODO under development
class LLMCall:
    """
    A class that represents an LLM model call. It contains the input messages or prompt, 
    the model configuration, and the model output.
    """

    def __init__(self, system_template_name: str, user_template_name: str = None, **model_params):
        """
        Initializes an LLMCall instance with the specified system and user templates.
        
        Args:
            system_template_name (str): The system-level template file name.
            user_template_name (str, optional): The user-level template file name.
            model_params (dict): Additional parameters for the LLM model.
        """
        self.system_template_name = system_template_name
        self.user_template_name = user_template_name
        self.model_params = model_params
        self.messages_or_prompt = None
        self.model_output = None

    def call(self, **rendering_configs):
        """
        Calls the LLM model with the specified rendering configurations.
        
        Args:
            rendering_configs (dict): The configurations for rendering the template.
            
        Returns:
            str: The model's output content if successful, or None otherwise.
        """
        # Use the wrapper function to dynamically generate the messages or prompt
        self.messages_or_prompt = compose_prompt_for_api(
            self.system_template_name, 
            self.user_template_name, 
            rendering_configs
        )

        # Call the LLM model
        self.model_output = client().send_message(self.messages_or_prompt, **self.model_params)

        if isinstance(self.model_output, dict) and 'content' in self.model_output:
            return self.model_output['content']
        elif isinstance(self.model_output, str):
            return self.model_output  # For Ollama, the output is directly a string
        else:
            logger.error(f"Model output does not contain 'content' key: {self.model_output}")
            return None

    def __repr__(self):
        return (
            f"LLMCall(messages_or_prompt={self.messages_or_prompt}, "
            f"model_params={self.model_params}, "
            f"model_output={self.model_output})"
        )



###########################################################################
# Client class
###########################################################################

class OpenAIClient:
    """
    A utility class for interacting with the OpenAI API.
    """

    def __init__(self, cache_api_calls=default["cache_api_calls"], cache_file_name=default["cache_file_name"]) -> None:
        logger.debug("Initializing OpenAIClient")

        # should we cache api calls and reuse them?
        self.set_api_cache(cache_api_calls, cache_file_name)
    
    def set_api_cache(self, cache_api_calls, cache_file_name=default["cache_file_name"]):
        """
        Enables or disables the caching of API calls.

        Args:
        cache_file_name (str): The name of the file to use for caching API calls.
        """
        self.cache_api_calls = cache_api_calls
        self.cache_file_name = cache_file_name
        if self.cache_api_calls:
            # load the cache, if any
            self.api_cache = self._load_cache()
    
    
    def _setup_from_config(self):
        """
        Sets up the OpenAI API configurations for this client.
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def send_message(self,
                    current_messages,
                     model=default["model"],
                     temperature=default["temperature"],
                     max_tokens=default["max_tokens"],
                     top_p=default["top_p"],
                     frequency_penalty=default["frequency_penalty"],
                     presence_penalty=default["presence_penalty"],
                     stop=[],
                     timeout=default["timeout"],
                     max_attempts=default["max_attempts"],
                     waiting_time=default["waiting_time"],
                     exponential_backoff_factor=default["exponential_backoff_factor"],
                     n = 1,
                     echo=False):
        """
        Sends a message to the OpenAI API and returns the response.

        Args:
        current_messages (list): A list of dictionaries representing the conversation history.
        model (str): The ID of the model to use for generating the response.
        temperature (float): Controls the "creativity" of the response. Higher values result in more diverse responses.
        max_tokens (int): The maximum number of tokens (words or punctuation marks) to generate in the response.
        top_p (float): Controls the "quality" of the response. Higher values result in more coherent responses.
        frequency_penalty (float): Controls the "repetition" of the response. Higher values result in less repetition.
        presence_penalty (float): Controls the "diversity" of the response. Higher values result in more diverse responses.
        stop (str): A string that, if encountered in the generated response, will cause the generation to stop.
        max_attempts (int): The maximum number of attempts to make before giving up on generating a response.
        timeout (int): The maximum number of seconds to wait for a response from the API.

        Returns:
        A dictionary representing the generated response.
        """

        def aux_exponential_backoff():
            nonlocal waiting_time
            logger.info(f"Request failed. Waiting {waiting_time} seconds between requests...")
            time.sleep(waiting_time)

            # exponential backoff
            waiting_time = waiting_time * exponential_backoff_factor
        

        # setup the OpenAI configurations for this client.
        self._setup_from_config()
        
        # We need to adapt the parameters to the API type, so we create a dictionary with them first
        chat_api_params = {
            "messages": current_messages,
            "temperature": temperature,
            "max_tokens":max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop,
            "timeout": timeout,
            "stream": False,
            "n": n,
        }


        i = 0
        while i < max_attempts:
            try:
                i += 1

                try:
                    logger.debug(f"Sending messages to OpenAI API. Token count={self._count_tokens(current_messages, model)}.")
                except NotImplementedError:
                    logger.debug(f"Token count not implemented for model {model}.")
                    
                start_time = time.monotonic()
                logger.debug(f"Calling model with client class {self.__class__.__name__}.")

                ###############################################################
                # call the model, either from the cache or from the API
                ###############################################################
                cache_key = str((model, chat_api_params)) # need string to be hashable
                if self.cache_api_calls and (cache_key in self.api_cache):
                    response = self.api_cache[cache_key]
                else:
                    logger.info(f"Waiting {waiting_time} seconds before next API request (to avoid throttling)...")
                    time.sleep(waiting_time)
                    
                    response = self._raw_model_call(model, chat_api_params)
                    if self.cache_api_calls:
                        self.api_cache[cache_key] = response
                        self._save_cache()
                
                
                logger.debug(f"Got response from API: {response}")
                end_time = time.monotonic()
                logger.debug(
                    f"Got response in {end_time - start_time:.2f} seconds after {i + 1} attempts.")

                return utils.sanitize_dict(self._raw_model_response_extractor(response))

            except InvalidRequestError as e:
                logger.error(f"[{i}] Invalid request error, won't retry: {e}")

                # there's no point in retrying if the request is invalid
                # so we return None right away
                return None
            
            except openai.BadRequestError as e:
                logger.error(f"[{i}] Invalid request error, won't retry: {e}")
                
                # there's no point in retrying if the request is invalid
                # so we return None right away
                return None
            
            except openai.RateLimitError:
                logger.warning(
                    f"[{i}] Rate limit error, waiting a bit and trying again.")
                aux_exponential_backoff()
            
            except NonTerminalError as e:
                logger.error(f"[{i}] Non-terminal error: {e}")
                aux_exponential_backoff()
                
            except Exception as e:
                logger.error(f"[{i}] Error: {e}")

        logger.error(f"Failed to get response after {max_attempts} attempts.")
        return None
    
    def _raw_model_call(self, model, chat_api_params):
        """
        Calls the OpenAI API with the given parameters. Subclasses should
        override this method to implement their own API calls.
        """
        
        chat_api_params["model"] = model # OpenAI API uses this parameter name
        return self.client.chat.completions.create(
                    **chat_api_params
                )

    def _raw_model_response_extractor(self, response):
        """
        Extracts the response from the API response. Subclasses should
        override this method to implement their own response extraction.
        """
        return response.choices[0].message.to_dict()

    def _count_tokens(self, messages: list, model: str):
        """
        Count the number of OpenAI tokens in a list of messages using tiktoken.

        Adapted from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

        Args:
        messages (list): A list of dictionaries representing the conversation history.
        model (str): The name of the model to use for encoding the string.
        """
        try:
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                logger.debug("Token count: model not found. Using cl100k_base encoding.")
                encoding = tiktoken.get_encoding("cl100k_base")
            if model in {
                "gpt-3.5-turbo-0613",
                "gpt-3.5-turbo-16k-0613",
                "gpt-4-0314",
                "gpt-4-32k-0314",
                "gpt-4-0613",
                "gpt-4-32k-0613",
                }:
                tokens_per_message = 3
                tokens_per_name = 1
            elif model == "gpt-3.5-turbo-0301":
                tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
                tokens_per_name = -1  # if there's a name, the role is omitted
            elif "gpt-3.5-turbo" in model:
                logger.debug("Token count: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
                return self._count_tokens(messages, model="gpt-3.5-turbo-0613")
            elif ("gpt-4" in model) or ("ppo" in model):
                logger.debug("Token count: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
                return self._count_tokens(messages, model="gpt-4-0613")
            else:
                raise NotImplementedError(
                    f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
                )
            num_tokens = 0
            for message in messages:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
            num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
            return num_tokens
        
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            return None

    def _save_cache(self):
        """
        Saves the API cache to disk. We use pickle to do that because some obj
        are not JSON serializable.
        """
        # use pickle to save the cache
        pickle.dump(self.api_cache, open(self.cache_file_name, "wb"))

    
    def _load_cache(self):

        """
        Loads the API cache from disk.
        """
        # unpickle
        return pickle.load(open(self.cache_file_name, "rb")) if os.path.exists(self.cache_file_name) else {}

    def get_embedding(self, text, model=default["embedding_model"]):
        """
        Gets the embedding of the given text using the specified model.

        Args:
        text (str): The text to embed.
        model (str): The name of the model to use for embedding the text.

        Returns:
        The embedding of the text.
        """
        response = self._raw_embedding_model_call(text, model)
        return self._raw_embedding_model_response_extractor(response)
    
    def _raw_embedding_model_call(self, text, model):
        """
        Calls the OpenAI API to get the embedding of the given text. Subclasses should
        override this method to implement their own API calls.
        """
        return self.client.embeddings.create(
            input=[text],
            model=model
        )
    
    def _raw_embedding_model_response_extractor(self, response):
        """
        Extracts the embedding from the API response. Subclasses should
        override this method to implement their own response extraction.
        """
        return response.data[0].embedding

class AzureClient(OpenAIClient):

    def __init__(self, cache_api_calls=default["cache_api_calls"], cache_file_name=default["cache_file_name"]) -> None:
        logger.debug("Initializing AzureClient")

        super().__init__(cache_api_calls, cache_file_name)
    
    def _setup_from_config(self):
        """
        Sets up the Azure OpenAI Service API configurations for this client,
        including the API endpoint and key.
        """
        self.client = AzureOpenAI(azure_endpoint= os.getenv("AZURE_OPENAI_ENDPOINT"),
                                  api_version = config["OpenAI"]["AZURE_API_VERSION"],
                                  api_key = os.getenv("AZURE_OPENAI_KEY"))
    
    def _raw_model_call(self, model, chat_api_params):
        """
        Calls the Azue OpenAI Service API with the given parameters.
        """
        chat_api_params["model"] = model 

        return self.client.chat.completions.create(
                    **chat_api_params
                )


class InvalidRequestError(Exception):
    """
    Exception raised when the request to the OpenAI API is invalid.
    """
    pass

class NonTerminalError(Exception):
    """
    Exception raised when an unspecified error occurs but we know we can retry.
    """
    pass


###########################################################################
# Ollama Client Class
###########################################################################

# class OllamaClient:
#     def __init__(self, base_url, model, temperature=0.7, top_p=0.95, timeout=60):
#         self.base_url = base_url
#         self.model = model
#         self.temperature = temperature
#         self.top_p = top_p
#         self.timeout = timeout


#     def send_message(self, messages):
#         """
#         Sends a message to the Ollama API and returns the full JSON response.

#         Args:
#             messages (list): A list of dictionaries representing the conversation.

#         Returns:
#             dict: The full JSON response from the API.
#         """
#         payload = {
#             "model": self.model,
#             "messages": messages,
#             "stream": False  # Ensure full response in one go
#         }
#         try:
#             response = requests.post(
#                 self.base_url,
#                 json=payload,
#                 timeout=self.timeout
#             )
#             response.raise_for_status()
            
#             # Parse response JSON
#             response_json = response.json()
#             logger.info(f"Ollama API Response: {response_json}")

#             # Return the full response JSON
#             return response_json
            
#         except requests.exceptions.RequestException as e:
#             logger.error(f"Error communicating with Ollama API: {e}")
#             return {"error": str(e)}
#         except ValueError as e:  # Handle JSON decoding errors
#             logger.error(f"Failed to parse API response: {e}")
#             return {"error": "Invalid JSON response"}

# class OllamaClient:
#     def __init__(self, base_url, model, temperature=0.3, top_p=0.95, timeout=60):
#         self.base_url = base_url
#         self.model = model
#         self.temperature = temperature
#         self.top_p = top_p
#         self.timeout = timeout

#     def send_message(self, messages):
#         """
#         Sends a message to the Ollama API and returns the full JSON response.
#         """
#         payload = {
#             "model": self.model,
#             "messages": messages,
#             "stream": False
#         }

#         try:
#             response = requests.post(
#                 self.base_url,
#                 json=payload,
#                 timeout=self.timeout
#             )
#             response.raise_for_status()
            
#             # Get the raw response
#             response_json = response.json()
#             logger.debug(f"Ollama API Response: {response_json}")

#             # If we have a message with content, try to use it directly
#             if 'message' in response_json and 'content' in response_json['message']:
#                 content = response_json['message']['content']
                
#                 # Return the raw response - let the agent handle parsing
#                 return response_json
            
#             logger.error(f"Unexpected response format: {response_json}")
#             return response_json

#         except requests.exceptions.RequestException as e:
#             logger.error(f"Error communicating with Ollama API: {e}")
#             return {"error": str(e)}
            
#         except ValueError as e:
#             logger.error(f"Failed to parse API response: {e}")
#             return {"error": "Invalid JSON response"}

# class OllamaClient:
#     def __init__(self, base_url, model, temperature=0.3, top_p=0.95, timeout=60):
#         self.base_url = base_url
#         self.model = model
#         self.temperature = temperature
#         self.top_p = top_p
#         self.timeout = timeout

#     def send_message(self, messages):
#         """
#         Sends a message to the Ollama API and returns the full JSON response.
#         """
#         payload = {
#             "model": self.model,
#             "messages": messages,
#             "stream": False
#         }
        
#         try:
#             response = requests.post(
#                 self.base_url,
#                 json=payload,
#                 timeout=self.timeout
#             )
#             response.raise_for_status()
            
#             # Get the raw response
#             response_json = response.json()
#             logger.debug(f"Ollama API Response: {response_json}")

#             # If we have a message with content
#             if 'message' in response_json and 'content' in response_json['message']:
#                 content = response_json['message']['content']
                
#                 # Try to parse content as JSON
#                 try:
#                     parsed_content = json.loads(content)
#                     if 'action' in parsed_content and 'cognitive_state' not in parsed_content:
#                         # Add cognitive_state if missing but action exists
#                         parsed_content['cognitive_state'] = {
#                             'goals': 'Continue current interaction',
#                             'attention': 'Current conversation',
#                             'emotions': 'Engaged'
#                         }
#                         return {'message': {'content': json.dumps(parsed_content)}}
#                 except:
#                     pass  # If parsing fails, return original response
                
#                 # Return the raw response
#                 return response_json

#             logger.error(f"Unexpected response format: {response_json}")
#             return response_json

#         except requests.exceptions.RequestException as e:
#             logger.error(f"Error communicating with Ollama API: {e}")
#             return {"error": str(e)}
            
#         except ValueError as e:
#             logger.error(f"Failed to parse API response: {e}")
#             return {"error": "Invalid JSON response"}
class OllamaClient:
    def __init__(self, base_url, model=None, temperature=0.7, top_p=0.95, timeout=60):
        self.base_url = base_url.rstrip('/')  # Remove trailing slash if present
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        
        # Get config to find the correct endpoint
        from tinytroupe import utils
        config = utils.read_config_file()
        self.endpoint = config["Ollama"].get("ENDPOINT", "/api/chat").lstrip('/')
        if not self.model:
            self.model = config["Ollama"].get("MODEL", "llama3.1")

    def send_message(self, messages, response_format=None, max_tokens=None, temperature=None, top_p=None):
        """
        Sends messages to Ollama and processes the response.
        """
        # Construct the full URL correctly
        url = f"{self.base_url}/{self.endpoint}"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "top_p": top_p or self.top_p,
            "stream": False
        }
        
        logger.info(f"Sending request to Ollama at: {url}")
        logger.debug(f"Payload: {payload}")

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            response_json = response.json()
            logger.debug(f"Ollama response: {response_json}")
            
            if response_format and hasattr(response_format, '__name__') and response_format.__name__ == "CognitiveActionModel":
                # Create properly formatted response for TinyTroupe's cognitive action model
                content = response_json.get("message", {}).get("content", "")
                
                # Format response with required cognitive state
                cognitive_response = {
                    "cognitive_state": {
                        "attention": "focused",
                        "emotion": "neutral",
                        "thoughts": "Processing the conversation",
                        "goals": ["Respond appropriately to the situation"]
                    },
                    "actions": [
                        {
                            "action_type": "talk",
                            "target": None,
                            "content": content
                        }
                    ]
                }
                
                return {
                    "role": "assistant",
                    "content": json.dumps(cognitive_response)
                }
            
            # For standard responses
            return {
                "role": "assistant",
                "content": response_json.get("message", {}).get("content", "")
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with Ollama API: {e}")
            
            # Return a properly formatted error response with cognitive state
            error_response = {
                "cognitive_state": {
                    "attention": "error",
                    "emotion": "concerned",
                    "thoughts": f"Technical difficulties: {str(e)}",
                    "goals": ["Resolve connection issues"]
                },
                "actions": [
                    {
                        "action_type": "talk",
                        "target": None,
                        "content": "I'm having trouble processing your request due to technical issues."
                    }
                ]
            }
            
            return {
                "role": "assistant",
                "content": json.dumps(error_response)
            }

###########################################################################
# Clients registry
#
# We can have potentially different clients, so we need a place to 
# register them and retrieve them when needed.
#
# We support both OpenAI and Azure OpenAI Service API by default.
# Thus, we need to set the API parameters based on the choice of the user.
# This is done within specialized classes.
#
# It is also possible to register custom clients, to access internal or
# otherwise non-conventional API endpoints.
###########################################################################
_api_type_to_client = {}
_api_type_override = None

def register_client(api_type, client):
    """
    Registers a client for the given API type.

    Args:
    api_type (str): The API type for which we want to register the client.
    client: The client to register.
    """
    _api_type_to_client[api_type] = client

def _get_client_for_api_type(api_type):
    """
    Returns the client for the given API type.

    Args:
    api_type (str): The API type for which we want to get the client.
    """
    try:
        return _api_type_to_client[api_type]
    except KeyError:
        raise ValueError(f"API type {api_type} is not supported. Please check the 'config.ini' file.")

def client():
    """
    Returns the client for the configured API type.
    """
    api_type = config["OpenAI"]["API_TYPE"] if _api_type_override is None else _api_type_override
    
    logger.debug(f"Using  API type {api_type}.")
    return _get_client_for_api_type(api_type)


# TODO simplify the custom configuration methods below

def force_api_type(api_type):
    """
    Forces the use of the given API type, thus overriding any other configuration.

    Args:
    api_type (str): The API type to use.
    """
    global _api_type_override
    _api_type_override = api_type

def force_api_cache(cache_api_calls, cache_file_name=default["cache_file_name"]):
    """
    Forces the use of the given API cache configuration, thus overriding any other configuration.

    Args:
    cache_api_calls (bool): Whether to cache API calls.
    cache_file_name (str): The name of the file to use for caching API calls.
    """
    # set the cache parameters on all clients
    for client in _api_type_to_client.values():
        client.set_api_cache(cache_api_calls, cache_file_name)

def force_default_value(key, value):
    """
    Forces the use of the given default configuration value for the specified key, thus overriding any other configuration.

    Args:
    key (str): The key to override.
    value: The value to use for the key.
    """
    global default

    # check if the key actually exists
    if key in default:
        default[key] = value
    else:
        raise ValueError(f"Key {key} is not a valid configuration key.")

# default client
register_client("openai", OpenAIClient())
register_client("azure", AzureClient())
# Registering the Ollama client
register_client("ollama", OllamaClient(
    base_url=config["Ollama"].get("BASE_URL"),
    model=config["Ollama"].get("MODEL"),
    temperature=float(config["Ollama"].get("TEMPERATURE", 0.7)),
    top_p=float(config["Ollama"].get("TOP_P", 0.95)),
    timeout=int(config["Ollama"].get("TIMEOUT", 60))
))



