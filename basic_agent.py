# import json
import re
import typing
import yaml
import os
# import re

from tenacity import retry, wait_random_exponential, stop_after_attempt, wait_fixed
from llm_clients import create_llm_client


def translate_messages_from_openai_to_gemini(
    messages_to_change: list[dict[str, str]],
) -> str:
    last_message = messages_to_change[-1]["content"]
    if len(messages_to_change) == 1:
        gemini_messages = []
    else:
        prev_messages = messages_to_change[:-1]
        gemini_messages = []
        for message in prev_messages:
            role = message["role"]
            content = message["content"]
            if role == "assistant":
                role = "model"

            gemini_messages.append({"role": role, "parts": [content]})

    return gemini_messages, last_message


class BasicAgent:
    def __init__(self):
        """Initializes the BasicAgent, loading configuration. brraaa"""
        config_path = "llm_config.yaml"
        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            self.llm_model_dict = config.get("llm_location", {})
            if not self.llm_model_dict:
                print(f"Warning: 'llm_location' not found or empty in {config_path}")
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {config_path}")
            self.llm_model_dict = {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML configuration file {config_path}: {e}")
            self.llm_model_dict = {}

        # Client state - initialized on first use or when model changes
        self.llm_client = None
        self.model_location = None
        self.llm_model_name = None
        self._current_llm_input = (
            None  # Tracks the input string used to create the current client
        )

    def get_text_response_from_llm(
        self,
        llm_model_input: str,
        messages: typing.Union[str, list[dict[str, str]]],
        code_tag: str = None,
        fallback_llm_model_input: str = 'priv_openai:gpt-4.1',  
    ) -> dict:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Try primary LLM
        try:
            return self._llm_response_inner(llm_model_input, messages, code_tag)
        except Exception as gemini_error:
            print(f"Primary LLM ({llm_model_input}) failed: {gemini_error}")
            if not fallback_llm_model_input:
                return {"text_response": None, "reasoning_content": None}

        # Try fallback LLM if provided
        try:
            print(f"Falling back to: {fallback_llm_model_input}")
            return self._llm_response_inner(fallback_llm_model_input, messages, code_tag)
        except Exception as fallback_error:
            print(f"Fallback LLM ({fallback_llm_model_input}) also failed: {fallback_error}")
            return {"text_response": None, "reasoning_content": None}
    

    @retry(
        wait=wait_fixed(2) + wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
    )
    def _get_text_response_from_llm(
        self,
        llm_model_input: str,  # Renamed parameter for clarity
        messages: typing.Union[str, list[dict[str, str]]],
        code_tag: str = None,
    ) -> dict:
        """
        Gets a text response from the specified LLM, handling client initialization.

        Args:
            llm_model_input: The model identifier string (e.g., "azure_openai:gpt-4", "gpt-4").
            messages: A single prompt string or a list of message dictionaries (OpenAI format).
            code_tag: Optional tag to extract code blocks from the response.

        Returns:
            A dictionary containing the 'text_response' and potentially 'reasoning_content'.
            Returns {'text_response': None, 'reasoning_content': None} on failure.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Check if the client needs to be (re)initialized
        if self.llm_client is None or llm_model_input != self._current_llm_input:
            print(f"Initializing LLM client for: {llm_model_input}")
            client, location, resolved_name = create_llm_client(
                llm_model_input, self.llm_model_dict
            )
            if client is None:
                print(
                    f"Failed to create LLM client for {llm_model_input}. Aborting request."
                )
                # Consider raising an exception here instead of returning None dict
                return {"text_response": None, "reasoning_content": None}

            self.llm_client = client
            self.model_location = location
            self.llm_model_name = resolved_name
            self._current_llm_input = (
                llm_model_input  # Store the input that created this client
            )
        else:
            print(f"Using existing LLM client for: {self._current_llm_input}")

        # Use the instance attributes for the API call
        llm_client = self.llm_client
        model_location = self.model_location
        llm_model_name_resolved = self.llm_model_name

        reasoning_content = None
        text_content = None

        try:  # Added try-except block for API calls
            if model_location in [
                "azure_openai",
                "dbrx",
                "groq",
                "openrouter",
                "priv_openai",
                "deepseek",
            ]:
                my_response = llm_client.chat.completions.create(
                    model=llm_model_name_resolved,
                    messages=messages,
                )
                text_content = my_response.choices[0].message.content

                # Check if reasoning_content exists (might vary by provider/model)
                if hasattr(my_response.choices[0].message, "reasoning_content"):
                    reasoning_content = my_response.choices[0].message.reasoning_content

            elif model_location == "google_ai_studio":  # Use == for clarity
                gemini_messages, last_message = (
                    translate_messages_from_openai_to_gemini(messages)
                )
                # Ensure llm_client is the GenerativeModel instance
                if hasattr(llm_client, "start_chat"):
                    chat_session = llm_client.start_chat(history=gemini_messages)
                    response = chat_session.send_message(last_message)
                    text_content = response.text

                else:
                    print("Error: Gemini client does not have 'start_chat' method.")
                    # Handle error appropriately
                    text_content = None  # Or raise exception

            else:
                # Should not happen if create_llm_client worked, but good practice
                print(
                    f"Error: Unsupported model location '{model_location}' encountered in get_text_response."
                )
                return {"text_response": None, "reasoning_content": None}

        except Exception as e:
            # Catch potential API errors during the call
            print(
                f"Error during LLM API call for {model_location} ({llm_model_name_resolved}): {e}"
            )
            return {"text_response": None, "reasoning_content": None}



        # Process response and extract code if needed
        if text_content is not None and code_tag is not None:
            tool_escaping_pattern = rf"```\s?{code_tag}\s?(.*?)```"
            match = re.search(tool_escaping_pattern, text_content, re.DOTALL)
            if match:
                extracted_content = match.group(1).strip()
                # Return only the extracted part if found
                return {
                    "text_response": extracted_content,
                    "reasoning_content": None,
                }  # Reasoning likely not applicable here
            else:
                # If tag specified but not found, return original text? Or indicate failure?
                # Current behavior: returns original text in the standard dict below
                print(
                    f"Warning: Code tag '{code_tag}' specified but not found in the response."
                )

        # Return the full response if no code tag or tag not found
        return {
            "text_response": text_content,
            "reasoning_content": reasoning_content,
        }
