# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:41:20 2024

@author: BishwajitPrasadGond
"""

import os
from pprint import pprint
from dotenv import load_dotenv
from genai.client import Client
from genai.credentials import Credentials
from genai.schema import (
    DecodingMethod,
    LengthPenalty,
    TextGenerationParameters,
    TextGenerationReturnOptions,
    ModerationParameters,
    ModerationSocialBias,
    ModerationSocialBiasInput,
    ModerationSocialBiasOutput
)

# Load environment variables
load_dotenv()

# Initialize the client with credentials
client = Client(credentials=Credentials.from_env())

def heading(text: str) -> str:
    """Helper function for centering text."""
    return "\n" + f" {text} ".center(80, "=") + "\n"

prompt_name = "My prompt"
model_id = "google/flan-t5-xl"

print(heading("Create prompt"))
template = "This is the recipe for {{meal}} as written by {{author}}: "
create_response = client.prompt.create(
    model_id=model_id,
    name=prompt_name,
    input=template,
    data={"meal": "goulash", "author": "Shakespeare"},
    parameters=TextGenerationParameters(
        length_penalty=LengthPenalty(decay_factor=1.5),
        decoding_method=DecodingMethod.SAMPLE,
        moderations=ModerationParameters(
            social_bias=ModerationSocialBias(
                input=ModerationSocialBiasInput(enabled=True, threshold=0.8),
                output=ModerationSocialBiasOutput(enabled=True, threshold=0.8)
            )
        )
    )
)
prompt_id = create_response.result.id
print(f"Prompt id: {prompt_id}")

print(heading("Get prompt details"))
retrieve_response = client.prompt.retrieve(id=prompt_id)
pprint(retrieve_response.result.model_dump())

print(heading("Generate text using prompt"))
for generation_response in client.text.generation.create(
    prompt_id=prompt_id,
    parameters=TextGenerationParameters(
        return_options=TextGenerationReturnOptions(input_text=True),
        moderations=ModerationParameters(
            social_bias=ModerationSocialBias(
                input=ModerationSocialBiasInput(enabled=True, threshold=0.8),
                output=ModerationSocialBiasOutput(enabled=True, threshold=0.8)
            )
        )
    ),
):
    result = generation_response.results[0]
    print(f"Prompt: {result.input_text}")
    print(f"Answer: {result.generated_text}")

print(heading("Override prompt template variables"))
for generation_response in client.text.generation.create(
    prompt_id=prompt_id,
    parameters=TextGenerationParameters(
        return_options=TextGenerationReturnOptions(input_text=True),
        moderations=ModerationParameters(
            social_bias=ModerationSocialBias(
                input=ModerationSocialBiasInput(enabled=True, threshold=0.8),
                output=ModerationSocialBiasOutput(enabled=True, threshold=0.8)
            )
        )
    ),
    data={"meal": "pancakes", "author": "Edgar Allan Poe"},
):
    result = generation_response.results[0]
    print(f"Prompt: {result.input_text}")
    print(f"Answer: {result.generated_text}")

print(heading("Show all existing prompts"))
prompt_list_response = client.prompt.list(search=prompt_name, limit=10, offset=0)
print("Total Count: ", prompt_list_response.total_count)
print("Results: ", prompt_list_response.results)

print(heading("Delete prompt"))
client.prompt.delete(id=prompt_id)
print("OK")
