import os
from openai import AzureOpenAI

endpoint = "https://fsp-azureai-2.openai.azure.com/"
model_name = "gpt-4.1-nano"
deployment = "gpt-4.1-nano"

subscription_key = "f66e4115cdf74de09838d56daca44711"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {
            "role": "system",
            "content": f"Translate the following text to English. Only return the translation, no additional text.",
        },
        {"role": "user", "content": "احا"},
    ],
)

print(response.choices[0].message.content)
