from openai import OpenAI
import os

client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://eu.api.openai.com/v1",
    )


response = client.responses.create(
    model="gpt-5.4-mini",
    input="Explain what an API is in one simple sentence."
)

print(response.output_text)