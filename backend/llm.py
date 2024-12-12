from enum import Enum
from typing import Any, Awaitable, Callable, List, cast
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionChunk
from config import IS_DEBUG_ENABLED
from debug.DebugFileWriter import DebugFileWriter
import google.generativeai as genai
import base64
from io import BytesIO
from PIL import Image

from utils import pprint_prompt


# Actual model versions that are passed to the LLMs and stored in our logs
class Llm(Enum):
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_VISION = "gpt-4-vision-preview"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4O = "gpt-4o"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    CLAUDE_3_OPUS = "claude-3-opus"
    CLAUDE_3_HAIKU = "claude-3-haiku"
    GEMINI_1_5_FLASH_8_b = "gemini-1.5-flash-8b"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_2_0_FLASH= "gemini-2.0-flash-exp"

# Will throw errors if you send a garbage string
def convert_frontend_str_to_llm(frontend_str: str) -> Llm:
    if frontend_str == "gpt_4_vision":
        return Llm.GPT_4_VISION
    elif frontend_str == "claude_3_sonnet":
        return Llm.CLAUDE_3_SONNET
    else:
        return Llm(frontend_str)


async def stream_openai_response(
    messages: List[ChatCompletionMessageParam],
    api_key: str,
    base_url: str | None,
    callback: Callable[[str], Awaitable[None]],
    model: Llm,
) -> str:
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # Base parameters
    params = {
        "model": model.value,
        "messages": messages,
        "stream": True,
        "timeout": 600,
        "temperature": 0.0,
    }

    # Add 'max_tokens' only if the model is a GPT4 vision or Turbo model
    if (
        model == Llm.GPT_4_VISION
        or model == Llm.GPT_4_TURBO
        or model == Llm.GPT_4O
    ):
        params["max_tokens"] = 4096

    stream = await client.chat.completions.create(**params)  # type: ignore
    full_response = ""
    async for chunk in stream:  # type: ignore
        assert isinstance(chunk, ChatCompletionChunk)
        if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content or ""
            full_response += content
            await callback(content)

    await client.close()

    return full_response


# TODO: Have a seperate function that translates OpenAI messages to Claude messages
async def stream_claude_response(
    messages: List[ChatCompletionMessageParam],
    api_key: str,
    callback: Callable[[str], Awaitable[None]],
) -> str:

    client = AsyncAnthropic(api_key=api_key)

    # Base parameters
    model = Llm.CLAUDE_3_SONNET
    max_tokens = 4096
    temperature = 0.0

    # Translate OpenAI messages to Claude messages
    system_prompt = cast(str, messages[0].get("content"))
    claude_messages = [dict(message) for message in messages[1:]]
    for message in claude_messages:
        if not isinstance(message["content"], list):
            continue

        for content in message["content"]:  # type: ignore
            if content["type"] == "image_url":
                content["type"] = "image"

                # Extract base64 data and media type from data URL
                # Example base64 data URL: data:image/png;base64,iVBOR...
                image_data_url = cast(str, content["image_url"]["url"])
                media_type = image_data_url.split(";")[0].split(":")[1]
                base64_data = image_data_url.split(",")[1]

                # Remove OpenAI parameter
                del content["image_url"]

                content["source"] = {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                }

    # Stream Claude response
    async with client.messages.stream(
        model=model.value,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=claude_messages,  # type: ignore
    ) as stream:
        async for text in stream.text_stream:
            await callback(text)

    # Return final message
    response = await stream.get_final_message()

    # Close the Anthropic client
    await client.close()

    return response.content[0].text


async def stream_claude_response_native(
    system_prompt: str,
    messages: list[Any],
    api_key: str,
    callback: Callable[[str], Awaitable[None]],
    include_thinking: bool = False,
    model: Llm = Llm.CLAUDE_3_OPUS,
) -> str:

    client = AsyncAnthropic(api_key=api_key)

    # Base model parameters
    max_tokens = 4096
    temperature = 0.0

    # Multi-pass flow
    current_pass_num = 1
    max_passes = 2

    prefix = "<thinking>"
    response = None

    # For debugging
    full_stream = ""
    debug_file_writer = DebugFileWriter()

    while current_pass_num <= max_passes:
        current_pass_num += 1

        # Set up message depending on whether we have a <thinking> prefix
        messages_to_send = (
            messages + [{"role": "assistant", "content": prefix}]
            if include_thinking
            else messages
        )

        pprint_prompt(messages_to_send)

        async with client.messages.stream(
            model=model.value,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages_to_send,  # type: ignore
        ) as stream:
            async for text in stream.text_stream:
                print(text, end="", flush=True)
                full_stream += text
                await callback(text)

        response = await stream.get_final_message()
        response_text = response.content[0].text

        # Write each pass's code to .html file and thinking to .txt file
        if IS_DEBUG_ENABLED:
            debug_file_writer.write_to_file(
                f"pass_{current_pass_num - 1}.html",
                debug_file_writer.extract_html_content(response_text),
            )
            debug_file_writer.write_to_file(
                f"thinking_pass_{current_pass_num - 1}.txt",
                response_text.split("</thinking>")[0],
            )

        # Set up messages array for next pass
        messages += [
            {"role": "assistant", "content": str(prefix) + response.content[0].text},
            {
                "role": "user",
                "content": "You've done a good job with a first draft. Improve this further based on the original instructions so that the app is fully functional and looks like the original video of the app we're trying to replicate.",
            },
        ]

        print(
            f"Token usage: Input Tokens: {response.usage.input_tokens}, Output Tokens: {response.usage.output_tokens}"
        )

    # Close the Anthropic client
    await client.close()

    if IS_DEBUG_ENABLED:
        debug_file_writer.write_to_file("full_stream.txt", full_stream)

    if not response:
        raise Exception("No HTML response found in AI response")
    else:
        return response.content[0].text


def encode_image_from_base64(image_base64):
    # Check if the input is a data URI
    if image_base64.startswith("data:image/"):
        # Extract the image data from the data URI
        image_data = image_base64.split(",")[1]
        # Decode the base64-encoded image data
        image_bytes = base64.b64decode(image_data)
    else:
        # Assume the input is already base64-encoded
        image_bytes = base64.b64decode(image_base64)

    # Convert the image bytes into a NumPy array
    import numpy as np
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

    # Return the encoded image array
    return image_array

async def stream_gemini_response(
    messages: List[ChatCompletionMessageParam],
    api_key: str,
    callback: Callable[[str], Awaitable[None]],
    model: Llm,
) -> str:
    print(f"model.value ==: { model.value }")
    # genai.configure(api_key=api_key)
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
        # "api_key":api_key,
    }
    # model = genai.GenerativeModel("gemini-1.5-flash")

    # Translate OpenAI messages to Gemini messages
    # gemini_messages = []
    # print(f"system ==: {str(messages)}")
    for message in messages:

        if message["role"] == "system":
            text_encoding = message["content"]
        # 打印messages
        # print(f"message的content: {message["content"]}")
        elif isinstance(message["content"], list):
            for content in message["content"]:
                if content["type"] == "image_url":
                    # Extract base64 data and media type from data URL
                    # Example base64 data URL: data:image/png;base64,iVBOR...
                    image_base64 = cast(str, content["image_url"]["url"])
                    # Encode the image from the data URI or base64 string
                    encoded_image = encode_image_from_base64(image_base64)


    model = genai.GenerativeModel(
        model_name=model.value,
        generation_config=generation_config,
        
        # safety_settings = Adjust safety settings
        # See https://ai.google.dev/gemini-api/docs/safety-settings
    )
    im = Image.open(BytesIO(encoded_image))
 # Call the Gemini AI API with the specified model and generation config
    genai.configure(api_key=api_key)
    
    response = model.generate_content([text_encoding, im], stream=True)
    # generate_content(gemini_messages, stream=True)

    for chunk in response:
        await callback(chunk.text)

    if response._result and response._result.candidates:
        candidate = response._result.candidates[0]
        if candidate.content and candidate.content.parts:
            part = candidate.content.parts[0]
            if part.text:
                return part.text
    return ""