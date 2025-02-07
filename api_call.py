
import os
from typing import Iterable, cast

import openai
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam

from model_wrap import ModelWrapper
from model_wrap import MODEL_DICT, DEEPSEEK_R1

OPENAI_CLIENT = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

DEEPSEEK_CLIENT = openai.OpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
)


def completion_api_call(
        system_msg: str,
        messages: Iterable[ChatCompletionMessageParam],
        model: ModelWrapper
):
    messages_send = [
        cast(ChatCompletionSystemMessageParam, {"role": "system", "content": system_msg})
    ] + list(messages)
    model_str = str(model)
    if model_str == MODEL_DICT[DEEPSEEK_R1]:
        client = DEEPSEEK_CLIENT
        print("Using DeepSeek")
    else:
        client = OPENAI_CLIENT
    responseObj = client.chat.completions.create(
        model=model_str,
        messages=messages_send
    )
    return responseObj
    # log(f"system_msg: {system_msg}")
    # log(f"messages: {messages}")
    # log(f"model: {model}")
    # class FakeContent:
    #     content = "Test response!"
    # class FakeMessage:
    #     message = FakeContent
    # class FakeResponse:
    #     # response.choices[0].message
    #     choices = [
    #         FakeMessage
    #     ]
    # return FakeResponse
