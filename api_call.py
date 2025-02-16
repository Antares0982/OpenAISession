
import os
from time import perf_counter
from typing import Any, Iterable, cast

import openai
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
)

from model_wrap import DEEPSEEK_R1, MODEL_DICT, O1, O3_MINI, ModelWrapper
from openai_session_logging import log

OPENAI_CLIENT = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

DEEPSEEK_CLIENT = openai.OpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
)


class ObjectDict(dict):
    """
    General json object that allows attributes 
    to be bound to and also behaves like a dict.
    """

    def __getattr__(self, attr: str):
        return self.get(attr)

    def __setattr__(self, attr: str, value):
        self[attr] = value


class CompletionAPIResponse(ObjectDict):
    role: str
    content: str
    reasoning_content: str


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
    # call
    _t0 = perf_counter()
    kw: dict[str, Any] = dict()
    if model_str == MODEL_DICT[O1] or model_str == MODEL_DICT[O3_MINI]:
        kw["reasoning_effort"] = "high"
    responseObj = client.chat.completions.create(
        model=model_str,
        messages=messages_send,
        stream=True,
        **kw
    )
    content = ""
    _reasoning_content = ""
    role = ""
    for chunk in responseObj:
        cur_delta = chunk.choices[0].delta
        if not role:
            role = cur_delta.role
        content += cur_delta.content if cur_delta.content else ""  # getattr(cur_delta, "reasoning_content", "")
        cur_reasoning_content = getattr(cur_delta, "reasoning_content", None)
        _reasoning_content += cur_reasoning_content if cur_reasoning_content else ""
    # response process done
    _t1 = perf_counter()
    log(f"API call took {_t1 - _t0:.2f}s, model={model_str}")
    reasoning_content = None if not _reasoning_content else _reasoning_content
    return CompletionAPIResponse(role=role, content=content, reasoning_content=reasoning_content)
