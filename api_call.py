
import os
from typing import Dict, List, Union

import openai

from model_wrap import ModelWrapper
from openai_typing import OpenAIMessageWrapper


client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)


def ChatCaller(
        system_msg: str,
        messages: List[Union[OpenAIMessageWrapper, Dict[str, str]]],
        model: ModelWrapper
):
    if not isinstance(system_msg, str):
        raise TypeError(
            f"Expected type of system message is str, got {type(system_msg)}"
        )

    def _genMsg(x) -> Dict[str, str]:
        if isinstance(x, OpenAIMessageWrapper):
            return {
                "role": x.role,
                "content": x.content
            }
        elif isinstance(x, dict):
            return x
        else:
            raise TypeError(
                f"Expected OpenAIMessageWrapper or Dict[str, str], got {type(x)}"
            )
    messages_send = [
        {"role": "system", "content": system_msg}
    ] + [
        _genMsg(x) for x in messages
    ]
    responseObj = client.chat.completions.create(
        model=str(model),
        messages=messages_send
    )
    return responseObj
