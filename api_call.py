
from typing import Dict, List, Union

import openai

from model_wrap import ModelWrapper
from openai_typing import OpenAIMessageWrapper, OpenAIResponseWrapper


def ChatCaller(
        systemMsg: str,
        messages: List[Union[OpenAIMessageWrapper, Dict[str, str]]],
        model: ModelWrapper
):
    if not isinstance(systemMsg, str):
        raise TypeError(
            f"Expected type of system message is str, got {type(systemMsg)}"
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
    messages = [
        {"role": "system", "content": systemMsg}
    ]+[
        _genMsg(x) for x in messages
    ]
    responseObj = openai.ChatCompletion.create(
        model=str(model),
        messages=messages
    )
    return OpenAIResponseWrapper(responseObj)
