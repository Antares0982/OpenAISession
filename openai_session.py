import json
import os
from dataclasses import dataclass
from os import urandom
from struct import unpack
from threading import Lock
from typing import Callable, Dict, List, Optional, Union

import openai
import tiktoken

from api_call import ObjectDict, completion_api_call
from model_wrap import (
    GPT3_5,
    MODEL_DICT,
    PRICING_DICT,
    TIKTOKEN_NAME_DICT,
    TOKEN_LIMIT_DICT,
    ModelWrapper,
)
from openai_session_logging import log
from openai_typing import OpenAIMessageWrapper

ModelType = Union[int, ModelWrapper]

_INPUT = 0
_OUTPUT = 1


class CallReturnData(object):
    msg: OpenAIMessageWrapper
    token_in: int
    token_out: int
    new_session_id: int
    reasoning_content: str | None


@dataclass
class SessionData(ObjectDict):
    id: int
    system_msg: str | None
    previous: int | None
    user_message: str
    assistant_message: str | None
    user_name: str | None
    assistant_name: str | None
    reasoning_content: str | None

    def gen_seq(self):
        return self.gen_seq_static(self.user_message, self.assistant_message, self.user_name, self.assistant_name)

    @staticmethod
    def gen_seq_static(user_message: str, assistant_message: str | None, user_name: str | None, assistant_name: str | None):
        ret = [
            {
                "role": "user",
                "content": user_message
            }
        ]
        if assistant_message:
            ret.append(
                {
                    "role": "assistant",
                    "content": assistant_message
                })
        if user_name:
            ret[0]["name"] = user_name
        if assistant_message and assistant_name:
            ret[1]["name"] = assistant_name
        return ret

    def serialize(self):
        return json.dumps(self, ensure_ascii=False, indent=4)

    @classmethod
    def deserialize(cls, fp):
        o = json.load(fp)
        if "reasoning_content" not in o:
            o["reasoning_content"] = None
        return cls(**o)


class OpenAISession:
    sessions_keeper: "SessionKeeper"

    def __init__(
        self,
        sid: int,
        system_msg: str | None,
        previous: int | None,
        user_message: str,
        assistant_message: str | None,
        user_name: str | None = None,
        assistant_name: str | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        if (previous is None) == (system_msg is None):
            raise RuntimeError("Logic error: previous and system_msg should be exclusive, and at least one should be provided")
        self.data = SessionData(sid, system_msg, previous, user_message, assistant_message, user_name, assistant_name, reasoning_content)
        self._lock = Lock()

    def save(self, folder: str) -> None:
        with self._lock:
            self._save(folder)

    def _save(self, folder: str) -> None:
        filename = os.path.join(folder, f"s_{self.data.id}.json")
        with open(filename, "w", encoding='utf-8') as f:
            f.write(self.data.serialize())

    # def parseHistory(self):
    #     for i, x in enumerate(self.history):
    #         if isinstance(x, dict):
    #             continue
    #         else:
    #             self.history[i] = {
    #                 "role": x.role,
    #                 "content": x.content
    #             }

    def call_self(self, model: ModelType):
        with self._lock:
            assert self.data.system_msg is not None
            model = model if isinstance(model, ModelWrapper) else ModelWrapper(model)
            response, token_in = self._internal_call(self.data.system_msg, self.data.gen_seq(), model)
            self._token_usage_hint(token_in, model, _INPUT)
            out_msg = response
            log(f"sid: {self.data.id} got response: {out_msg}")
            token_out = self._out_token_usage_check(model, out_msg.content)
            self.data.assistant_message = out_msg.content
            keeper = self.sessions_keeper
            self._save(keeper.data_directory)  # pylint: disable=no-member
            ret = CallReturnData()
            ret.msg = out_msg
            ret.token_in = token_in
            ret.token_out = token_out
            ret.new_session_id = self.data.id
            ret.reasoning_content = out_msg.reasoning_content
            return ret

    def call(self, new_msg: str, model: ModelType = GPT3_5) -> CallReturnData:
        with self._lock:
            chain = self.get_chain()
            sys_msg = chain[0].data.system_msg
            assert sys_msg is not None
            history = self.parse_history(chain)
            history += SessionData.gen_seq_static(new_msg, None, self.data.user_name, None)
            model = model if isinstance(model, ModelWrapper) else ModelWrapper(model)
            # do call
            response, token_in = self._internal_call(sys_msg, history, model)
            #
            self._token_usage_hint(token_in, model, _INPUT)
            out_msg = response
            log(f"sid: {self.data.id} got response: {out_msg}")
            # called successfully
            keeper = self.sessions_keeper
            # pylint: disable=no-member
            new_session_id = keeper.new_id()
            new_session = keeper.create(new_session_id, None, self.data.id, new_msg, out_msg.content, self.data.user_name,
                                        self.data.assistant_name, out_msg.reasoning_content)
            new_session.save(keeper.data_directory)
            # pylint: enable=no-member
            # check token usage
            token_out = self._out_token_usage_check(model, out_msg.content)
            ret = CallReturnData()
            ret.msg = out_msg
            ret.token_in = token_in
            ret.token_out = token_out
            ret.new_session_id = new_session_id
            ret.reasoning_content = out_msg.reasoning_content
            return ret

    def get_chain(self) -> List["OpenAISession"]:
        encountered = set()
        ret = []
        t = self
        while t is not None:
            if t.data.id in encountered:
                raise RuntimeError("Logic error: circular reference in session chain")
            ret.append(t)
            encountered.add(t.data.id)
            if t.data.previous is None:
                break
            t = self.sessions_keeper.get(t.data.previous)  # pylint: disable=no-member
            if t is None:
                raise RuntimeError("Logic error: cannot find previous session")
        ret.reverse()
        return ret

    @classmethod
    def parse_history(cls, chain: List["OpenAISession"]) -> List[Dict[str, str]]:
        return sum((x.data.gen_seq() for x in chain), [])

    def _internal_call(self, sys_msg: str, new_history: List[Dict[str, str]], model: ModelWrapper):
        index, token_used = self._calculate_propriate_cut_index(sys_msg, new_history, model)
        response = None
        #
        while index < len(new_history):
            try:
                response = completion_api_call(
                    sys_msg,
                    new_history[index:],  # type: ignore
                    model,
                )
                break
            except openai.BadRequestError as e:
                if index < len(new_history) - 1 and str(e).lower().find("maximum context length") != -1:
                    index += 2
                    log(f"Content too long for sid {self.data.id}, dicarding history...")
                    continue
                raise e
        if response is None:
            raise RuntimeError("Logic error: response is None")
        return response, token_used

    def _calculate_propriate_cut_index(self, sys_msg: str, new_history: List[Dict[str, str]], model: ModelWrapper) -> tuple[int, int]:
        token_max = self._get_token_max(model)
        token_start = self._count_token_for(model, sys_msg) + self._count_token_for(model, new_history[-1]["content"])
        #
        token = token_start
        tmp = 0
        index = len(new_history) - 2
        while token + tmp < token_max and index >= 0:
            token += tmp
            tmp = self._count_token_for(model, new_history[index]["content"])
            index -= 1
        cut_index = index + 1
        if (cut_index % 2) != 0:
            token -= self._count_token_for(model, new_history[cut_index]["content"])
            cut_index += 1
        return cut_index, token

    @staticmethod
    def _count_token_for(model: ModelWrapper, msg: str):
        """
        Count token.
        Return the token count of the message using the model.
        """
        enc = tiktoken.encoding_for_model(TIKTOKEN_NAME_DICT[model.id])
        return len(enc.encode(msg))

    @staticmethod
    def _get_token_max(model: ModelWrapper):
        ret = TOKEN_LIMIT_DICT.get(model.id)
        if ret is None:
            raise RuntimeError(f"Cannot find token limit of model id: {model.id}")
        return ret

    def _token_usage_hint(self, token_count: int, model: ModelWrapper, usage=_INPUT):
        price_per_1k = PRICING_DICT[model.id][usage]
        hint = 'input' if usage == _INPUT else 'output'
        log(f"Using model: {MODEL_DICT[model.id]}, token used ({hint}): {token_count}, estimated price: ${(token_count / 1000) * price_per_1k:.8f}")

    def _out_token_usage_check(self, model: ModelWrapper, content: str) -> int:
        """
        log and return the token count of the output
        """
        # the last message is the output
        token_count = self._count_token_for(model, content)
        self._token_usage_hint(token_count, model, _OUTPUT)
        return token_count


class SessionKeeper:
    def __init__(self, data_directory: str) -> None:
        self._sessions: Dict[int, OpenAISession] = {}
        self.data_directory = data_directory
        self._lock = Lock()
        self.load()

    def get(self, sid: int):
        with self._lock:
            return self._get(sid)

    def _get(self, sid: int):
        return self._sessions.get(sid)

    def create(
        self,
        sid: int,
        system_msg: str | None,
        previous: int | None,
        user_message: str,
        assistant_message: str,
        user_name: str | None = None,
        assistant_name: str | None = None,
        reasoning_content: str | None = None,
    ):
        with self._lock:
            return self._create(sid, system_msg, previous, user_message, assistant_message, user_name, assistant_name, reasoning_content)

    def _create(
        self,
        sid: int,
        system_msg: str | None,
        previous: int | None,
        user_message: str,
        assistant_message: str | None,
        user_name: str | None = None,
        assistant_name: str | None = None,
        reasoning_content: str | None = None,
    ):
        if sid in self._sessions:
            raise ValueError(f"Session {sid} already exists")
        t = OpenAISession(sid, system_msg, previous, user_message, assistant_message, user_name, assistant_name, reasoning_content)
        self._sessions[sid] = t
        o = self
        t.sessions_keeper = o
        return t

    def new_id(self, hint: Optional[int] = None) -> int:
        with self._lock:
            if hint is not None and not self._has(hint):
                return hint

            def _do() -> int:
                return unpack("!Q", urandom(8))[0]
            ans = _do()
            while self._has(ans):
                ans = _do()
            return ans

    def call(self, sid: int | None, new_msg: str, model: ModelType, system_msg: str | None,
             user_name: str | None = None,
             assistant_name: str | None = None) -> CallReturnData:
        if sid is None:
            new_id = self.new_id()
        with self._lock:
            if sid is None:
                if system_msg is None:
                    raise RuntimeError("system_msg is None when creating new session")
                outside_lock_call = self._call_new(new_id, new_msg, model, system_msg, user_name, assistant_name)
            else:
                outside_lock_call = self._call(sid, new_msg, model)
        # exit lock
        return outside_lock_call()

    def _call_new(
        self,
        new_id: int,
        new_msg: str,
        model: ModelType,
        system_msg: str | None,
        user_name: str | None,
        assistant_name: str | None
    ) -> Callable[[], CallReturnData]:
        session = self._create(new_id, system_msg, None, new_msg,
                               None, user_name, assistant_name, None)

        def outside_lock_call():
            return session.call_self(model)
        return outside_lock_call

    def _call(self, sid: int, new_msg: str, model: ModelType = GPT3_5) -> Callable[[], CallReturnData]:
        if not self._has(sid):
            raise ValueError("Invalid sid")
        session = self._sessions[sid]

        def outside_lock_call():
            return session.call(new_msg, model)
        return outside_lock_call

    def has(self, sid: int) -> bool:
        with self._lock:
            return self._has(sid)

    def _has(self, sid: int) -> bool:
        return sid in self._sessions

    def _create_with_data(self, data: SessionData):
        return self._create(data.id, data.system_msg, data.previous, data.user_message, data.assistant_message, data.user_name, data.assistant_name, data.reasoning_content)

    def load(self):
        with self._lock:
            for x in os.listdir(self.data_directory):
                _END = ".json"
                _START = "s_"
                if x.endswith(_END) and x.startswith(_START):
                    _s_name = x[len(_START):-len(_END)]
                    if _s_name.isdigit():
                        try:
                            int(_s_name)
                        except Exception:
                            continue
                    else:
                        continue
                    #
                    with open(os.path.join(self.data_directory, x), "r", encoding='utf-8') as f:
                        data = SessionData.deserialize(f)
                    self._create_with_data(data)

    def save(self):
        with self._lock:
            for x in self._sessions.values():
                x.save(self.data_directory)
