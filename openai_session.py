import json
import os
import sys
import uuid
from threading import Lock
from typing import Dict, List, Optional, Union

import openai
import tiktoken

from api_call import ChatCaller
from model_wrap import (GPT3_5, TIKTOKEN_NAME_DICT, TOKEN_LIMIT_DICT,
                        ModelWrapper)
from openai_typing import OpenAIMessageWrapper

ModelType = Union[int, ModelWrapper]


class OpenAISession(object):
    def __init__(self, sid: int, system_msg: str) -> None:
        self.id = sid
        self._system_msg = system_msg
        self.lock = Lock()
        self.history: List[Union[OpenAIMessageWrapper, Dict[str, str]]] = []
        self.sessions: "SessionKeeper" = None  # type: ignore
        self._cached_token_count: List[int] = []

    def save(self, folder: str) -> None:
        with self.lock:
            self._save(folder)

    def _save(self, folder: str) -> None:
        self.parseHistory()
        d = {
            "systemMsg": self._system_msg,
            "history": self.history
        }
        filename = os.path.join(folder, f"s_{self.id}.json")
        with open(filename, "w") as f:
            json.dump(d, f, indent=4, ensure_ascii=False)

    def parseHistory(self):
        for i, x in enumerate(self.history):
            if isinstance(x, OpenAIMessageWrapper):
                self.history[i] = {
                    "role": x.role,
                    "content": x.content
                }
            elif isinstance(x, dict):
                continue
            else:
                raise TypeError(
                    f"Expected OpenAIMessageWrapper or Dict[str, str], got {type(x)}"
                )

    def call(self, new_msg: str, model: ModelType = GPT3_5, override_system_msg: Optional[str] = None) -> OpenAIMessageWrapper:
        with self.lock:
            if override_system_msg is not None:
                self._system_msg = override_system_msg
                if len(self._cached_token_count) > 1:
                    self._cached_token_count[0] = -1
            tmpd = {
                "role": "user",
                "content": new_msg
            }
            self.parseHistory()
            newHistory = self.history + [tmpd]
            model = model if isinstance(model, ModelWrapper) else ModelWrapper(model)
            response, cutIndex = self._internal_call(newHistory, model)
            in_msg = response.choices[0].message
            print(f"sid: {self.id} got response: {in_msg}")
            # called successfully
            newHistory = newHistory[cutIndex:]
            newHistory.append(in_msg)
            self.history = newHistory
            self._save(self.sessions.dataFolder)
            return in_msg

    def _internal_call(self, new_history, model: ModelWrapper):
        index = self._calculate_propriate_cut_index(model)
        response = None
        #
        while index < len(new_history):
            try:
                response = ChatCaller(
                    self._system_msg, new_history[index:], model
                )
                break
            except openai.InvalidRequestError as e:
                if index < len(new_history)-1 and str(e).lower().find("maximum context length") != -1:
                    index += 2
                    print(f"Content too long for sid {self.id}, dicarding history...",
                          file=sys.stderr)
                    continue
                raise e
        if response is None:
            raise RuntimeError("Logic error: response is None")
        return response, index

    def _calculate_propriate_cut_index(self, model: ModelWrapper):
        end = len(self.history)
        #
        token_count = self._count_token(model, -1)
        for idx in range(end):
            token_count += self._count_token(model, idx)
        #
        token_max = self._get_token_max(model)
        if token_count < token_max:
            return 0
        #
        start = 0
        while start < end-1:
            token_count -= self._count_token(model, start)
            start += 1
            token_count -= self._count_token(model, start)
            start += 1
            if token_count < token_max:
                print(f"Warning: history too long, discarding history from index {start}")
                return start
        #
        raise RuntimeError("Logic error: cannot find appropriate cut index")

    def _count_token(self, model: ModelWrapper, idx: int):
        if len(self._cached_token_count) < len(self.history)+1:
            self._cached_token_count += [-1]*(len(self.history)+1-len(self._cached_token_count))
        if self._cached_token_count[idx+1] != -1:
            return self._cached_token_count[idx+1]
        #
        if idx == -1:
            msg = self._system_msg
        else:
            msg = self.history[idx]["content"]  # type: ignore
        enc = tiktoken.encoding_for_model(TIKTOKEN_NAME_DICT[model.id])
        r = len(enc.encode(msg))
        self._cached_token_count[idx+1] = r
        return r

    def _get_token_max(self, model: ModelWrapper):
        ret = TOKEN_LIMIT_DICT.get(model.id)
        if ret is None:
            raise RuntimeError(f"Cannot find token limit of model id: {model.id}")
        return ret


class SessionKeeper(object):
    def __init__(self, dataFolder: str) -> None:
        self.sessions: Dict[int, OpenAISession] = {}
        self.dataFolder = dataFolder
        self.lock = Lock()
        self.load()

    def create(self, sid: int, system_msg: str):
        with self.lock:
            self._create(sid, system_msg)

    def _create(self, sid: int, system_msg: str):
        if sid in self.sessions:
            raise ValueError(f"Session {sid} already exists")
        t = OpenAISession(sid, system_msg)
        self.sessions[sid] = t
        t.sessions = self

    def newId(self, hint: Optional[int] = None) -> int:
        with self.lock:
            if hint is not None and not self._has(hint):
                return hint

            def _do() -> int:
                return uuid.uuid4().int
            ans = _do()
            while self._has(ans):
                ans = _do()
            return ans

    def call(self, sid: int, new_msg: str, model: ModelType = GPT3_5, override_system_msg: Optional[str] = None) -> OpenAIMessageWrapper:
        with self.lock:
            return self._call(sid, new_msg, model, override_system_msg)

    def _call(self, sid: int, new_msg: str, model: ModelType = GPT3_5, override_system_msg: Optional[str] = None) -> OpenAIMessageWrapper:
        t = self.sessions.get(sid)
        if t is None:
            raise ValueError(f"Session {sid} does not exist")
        return t.call(new_msg, model, override_system_msg)

    def callCreateIfNotExist(
            self,
            sid: int,
            new_msg: str,
            system_msg: str,
            model: ModelType = GPT3_5
    ) -> OpenAIMessageWrapper:
        print(f"Calling session {sid}...")
        with self.lock:
            if not self._has(sid):
                print(f"Session {sid} does not exist, creating...")
                self._create(sid, system_msg)
            return self._call(sid, new_msg, model)

    def has(self, sid: int) -> bool:
        with self.lock:
            return self._has(sid)

    def _has(self, sid: int) -> bool:
        return sid in self.sessions

    def load(self):
        def _check_history(history: list):
            for x in history:
                if not isinstance(x, dict) or x.get("role") is None or x.get("content") is None:
                    return False
                if len(x) != 2:
                    return False
            return True

        with self.lock:
            for x in os.listdir(self.dataFolder):
                _END = ".json"
                _START = "s_"
                if x.endswith(_END) and x.startswith(_START):
                    _s_name = x[len(_START):-len(_END)]
                    if _s_name.isdigit():
                        sid = int(_s_name)
                    else:
                        continue
                    #
                    try:
                        with open(os.path.join(self.dataFolder, x), "r") as f:
                            d: dict = json.load(f)
                    except Exception:
                        continue
                    #
                    system_msg = d.get("systemMsg")
                    history = d.get("history")
                    if system_msg is None or history is None:
                        continue
                    if not isinstance(system_msg, str) or not isinstance(history, list):
                        continue
                    if not _check_history(history):
                        continue
                    self._create(sid, system_msg)
                    t = self.sessions[sid]
                    t.history = history

    def save(self):
        with self.lock:
            for x in self.sessions.values():
                x.save(self.dataFolder)
