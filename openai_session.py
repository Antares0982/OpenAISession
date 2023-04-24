import json
import os
import random
import sys
from threading import Lock
from typing import Dict, List, Optional, Union

import openai

from api_call import ChatCaller
from model_wrap import GPT3_5, ModelWrapper
from openai_typing import OpenAIMessageWrapper

ModelType = Union[int, ModelWrapper]


class OpenAISession(object):
    def __init__(self, sid: int, systemMsg: str) -> None:
        self.id = sid
        self.systemMsg = systemMsg
        self.lock = Lock()
        self.history: List[Union[OpenAIMessageWrapper, Dict[str, str]]] = []
        self.sessions: "SessionKeeper" = None

    def save(self, folder: str) -> None:
        with self.lock:
            self._save(folder)

    def _save(self, folder: str) -> None:
        self.parseHistory()
        d = {
            "systemMsg": self.systemMsg,
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

    def call(self, newMsg: str, model: ModelType = GPT3_5) -> OpenAIMessageWrapper:
        with self.lock:
            tmpd = {
                "role": "user",
                "content": newMsg
            }
            self.parseHistory()
            newHistory = self.history + [tmpd]
            model = model if isinstance(
                model, ModelWrapper) else ModelWrapper(model)
            response, cutIndex = self._internal_call(newHistory, model)
            inMsg = response.choices[0].message
            print(f"sid: {self.id} got response: {inMsg}")
            # called successfully
            newHistory = newHistory[cutIndex:]
            newHistory.append(inMsg)
            self.history = newHistory
            self._save(self.sessions.dataFolder)
            return inMsg

    def _internal_call(self, newHistory, model: ModelWrapper):
        response = None
        index = 0
        while index < len(newHistory):
            try:
                response = ChatCaller(
                    self.systemMsg, newHistory[index:], model
                )
                break
            except openai.InvalidRequestError as e:
                if index < len(newHistory)-1 and str(e).lower().find("maximum context length") != -1:
                    index += 2
                    print(f"Content too long for sid {self.id}, dicarding history...",
                          file=sys.stderr)
                    continue
                raise e
        if response is None:
            raise RuntimeError("Logic error: response is None")
        return response, index


class SessionKeeper(object):
    def __init__(self, dataFolder: str) -> None:
        self.sessions: Dict[int, OpenAISession] = {}
        self.dataFolder = dataFolder
        self.lock = Lock()
        self.load()

    def create(self, sid: int, systemMsg: str):
        with self.lock:
            self._create(sid, systemMsg)

    def _create(self, sid: int, systemMsg: str):
        if sid in self.sessions:
            raise ValueError(f"Session {sid} already exists")
        t = OpenAISession(sid, systemMsg)
        self.sessions[sid] = t
        t.sessions = self

    def newId(self, hint: Optional[int]) -> int:
        with self.lock:
            if hint is not None and not self._has(hint):
                return hint
            ans = random.randint(0, (2**32)-1)
            while self._has(ans):
                ans = random.randint(0, (2**32)-1)

    def call(self, sid: int, newMsg: str, model: ModelType = GPT3_5) -> OpenAIMessageWrapper:
        with self.lock:
            return self._call(sid, newMsg, model)

    def _call(self, sid: int, newMsg: str, model: ModelType = GPT3_5) -> OpenAIMessageWrapper:
        t = self.sessions.get(sid)
        if t is None:
            raise ValueError(f"Session {sid} does not exist")
        return t.call(newMsg, model)

    def callCreateIfNotExist(
            self,
            sid: int,
            newMsg: str,
            systemMsg: str,
            model: ModelType = GPT3_5
    ) -> OpenAIMessageWrapper:
        print(f"Calling session {sid}...")
        with self.lock:
            if not self._has(sid):
                print(f"Session {sid} does not exist, creating...")
                self._create(sid, systemMsg)
            return self._call(sid, newMsg, model)

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
                if x.endswith(".json") and x.startswith("s_"):
                    try:
                        sid = int(x[2:-5])
                    except Exception:
                        continue
                    with open(os.path.join(self.dataFolder, x), "r") as f:
                        d = json.load(f)

                    systemMsg = d.get("systemMsg")
                    history = d.get("history")
                    if systemMsg is None or history is None:
                        continue
                    if not isinstance(systemMsg, str) or not isinstance(history, list):
                        continue
                    if not _check_history(history):
                        continue
                    self._create(sid, systemMsg)
                    t = self.sessions[sid]
                    t.history = history

    def save(self):
        with self.lock:
            for x in self.sessions.values():
                x.save(self.dataFolder)
