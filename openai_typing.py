from typing import TYPE_CHECKING, Any, List


class OpenAIMessageWrapper(object):
    def __init__(self, msgObj) -> None:
        self.api_key: str = msgObj.api_key
        self.organization: str = msgObj.organization
        self.role: str = msgObj.role
        self.content: str = msgObj.content

    def __str__(self) -> str:
        return str({
            "role": self.role,
            "content": self.content
        })

    if TYPE_CHECKING:
        def __getitem__(self, key: str) -> Any:
            return self.__dict__[key]


class OpenAIChoicesWrapper(object):
    def __init__(self,  chObj) -> None:
        self.index: int = chObj.index
        self.api_key: str = chObj.api_key
        self.organization: str = chObj.organization
        self.message: OpenAIMessageWrapper = OpenAIMessageWrapper(
            chObj.message)
        self.finish_reason: str = chObj.finish_reason


class OpenAIResponseWrapper(object):
    def __init__(self, responseObj) -> None:
        self.api_key: str = responseObj.api_key
        self.openai_id: str = responseObj.openai_id
        self.organization: str = responseObj.organization
        self.response_ms: int = responseObj.response_ms
        self.id: str = responseObj.id
        self.object: str = responseObj.object
        self.created: int = responseObj.created
        self.model: str = responseObj.model
        self.choices: List[OpenAIChoicesWrapper] = [
            OpenAIChoicesWrapper(x) for x in responseObj.choices]
