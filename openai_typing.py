from typing import TYPE_CHECKING, Any


class OpenAIMessageWrapper(object):
    def __init__(self, msgObj) -> None:
        # self.api_key: str = msgObj.api_key
        # self.organization: str = msgObj.organization
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
