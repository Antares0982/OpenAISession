GPT3_5 = 0
GPT3_5_0301 = 1
GPT4 = 2
GPT4_0314 = 3
GPT4_32K = 4
GPT4_32K_0314 = 5


class ModelWrapper(object):
    ModelDict = {
        GPT3_5_0301: "gpt-3.5-turbo-0301",
        GPT3_5: "gpt-3.5-turbo",
        GPT4: "gpt-4",
        GPT4_0314: "gpt-4-0314",
        GPT4_32K: "gpt-4-32k",
        GPT4_32K_0314: "gpt-4-32k-0314"
    }

    def __init__(self, modelId: int) -> None:
        self.id: int = modelId

    def get(self) -> str:
        ans = ModelWrapper.ModelDict.get(self.id)
        if ans is None:
            raise ValueError(f"Model id {self.id} is not valid")
        return ans

    def __str__(self) -> str:
        return self.get()


def modelStringToModel(s: str) -> ModelWrapper:
    if s == "GPT3_5":
        return ModelWrapper(GPT3_5)
    if s == "GPT3_5_0301":
        return ModelWrapper(GPT3_5_0301)
    if s == "GPT4":
        return ModelWrapper(GPT4)
    if s == "GPT4_0314":
        return ModelWrapper(GPT4_0314)
    if s == "GPT4_32K":
        return ModelWrapper(GPT4_32K)
    if s == "GPT4_32K_0314":
        return ModelWrapper(GPT4_32K_0314)
    if s.isdigit():
        return ModelWrapper(int(s))
    raise ValueError(f"Model {s} is not valid")
