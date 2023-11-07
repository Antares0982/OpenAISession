GPT3_5 = 0
GPT3_5_16K = 1
GPT4 = 2
GPT4_32K = 3
GPT4_TURBO = 4
MAX = 5

MODEL_DICT = {
    GPT3_5: "gpt-3.5-turbo",
    GPT3_5_16K: "gpt-3.5-turbo-16k",
    GPT4: "gpt-4",
    GPT4_32K: "gpt-4-32k",
    GPT4_TURBO: "gpt-4-1106-preview",
}

TIKTOKEN_NAME_DICT = {
    GPT3_5: "gpt-3.5-turbo",
    GPT3_5_16K: "gpt-3.5-turbo",
    GPT4: "gpt-4",
    GPT4_32K: "gpt-4",
    GPT4_TURBO: "gpt-4",
}

TOKEN_LIMIT_DICT = {
    GPT3_5: 4097,
    GPT3_5_16K: 16385,
    GPT4: 8192,
    GPT4_32K: 32768,
    GPT4_TURBO: 128000,
}

STR_MODEL_DICT = {
    "GPT3_5": GPT3_5,
    "GPT3_5_16K": GPT3_5_16K,
    "GPT4": GPT4,
    "GPT4_32K": GPT4_32K,
    "GPT4_TURBO": GPT4_TURBO,
}

PRICING_DICT = {
    GPT3_5: (0.001, 0.002),
    GPT3_5: (0.001, 0.002),
    GPT4: (0.03, 0.06),
    GPT4_32K: (0.06, 0.12),
    GPT4_TURBO: (0.01, 0.03),
}


class ModelWrapper(object):
    def __init__(self, modelId: int) -> None:
        self.id: int = modelId

    def get(self) -> str:
        ans = MODEL_DICT.get(self.id)
        if ans is None:
            raise ValueError(f"Model id {self.id} is not valid")
        return ans

    def __str__(self) -> str:
        return self.get()


def model_string_to_model(s: str) -> ModelWrapper:
    model_int = STR_MODEL_DICT.get(s)
    if model_int is None:
        if s.isdigit():
            model_int = int(s)
            if model_int < 0 or model_int >= MAX:
                raise ValueError(f"Model {s} is not valid")
            return ModelWrapper(model_int)
        raise ValueError(f"Model {s} is not valid")
    return ModelWrapper(model_int)
