# OpenAISession
OpenAI session server

### API

#### `/api`, methods: `POST`

data format: JSON.

returns: reply text from openai GPT model.

JSON fields:

* `"sid"`: session id, int (required)
* `"msg"`: message to send, string (required)
* `"system_msg"`: system message to override, string (optional, default not to override). If provided, the system message of this session will be replaced by the new message in later API calls.
* `"model"`: model to use, string (optional, default `"GPT4"`). Supported values:
  * `"GPT3_5"`
  * `"GPT3_5_0301"`
  * `"GPT4"`
  * `"GPT4_0314"`
  * `"GPT4_32K"`
  * `"GPT4_32K_0314"`

#### `/create`, methods: `POST`

data format: JSON.

returns: the real session id (sid) to use in `/api`.

JSON fields:

* `"sid"` hint `sid`, (optional). If provided and the value is unoccupied, the server returns the value as `sid`. Otherwise, return a new unoccupied `sid`.

* `system_msg`: system message for this session, string (optional, default `"You are a helpful assistant."`)

