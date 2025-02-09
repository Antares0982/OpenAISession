#!/usr/bin/env -S python3 -O

import os
import sys
from typing import TYPE_CHECKING

import openai

from model_wrap import STR_MODEL_DICT, model_string_to_model
from openai_session import SessionKeeper
from openai_session_logging import log


if TYPE_CHECKING:
    from flask.typing import ResponseReturnValue


SYSTEM_MSG_DEFAULT = "You are a helpful assistant."
DEFAULT_MODEL = "GPT4O_MINI"


def set_default_model(model_raw_string) -> None:
    from model_wrap import MODEL_DICT
    _k = None
    for k, v in MODEL_DICT.items():
        if v == model_raw_string:
            _k = k
            break
    if _k is None:
        return
    global DEFAULT_MODEL

    for k2, v2 in STR_MODEL_DICT.items():
        if v2 == _k:
            DEFAULT_MODEL = k2
            log(f"Set default model to {model_raw_string}")
            return
    log(f"Failed to set default model to {model_raw_string}")


if __name__ == "__main__":
    if os.environ.get("http_proxy") is None or os.environ.get("https_proxy") is None:
        print("Please set http_proxy and https_proxy environment variables", file=sys.stderr)
        exit(1)

    import flask

    data_directory = os.environ.get("OPENAI_DATA_FOLDER")
    if data_directory is None:
        print("environment variable OPENAI_DATA_FOLDER not set", file=sys.stderr)
        exit(1)
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    port_str = os.environ.get("OPENAI_PORT")
    if port_str is None:
        print("environment variable OPENAI_PORT not set", file=sys.stderr)
        exit(1)
    port = int(port_str)

    _use_model_name = os.environ.get("OPENAI_DEFAULT_MODEL_STRING")
    if _use_model_name is not None:
        set_default_model(_use_model_name)

    sessions = SessionKeeper(data_directory)
    app = flask.Flask(__name__)

    def _on_exception(e: Exception):
        try:
            if app.debug:
                import traceback
                ret = traceback.format_exc()
                log(ret)
                return ret
        except Exception:
            ...
        ret = repr(e)
        log(ret)
        return ret

    @app.route("/api", methods=["POST"])
    def api() -> "ResponseReturnValue":
        data: dict = flask.request.json  # type: ignore
        try:
            if not data:
                return "No data provided", 400
            if "msg" not in data:
                return "No message provided", 400
            sid_str = data.get("sid")
            sid = int(sid_str) if sid_str else None
            msg = data["msg"]
            system_msg = data.get("system_msg")
            if sid is not None and system_msg is not None:
                return "Cannot specify system_msg for an existing session", 400
            if sid is None and system_msg is None:
                system_msg = SYSTEM_MSG_DEFAULT
            user_name = data.get("user_name")
            if sid is not None and user_name is not None:
                return "Cannot specify user_name for an existing session", 400
            assistant_name = data.get("assistant_name")
            if sid is not None and assistant_name is not None:
                return "Cannot specify assistant_name for an existing session", 400
            model = model_string_to_model(str(data.get("model", DEFAULT_MODEL)))

            def string_check(x):
                return not (x is not None and not isinstance(x, str))
            #
            if not all(map(string_check, [msg, system_msg, user_name, assistant_name])):
                return "msg, system_msg, user_name, assistant_name must be strings", 400
            # all check completed
            response = sessions.call(sid, msg, model, system_msg, user_name, assistant_name)
            ret = {
                "text": response.msg.content,
                "token_in": response.token_in,
                "token_out": response.token_out,
                "new_session_id": response.new_session_id,
            }
            reasoning_content = getattr(response.msg, "reasoning_content", None)
            if reasoning_content:
                ret["reasoning_content"] = reasoning_content
            return ret
        except openai.RateLimitError:
            log("Token rate limit exceeded!")
            return "Token rate limit exceeded", 503
        except Exception as e:
            err = _on_exception(e)
            print(err)
            return err, 400

    @app.route("/list_models", methods=["GET"])
    def list_models() -> "ResponseReturnValue":
        return flask.jsonify(list(STR_MODEL_DICT.keys()))


if __name__ == "__main__":
    debug = False
    if debug:
        import traceback
        from format_exc import format_exception_with_local_vars
        traceback.format_exception = format_exception_with_local_vars
    app.run(host="127.0.0.1", port=port, debug=debug)
