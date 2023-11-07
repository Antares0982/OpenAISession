#!/usr/bin/env -S python3 -O

import os
import sys
from typing import TYPE_CHECKING, Any

from model_wrap import model_string_to_model
from openai_session import SessionKeeper
from openai_session_logging import close_conn, log

if TYPE_CHECKING:
    from flask.typing import ResponseReturnValue


SYSTEM_MSG_DEFAULT = "You are a helpful assistant."


if __name__ == "__main__":
    import atexit

    def on_exit():
        try:
            close_conn()
        except Exception:
            ...
    atexit.register(on_exit)
    import flask

    dataFolder = os.environ.get("OPENAI_DATA_FOLDER")
    if dataFolder is None:
        print("environment variable OPENAI_DATA_FOLDER not set", file=sys.stderr)
        exit(1)
    if not os.path.exists(dataFolder):
        os.makedirs(dataFolder)

    port_str = os.environ.get("OPENAI_PORT")
    if port_str is None:
        print("environment variable OPENAI_PORT not set", file=sys.stderr)
        exit(1)
    port = int(port_str)

    sessions = SessionKeeper(dataFolder)
    app = flask.Flask(__name__)

    def _on_exception(e: Exception):
        try:
            if app.debug:
                import traceback
                return traceback.format_exc()
        except Exception:
            ...
        return repr(e)

    @app.route("/api", methods=["POST"])
    def api() -> "ResponseReturnValue":
        data: dict = flask.request.json  # type: ignore
        try:
            if data is None:
                return "No data provided", 400
            if "sid" not in data or not str(data["sid"]).isdigit():
                return "No session id provided", 400
            if "msg" not in data:
                return "No message provided", 400
            sid = int(data["sid"])
            msg = str(data["msg"])
            override_msg = data.get("system_msg", None)
            if override_msg is not None:
                override_msg = str(override_msg)
            model = model_string_to_model(str(data.get("model", "GPT4")))
            response = sessions.call(sid, msg, model, override_msg)
            ret = response.content
        except Exception as e:
            return _on_exception(e), 400
        return ret

    @app.route("/create", methods=["POST"])
    def create() -> "ResponseReturnValue":
        try:
            data: dict = flask.request.json  # type: ignore
            # get id from args
            _sid: Any = data.get("sid")
            system_msg: Any = data.get("system_msg")
            sid = sessions.newId(None if _sid is None else int(_sid))
            #
            sessions.create(sid, str(system_msg) if system_msg is not None else SYSTEM_MSG_DEFAULT)  # throw on error
            ret = str(sid)
        except Exception as e:
            return _on_exception(e), 400
        try:
            log(f"Created session {sid}")
        except Exception:
            ...
        return ret

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=port, debug=False)
