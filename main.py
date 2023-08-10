#!/usr/bin/env -S python3 -O

import os
import sys

import flask

from model_wrap import modelStringToModel
from openai_session import SessionKeeper

if __name__ == "__main__":
    dataFolder = os.environ.get("OPENAI_DATA_FOLDER")
    if dataFolder is None:
        print("environment variable OPENAI_DATA_FOLDER not set", file=sys.stderr)
        exit(1)
    if not os.path.exists(dataFolder):
        os.makedirs(dataFolder)
    systemMsgDefault = os.environ.get(
        "OPENAI_SYSTEM_MSG", "You are a helpful assistant.")
    port = os.environ.get("OPENAI_PORT")
    if port is None:
        print("environment variable OPENAI_PORT not set", file=sys.stderr)
        exit(1)
    port = int(port)

    sessions = SessionKeeper(dataFolder)
    app = flask.Flask(__name__)

    @app.route("/api", methods=["POST"])
    def callOpenAI():
        data: dict = flask.request.json
        try:
            if data is None:
                return "No data provided", 400
            if "sid" not in data:
                return "No session id provided", 400
            if "msg" not in data:
                return "No message provided", 400
            sid = int(data["sid"])
            msg = str(data["msg"])
            ensure_id = bool(data.get("ensure_id", False))
            systemMsg = str(
                data.get("systemMsg", systemMsgDefault))
            model = modelStringToModel(str(data.get("model", "GPT4")))

            if not ensure_id:
                return sessions.callCreateIfNotExist(sid, msg, systemMsg, model).content
            else:
                return sessions.call(sid, msg, model).content
        except Exception as e:
            return str(e), 400

    @app.route("/newid", methods=["GET"])
    def newId():
        # get id hint
        hint = flask.request.args.get("hint")
        try:
            hint = int(hint)
        except Exception:
            hint = None
        return str(sessions.newId(hint))

    @app.route("/create", methods=["GET"])
    def create():
        try:
            # get id from args
            sid = flask.request.args.get("sid")
            systemMsg = flask.request.args.get("systemMsg")
            if systemMsg is None:
                systemMsg = systemMsgDefault
            if sid is None:
                raise ValueError("No session id provided")
            sessions.create(int(sid), systemMsg)  # throw on error
            return "OK"
        except Exception as e:
            return str(e), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=port, debug=False)
