
__inited = False
__logging_interface = None


def log(msg, key=None):
    global __inited, __logging_interface
    if not __inited:
        __inited = True
        try:
            import subprocess
            # reinstall rabbitmq_interface to ensure using the latest version
            subprocess.run(
                'curl https://api.github.com/repos/Antares0982/RabbitMQInterface/contents/rabbitmq_interface.py | jq -r ".content" | base64 --decode > rabbitmq_interface.py',
                shell=True
            )
            import rabbitmq_interface
            __logging_interface = rabbitmq_interface.send_message
        except Exception:
            print("Failed to import rabbitmq_interface")
    if __logging_interface is None:
        return
    #
    if key is None:
        routing_key = "logging.openai_session"
    else:
        routing_key = f"logging.openai_session.{key}"
    #
    try:
        __logging_interface(routing_key, msg)
        print(f"[{routing_key}] {msg}")
    except Exception as e:
        print(e)
