
__inited = False
__logging_interface = None


def _try_import():
    try:
        import pika_interface_blocking
        return True
    except ImportError:
        return False


def log(msg, key=None):
    global __inited, __logging_interface
    if not __inited:
        __inited = True
        try:
            if not _try_import():
                import subprocess
                subprocess.run(
                    'curl -O https://raw.githubusercontent.com/Antares0982/PikaInterface/main/pika_interface_blocking.py',
                    shell=True
                )
            import pika_interface_blocking
            __logging_interface = pika_interface_blocking.send_message
        except Exception:
            print("Failed to import pika_interface_blocking")
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
