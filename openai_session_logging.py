_inited = False
_logging_interface = None


def _try_import():
    try:
        import pika_interface_blocking  # pylint: disable=W0611
        return True
    except ImportError:
        return False


def log(msg, key=None):
    global _inited, _logging_interface
    if not _inited:
        _inited = True
        try:
            if not _try_import():
                import subprocess
                subprocess.run(
                    'curl -O https://raw.githubusercontent.com/Antares0982/PikaInterface/main/pika_interface_blocking.py',
                    shell=True,
                    check=True,
                )
            import pika_interface_blocking
            _logging_interface = pika_interface_blocking.send_message
        except Exception:
            print("Failed to import pika_interface_blocking")
    if _logging_interface is None:
        return
    #
    if key is None:
        routing_key = "logging.openai_session"
    else:
        routing_key = f"logging.openai_session.{key}"
    #
    try:
        _logging_interface(routing_key, msg)
        print(f"[{routing_key}] {msg}")
    except Exception as e:
        print(e)
