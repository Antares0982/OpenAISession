
__connection = None


def _conn():
    global __connection
    if __connection is None:
        import pika
        __connection = pika.BlockingConnection()
    return __connection


def log(msg, key=None):
    try:
        import rabbitmq_interface
    except ImportError:
        import subprocess
        subprocess.run(
            'curl https://api.github.com/repos/Antares0982/RabbitMQInterface/contents/rabbitmq_interface.py | jq -r ".content" | base64 --decode > rabbitmq_interface.py',
            shell=True
        )
        import rabbitmq_interface
    if key is None:
        routing_key = "logging.openai_session"
    else:
        routing_key = f"logging.openai_session.{key}"
    try:
        rabbitmq_interface.send_message(routing_key, msg, _conn())
        print(f"[{routing_key}] {msg}")
    except Exception as e:
        print(e)


def close_conn():
    global __connection
    if __connection is not None:
        print("Closing connection...")
        __connection.close()
        __connection = None
