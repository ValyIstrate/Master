import datetime
import sys
import traceback


def _format_message(level: str, msg: str) -> str:
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return f"{current_time} ---> {level}: {msg}"


def exception(msg: str):
    log_message = _format_message("EXCEPTION", msg)
    print(log_message)
    traceback_message = traceback.format_exc()
    print(traceback_message)


def warn(msg: str):
    log_message = _format_message("WARNING", msg)
    print(log_message)


def simple_text(msg: str):
    print(msg)


def error(msg: str):
    log_message = _format_message("ERROR", msg)
    print(log_message)


def debug(msg: str):
    log_message = _format_message("DEBUG", msg)
    print(log_message)


def info(msg: str):
    log_message = _format_message("INFO", msg)
    print(log_message)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    exception("An unhandled exception occurred:")
    error("The program encountered a fatal error and will exit. Please check the log file for details.")
    sys.exit(1)
