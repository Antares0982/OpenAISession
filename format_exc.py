import json
import re
import traceback
from traceback import TracebackException
from types import TracebackType
from typing import Any

from httpx import Response

_sentinel = getattr(traceback, "_sentinel")
_parse_value_tb = getattr(traceback, "_parse_value_tb")
_pattern = re.compile(r"^([\s]*)File")


def _matcher(text: str) -> int:
    match = re.match(_pattern, text)
    if match:
        return len(match.group(1))
    else:
        return -1


def _short_format(val):
    if type(val).__name__ == "Response":
        ret = "Response:\n"
        for k, v in val.__dict__.items():
            ret += f"  {k} = {v}\n"
        return ret
    rep = str(val)
    if len(rep) > 256:
        suffix = "...<Too long to show>"
        rep = rep[:256 - len(suffix)] + suffix
    if isinstance(val, str):
        rep = f'"{rep}"'
    return rep


class _TracebackExceptionWithLocalVars(TracebackException):
    def format(self, *, tb: TracebackType | Any = _sentinel, chain=True, _ctx=None):
        frame = tb.tb_frame
        for val in super().format(chain=chain, _ctx=_ctx):
            yield val
            space_count = _matcher(val)
            if tb is not None and tb is not _sentinel and space_count >= 0:
                prefix = ' ' * (space_count + 2)
                prefix2 = ' ' * (space_count + 4)
                cur_locals = frame.f_locals
                if len(cur_locals) > 0:
                    yield f"{prefix}Local variables:"
                    for k, v in cur_locals.items():
                        yield f"{prefix2}{k} = {_short_format(v)}"
                    yield "\n"
                tb = tb.tb_next
                frame = tb.tb_frame if tb is not None else None


def format_exception_with_local_vars(exc, /, value=_sentinel, tb=_sentinel, limit=None, chain=True):
    value, tb = _parse_value_tb(exc, value, tb)
    te = _TracebackExceptionWithLocalVars(type(value), value, tb, limit=limit, compact=True)
    return list(te.format(tb=tb, chain=chain))
