import logging
import weakref
from typing import Any

from copy import deepcopy

logger = logging.getLogger(__name__)


class Handler:

    def __init__(self, hook_id: str, runner):
        self._hook_id = hook_id
        self._runner = weakref.ref(runner)

    def clear(self):
        runner = self._runner()
        if runner is None:
            return

        hooks = runner._hooks
        ops = runner._ops
        if self._hook_id not in hooks:
            return

        for op_id in hooks[self._hook_id]:
            if op_id not in ops:
                continue

            op = ops[op_id]
            op.clear(runner)
            runner.delete_op(op_id)

        del hooks[self._hook_id]

    def __enter__(self):
        return self

    def __exit__(self, type: Any, value: Any, tb: Any):
        self.clear()

    def __str__(self):
        return f"Handler: hook {self._hook_id}"
