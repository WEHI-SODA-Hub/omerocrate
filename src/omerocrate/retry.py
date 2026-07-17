import logging

import omero
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3


def _log_conflict(retry_state: RetryCallState) -> None:
    # Positional args (minus self) identify the object being saved
    fn = retry_state.fn
    args = ", ".join(repr(arg) for arg in retry_state.args[1:])
    logger.warning(
        "Conflicting update in %s(%s) (attempt %d/%d), refetching and retrying",
        fn.__qualname__ if fn is not None else "<unknown>",
        args,
        retry_state.attempt_number,
        MAX_ATTEMPTS,
    )


retry_omero_conflict = retry(
    retry=retry_if_exception_type(omero.OptimisticLockException),
    stop=stop_after_attempt(MAX_ATTEMPTS),
    wait=wait_fixed(2),
    before_sleep=_log_conflict,
    reraise=True,
)
"""
Retries an OMERO write that failed with ``OptimisticLockException``

The decorated function must refetch the object it saves — retrying a save of the
same stale wrapper fails every time. Once all attempts are exhausted the original
``OptimisticLockException`` is re-raised.
"""
