from asyncio import timeout as asyncio_timeout
from types import TracebackType
from typing import Any, Optional, Type
import asyncio
import enum
import sys
if sys.version_info[:2] >= (3, 11):
else:

    def asyncio_timeout(delay: Optional[float]) -> 'Timeout':
        """timeout context manager.
        Useful in cases when you want to apply timeout logic around block
        of code or in cases when asyncio.wait_for is not suitable. For example:
        >>> async with timeout(0.001):
        ...     async with aiohttp.get('https://github.com') as r:
        ...         await r.text()
        delay - value in seconds or None to disable timeout logic
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + delay if delay is not None else None
        return Timeout(deadline, loop)

    class _State(enum.Enum):
        INIT = 'INIT'
        ENTER = 'ENTER'
        TIMEOUT = 'TIMEOUT'
        EXIT = 'EXIT'

    class Timeout:
        __slots__ = ('_deadline', '_loop', '_state', '_timeout_handler')

        def __init__(self, deadline: Optional[float], loop: asyncio.AbstractEventLoop) -> None:
            self._loop = loop
            self._state = _State.INIT
            self._timeout_handler = None
            if deadline is None:
                self._deadline = None
            else:
                self.update(deadline)

        async def __aenter__(self) -> 'Timeout':
            self._do_enter()
            return self

        async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> Optional[bool]:
            self._do_exit(exc_type)
            return None

        @property
        def expired(self) -> bool:
            """Is timeout expired during execution?"""
            return self._state == _State.TIMEOUT

        @property
        def deadline(self) -> Optional[float]:
            return self._deadline

        def reject(self) -> None:
            """Reject scheduled timeout if any."""
            if self._state not in (_State.INIT, _State.ENTER):
                raise RuntimeError(f'invalid state {self._state.value}')
            self._reject()

        def _reject(self) -> None:
            if self._timeout_handler is not None:
                self._timeout_handler.cancel()
                self._timeout_handler = None

        def shift(self, delay: float) -> None:
            """Advance timeout on delay seconds.
            The delay can be negative.
            Raise RuntimeError if shift is called when deadline is not scheduled
            """
            deadline = self._deadline
            if deadline is None:
                raise RuntimeError('cannot shift timeout if deadline is not scheduled')
            self.update(deadline + delay)

        def update(self, deadline: float) -> None:
            """Set deadline to absolute value.
            deadline argument points on the time in the same clock system
            as loop.time().
            If new deadline is in the past the timeout is raised immediately.
            Please note: it is not POSIX time but a time with
            undefined starting base, e.g. the time of the system power on.
            """
            if self._state == _State.EXIT:
                raise RuntimeError('cannot reschedule after exit from context manager')
            if self._state == _State.TIMEOUT:
                raise RuntimeError('cannot reschedule expired timeout')
            if self._timeout_handler is not None:
                self._timeout_handler.cancel()
            self._deadline = deadline
            if self._state != _State.INIT:
                self._reschedule()

        def _reschedule(self) -> None:
            assert self._state == _State.ENTER
            deadline = self._deadline
            if deadline is None:
                return
            now = self._loop.time()
            if self._timeout_handler is not None:
                self._timeout_handler.cancel()
            task = asyncio.current_task()
            if deadline <= now:
                self._timeout_handler = self._loop.call_soon(self._on_timeout, task)
            else:
                self._timeout_handler = self._loop.call_at(deadline, self._on_timeout, task)

        def _do_enter(self) -> None:
            if self._state != _State.INIT:
                raise RuntimeError(f'invalid state {self._state.value}')
            self._state = _State.ENTER
            self._reschedule()

        def _do_exit(self, exc_type: Optional[Type[BaseException]]) -> None:
            if exc_type is asyncio.CancelledError and self._state == _State.TIMEOUT:
                self._timeout_handler = None
                raise asyncio.TimeoutError
            self._state = _State.EXIT
            self._reject()
            return None

        def _on_timeout(self, task: 'Optional[asyncio.Task[Any]]') -> None:
            if task:
                task.cancel()
            self._state = _State.TIMEOUT
            self._timeout_handler = None