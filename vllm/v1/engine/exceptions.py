class EngineGenerateError(Exception):
    """Raised when a AsyncLLM.generate() fails. Recoverable."""
    pass

class EngineDeadError(Exception):
    """Raised when the EngineCore dies. Unrecoverable."""

    def __init__(self, *args, suppress_context: bool=False, **kwargs):
        ENGINE_DEAD_MESSAGE = 'EngineCore encountered an issue. See stack trace (above) for the root cause.'
        super().__init__(ENGINE_DEAD_MESSAGE, *args, **kwargs)
        self.__suppress_context__ = suppress_context