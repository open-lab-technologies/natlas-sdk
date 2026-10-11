"""Exceptions raised by the N-ATLAS clients. All derive from `NAtlasError`."""

from __future__ import annotations

from typing import Any


class NAtlasError(Exception):
    """Base class. `status_code` is set when the server answered with an HTTP error."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class APIConnectionError(NAtlasError):
    """The endpoint could not be reached, or the connection dropped."""


class APITimeoutError(APIConnectionError):
    """The request took longer than the client's timeout."""


class APIStatusError(NAtlasError):
    """The endpoint returned a non-2xx response. `body` holds the parsed error, if any."""

    def __init__(self, message: str, status_code: int, body: Any = None):
        super().__init__(message, status_code)
        self.body = body


class BadRequestError(APIStatusError):
    """400 or 422: the request was malformed, or the prompt does not fit the context window."""


class AuthenticationError(APIStatusError):
    """401 or 403: missing or wrong API key."""


class NotFoundError(APIStatusError):
    """404: wrong base URL, or the endpoint does not offer this feature."""


class RateLimitError(APIStatusError):
    """429: retried automatically before this is raised."""


class InternalServerError(APIStatusError):
    """5xx: retried automatically before this is raised."""


def status_error(status: int, message: str, body: Any = None) -> APIStatusError:
    if status in (400, 422):
        cls: type[APIStatusError] = BadRequestError
    elif status in (401, 403):
        cls = AuthenticationError
    elif status == 404:
        cls = NotFoundError
    elif status == 429:
        cls = RateLimitError
    elif status >= 500:
        cls = InternalServerError
    else:
        cls = APIStatusError
    return cls(message, status, body)


class StructuredOutputError(NAtlasError):
    """The reply did not match the requested `response_format`. `text` is the raw reply.

    Endpoints that enforce the schema while generating, such as `pynatlas serve`,
    vLLM and llama.cpp's server, should not produce this; others may.
    """

    def __init__(self, message: str, text: str):
        super().__init__(message)
        self.text = text


class ToolLoopError(NAtlasError):
    """`run_tools` reached `max_rounds` with the model still calling tools."""
