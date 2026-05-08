from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx


class ApiClientError(RuntimeError):
    """Raised when OpenCode API calls fail."""


@dataclass(slots=True)
class OpencodeApiClient:
    base_url: str = "http://127.0.0.1:4096"
    username: str | None = None
    password: str | None = None
    timeout: float = 10.0
    retries: int = 2
    _http: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        auth = None
        if self.username is not None and self.password is not None:
            auth = (self.username, self.password)
        self._http = httpx.Client(base_url=self.base_url, timeout=self.timeout, auth=auth)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "OpencodeApiClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", path, params=params)

    def post(self, path: str, json: dict[str, Any] | None = None) -> Any:
        return self._request("POST", path, json=json)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        if not path.startswith("/"):
            path = f"/{path}"

        last_exc: Exception | None = None
        attempts = max(1, self.retries + 1)
        for _ in range(attempts):
            try:
                response = self._http.request(method, path, **kwargs)
                response.raise_for_status()
                payload = response.json()
                return self._unwrap_json(payload)
            except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as exc:
                last_exc = exc
        raise ApiClientError(f"OpenCode API request failed: {method} {path}") from last_exc

    @staticmethod
    def _unwrap_json(payload: Any) -> Any:
        if isinstance(payload, dict):
            if "data" in payload:
                return payload["data"]
            if "result" in payload:
                return payload["result"]
        return payload
