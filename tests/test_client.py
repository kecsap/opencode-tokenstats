from __future__ import annotations

import httpx
import pytest

from opencode_tokenstats.client import ApiClientError, OpencodeApiClient


def _mock_transport(status: int, payload: object) -> httpx.MockTransport:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


def test_unwraps_data_field() -> None:
    client = OpencodeApiClient()
    client._http.close()
    client._http = httpx.Client(
        base_url="http://127.0.0.1:4096",
        transport=_mock_transport(200, {"data": {"ok": True}}),
    )
    assert client.get("/x") == {"ok": True}


def test_unwraps_result_field() -> None:
    client = OpencodeApiClient()
    client._http.close()
    client._http = httpx.Client(
        base_url="http://127.0.0.1:4096",
        transport=_mock_transport(200, {"result": [1, 2]}),
    )
    assert client.get("/x") == [1, 2]


def test_raises_after_retries() -> None:
    attempts = {"count": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        return httpx.Response(503, json={"error": "down"})

    client = OpencodeApiClient(retries=2)
    client._http.close()
    client._http = httpx.Client(
        base_url="http://127.0.0.1:4096",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ApiClientError):
        client.get("/x")
    assert attempts["count"] == 3
