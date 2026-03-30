import datetime as dt
import unittest
from unittest.mock import Mock, patch

import app
import requests


def make_feed(count: int) -> str:
    entries: list[str] = []
    for idx in range(count):
        entries.append(
            f"""
  <entry>
    <id>http://arxiv.org/abs/1234.{idx:05d}</id>
    <updated>2026-03-30T00:00:00Z</updated>
    <published>2026-03-29T00:00:00Z</published>
    <title>Paper {idx}</title>
    <summary>Summary {idx}</summary>
    <author><name>Author {idx}</name></author>
    <category term="cs.AI" />
    <link href="https://arxiv.org/abs/1234.{idx:05d}" rel="alternate" type="text/html" />
  </entry>
"""
        )
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<feed xmlns=\"http://www.w3.org/2005/Atom\">\n"
        f"{''.join(entries)}"
        "</feed>\n"
    )


def make_response(status_code: int, text: str = "", headers: dict[str, str] | None = None) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = status_code
    response.text = text
    response.headers = headers or {}
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(response=response)
    else:
        response.raise_for_status.return_value = None
    return response


class ArxivRateLimitTests(unittest.TestCase):
    def setUp(self) -> None:
        app._last_arxiv_request_monotonic = 0.0

    def test_fetch_arxiv_papers_retries_http_429(self) -> None:
        rate_limited = make_response(429, headers={"Retry-After": "7"})
        success = make_response(200, text=make_feed(1))

        start = dt.datetime(2026, 3, 29, tzinfo=dt.timezone.utc)
        end = dt.datetime(2026, 3, 30, tzinfo=dt.timezone.utc)

        with patch("app.requests.get", side_effect=[rate_limited, success]) as mock_get:
            with patch("app.time.sleep") as mock_sleep:
                with patch("app.time.monotonic", side_effect=[100.0, 100.0, 100.0]):
                    papers = app.fetch_arxiv_papers(start, end, 1, dbg=True)

        self.assertEqual(len(papers), 1)
        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(mock_sleep.call_args_list[0].args[0], 7.0)
        self.assertGreaterEqual(mock_sleep.call_args_list[1].args[0], app.ARXIV_MIN_REQUEST_GAP_SECONDS)

    def test_fetch_arxiv_papers_spaces_paginated_requests(self) -> None:
        first_page = make_response(200, text=make_feed(app.ARXIV_PAGE_SIZE))
        second_page = make_response(200, text=make_feed(1))

        start = dt.datetime(2026, 3, 29, tzinfo=dt.timezone.utc)
        end = dt.datetime(2026, 3, 30, tzinfo=dt.timezone.utc)

        with patch("app.requests.get", side_effect=[first_page, second_page]) as mock_get:
            with patch("app.time.sleep") as mock_sleep:
                with patch("app.time.monotonic", side_effect=[100.0, 101.0, 101.0, 101.0]):
                    papers = app.fetch_arxiv_papers(start, end, app.ARXIV_PAGE_SIZE + 1)

        self.assertEqual(len(papers), app.ARXIV_PAGE_SIZE + 1)
        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(len(mock_sleep.call_args_list), 1)
        self.assertAlmostEqual(
            mock_sleep.call_args_list[0].args[0],
            app.ARXIV_MIN_REQUEST_GAP_SECONDS - 1.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
