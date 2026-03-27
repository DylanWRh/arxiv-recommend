import datetime as dt
import json
import unittest
from unittest.mock import Mock, patch

import requests

import app


class LlmRetryTests(unittest.TestCase):
    def _paper(self) -> app.Paper:
        now = dt.datetime(2026, 3, 27, tzinfo=dt.timezone.utc)
        return app.Paper(
            paper_id="1234.5678",
            title="A Paper",
            summary="Summary",
            authors=["A"],
            categories=["cs.AI"],
            published=now,
            updated=now,
            link="https://arxiv.org/abs/1234.5678",
        )

    @patch.dict(
        "os.environ",
        {
            "LLM_API_RETRY_ATTEMPTS": "",
            "LLM_API_RETRY_SLEEP_SECONDS": "",
        },
        clear=False,
    )
    def test_llm_retry_config_defaults(self) -> None:
        attempts, sleep_seconds = app.llm_retry_config()
        self.assertEqual(attempts, 5)
        self.assertEqual(sleep_seconds, 10)

    @patch("app.time.sleep")
    @patch("app.requests.post")
    @patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "k",
            "LLM_API_RETRY_ATTEMPTS": "3",
            "LLM_API_RETRY_SLEEP_SECONDS": "1",
        },
        clear=False,
    )
    def test_recommend_with_llm_batch_retries_then_succeeds(self, mock_post: Mock, mock_sleep: Mock) -> None:
        success_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "recommended": [
                                    {
                                        "paper_id": "1234.5678",
                                        "score": 88,
                                        "matched_interests": ["ai"],
                                        "reason": "match",
                                        "summary": "s",
                                        "title_zh": "标题",
                                        "abstract_zh": "摘要",
                                    }
                                ]
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

        failed_response = Mock()
        failed_response.raise_for_status.side_effect = requests.HTTPError("503")

        success_response = Mock()
        success_response.raise_for_status.return_value = None
        success_response.json.return_value = success_payload

        mock_post.side_effect = [failed_response, success_response]

        results = app.recommend_with_llm_batch(
            papers=[self._paper()],
            research_profile="profile",
            llm_model="gpt-4o-mini",
            llm_timeout=10,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].paper.paper_id, "1234.5678")
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once_with(1.0)

    @patch("app.time.sleep")
    @patch("app.requests.post")
    @patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "k",
            "LLM_API_RETRY_ATTEMPTS": "2",
            "LLM_API_RETRY_SLEEP_SECONDS": "1",
        },
        clear=False,
    )
    def test_recommend_with_llm_batch_raises_after_retries(self, mock_post: Mock, mock_sleep: Mock) -> None:
        failed_response = Mock()
        failed_response.raise_for_status.side_effect = requests.HTTPError("503")
        mock_post.side_effect = [failed_response, failed_response]

        with self.assertRaises(requests.HTTPError):
            app.recommend_with_llm_batch(
                papers=[self._paper()],
                research_profile="profile",
                llm_model="gpt-4o-mini",
                llm_timeout=10,
            )

        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once_with(1.0)

    @patch("app.time.sleep")
    @patch("app.requests.post")
    @patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "k",
            "LLM_API_RETRY_ATTEMPTS": "3",
            "LLM_API_RETRY_SLEEP_SECONDS": "1",
        },
        clear=False,
    )
    def test_normalize_times_with_llm_retries_then_succeeds(self, mock_post: Mock, mock_sleep: Mock) -> None:
        bad_json_response = Mock()
        bad_json_response.raise_for_status.return_value = None
        bad_json_response.json.return_value = {"choices": [{"message": {"content": "{bad json"}}]}

        success_response = Mock()
        success_response.raise_for_status.return_value = None
        success_response.json.return_value = {
            "choices": [{"message": {"content": '{"start":"2026-03-01T00:00:00+00:00","end":null}'}}]
        }

        mock_post.side_effect = [bad_json_response, success_response]

        start, end = app.normalize_times_with_llm(
            start_raw="2026-03-01",
            end_raw=None,
            tz_name="UTC",
            reference_now=dt.datetime(2026, 3, 27, tzinfo=dt.timezone.utc),
            llm_model="gpt-4o-mini",
            llm_timeout=10,
        )

        self.assertEqual(start, "2026-03-01T00:00:00+00:00")
        self.assertIsNone(end)
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once_with(1.0)


if __name__ == "__main__":
    unittest.main()
