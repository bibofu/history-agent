from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from history_agent.answering.models import AnswerResponse, Citation, QuestionRequest
from history_agent.answering.service import (
    _extractive_answer,
    _llm_answer,
    _quote_for_hit,
    _unsupported_leading_entity,
)
from history_agent.answering.validation import validate_grounded_answer
from history_agent.config import Settings
from history_agent.retrieval.models import SearchHit
from history_agent.web import app as web_module


def _citation(quote: str = "【1956年1月】参加会议并讨论科学规划。") -> Citation:
    return Citation(
        evidence_id="E1",
        document_id="zhou",
        document="周恩来年谱",
        pdf_page=688,
        section=["1956年"],
        quote=quote,
        source_type="chronology",
        verification_status="verified",
        extraction_methods=["pymupdf"],
    )


def test_extractive_answer_keeps_evidence_marker() -> None:
    answer = _extractive_answer("timeline", [_citation()])

    assert "[E1]" in answer
    assert "1956年" in answer


def test_extractive_answer_skips_bare_ellipsis_sentence() -> None:
    answer = _extractive_answer(
        "viewpoint",
        [_citation("……。要做系统的由历史到现状的调查研究。后续文字。")],
    )

    assert "- ……。[E1]" not in answer
    assert "要做系统" in answer


def test_quote_window_preserves_nearby_supporting_facts() -> None:
    hit = SearchHit(
        rank=1,
        chunk_id="observation",
        document_id="red-star",
        title="西行漫记",
        filename="red-star.pdf",
        source_type="contemporary_observation",
        verification_status="verified",
        pdf_page_start=118,
        pdf_page_end=118,
        section_path=[],
        text="毛泽东" + "性格质朴。" * 45 + "他是军事和政治战略家。" + "尾声。" * 30,
        year_mentions=[],
        people=["毛泽东"],
        extraction_methods=["ocr"],
        score=1.0,
        matched_terms=[],
    )

    quote = _quote_for_hit(hit, ["毛泽东"])

    assert "军事和政治战略家" in quote
    assert len(quote) <= 422


def test_question_request_rejects_unbounded_history() -> None:
    request = QuestionRequest(question="测试问题")

    assert request.top_k == 8
    assert request.history == []


def test_intersection_quote_keeps_the_complete_interaction_after_long_background() -> None:
    background = "长征途中，毛泽东提出意见。" + "会议回顾了此前的军事部署和行动方针。" * 30
    interaction = "周恩来、朱德等也是支持毛泽东的。"
    hit = SearchHit(
        rank=1,
        chunk_id="interaction",
        document_id="history",
        title="党史",
        filename="history.pdf",
        source_type="official_history",
        verification_status="verified",
        pdf_page_start=481,
        pdf_page_end=481,
        section_path=[],
        text=background + interaction + "随后讨论会议安排。" * 30,
        year_mentions=[1935],
        people=["毛泽东", "周恩来"],
        extraction_methods=["text_layer"],
        score=1.0,
        matched_terms=[],
    )
    quote = _quote_for_hit(hit, ["长征", "毛泽东", "周恩来"], query_people=["毛泽东", "周恩来"])
    assert interaction in quote
    assert quote.removeprefix("……").removesuffix("……") in hit.text
    assert len(quote) <= 424

    # A short list of attendees at the end must retain the sentence naming its event.
    event = "1935年1月15日至17日，中央政治局在遵义召开扩大会议。"
    roster = "出席会议的政治局委员有毛泽东、张闻天、周恩来、朱德。"
    tail_hit = hit.model_copy(update={"text": background + event + roster})
    tail_quote = _quote_for_hit(
        tail_hit, ["长征", "毛泽东", "周恩来"], query_people=["毛泽东", "周恩来"]
    )
    assert event in tail_quote
    assert roster in tail_quote


def test_unknown_leading_person_must_appear_in_evidence() -> None:
    assert _unsupported_leading_entity("爱因斯坦在1925年担任了什么党内职务？", []) == "爱因斯坦"


def test_compound_leading_people_are_checked_individually() -> None:
    hit = SearchHit(
        rank=1,
        chunk_id="intersection",
        document_id="history",
        title="测试史料",
        filename="history.pdf",
        source_type="history",
        verification_status="verified",
        pdf_page_start=1,
        pdf_page_end=1,
        section_path=[],
        text="1975年，毛泽东听取邓小平汇报。",
        year_mentions=[1975],
        people=["毛泽东", "邓小平"],
        extraction_methods=["text_layer"],
        score=1.0,
        matched_terms=[],
    )

    assert _unsupported_leading_entity("毛泽东和邓小平在1975年有哪些交集？", [hit]) is None


def test_api_health_and_question_contract(monkeypatch: Any) -> None:
    settings = Settings(project_root=Path.cwd(), data_dir=Path("test-data-that-does-not-exist"))

    expected = AnswerResponse(
        question="周恩来在1956年做了什么？",
        answer="参加有关会议。[E1]",
        evidence_status="supported",
        generator_mode="extractive",
        llm_status="disabled",
        retrieval_mode="hybrid_rrf",
        query_intent="timeline",
        citations=[_citation()],
    )

    def fake_answer(active_settings: Settings, request: QuestionRequest) -> AnswerResponse:
        assert active_settings is settings
        assert request.question == expected.question
        return expected

    monkeypatch.setattr(web_module, "answer_question", fake_answer)
    client = TestClient(web_module.create_app(settings))

    health = client.get("/api/health")
    index = client.get("/")
    javascript = client.get("/app.js")
    response = client.post("/api/questions", json={"question": expected.question})

    assert health.status_code == 200
    assert health.json()["indexes"] == {"keyword": False, "vector": False}
    assert index.headers["cache-control"] == "no-store"
    assert "app.js?v=stream-v2" in index.text
    assert javascript.headers["cache-control"] == "no-store"
    assert "/api/questions/stream" in javascript.text
    assert response.status_code == 200
    assert response.json()["citations"][0]["pdf_page"] == 688


def test_deepseek_v4_request_and_usage(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured["url"] = url
        captured.update(kwargs)
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "choices": [{"message": {"content": "参加有关会议。[E1]"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    settings = Settings(_env_file=None, llm_api_key="sk-test", llm_thinking=True)
    result = _llm_answer(
        settings=settings,
        request=QuestionRequest(question="周恩来在1956年做了什么？"),
        citations=[_citation()],
    )

    assert result.answer == "参加有关会议。[E1]"
    assert result.usage == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
    }
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    body = captured["json"]
    assert body["model"] == "deepseek-v4-pro"
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "high"
    assert "temperature" not in body
    assert "不能把人名共现推断成共同参与" in body["messages"][0]["content"]
    assert "检索年份范围只是召回线索" in body["messages"][0]["content"]


def test_deepseek_rejects_unknown_evidence_marker(monkeypatch: Any) -> None:
    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": "错误引用。[E99]"}}]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    settings = Settings(_env_file=None, llm_api_key="sk-test")
    result = _llm_answer(
        settings=settings,
        request=QuestionRequest(question="周恩来在1956年做了什么？"),
        citations=[_citation()],
    )

    assert result.answer is None
    assert result.error_code == "invalid_evidence_marker"


def test_validation_accepts_cited_core_fact_lines() -> None:
    result = validate_grounded_answer(
        "## 主要经历\n- 1956年1月，周恩来参加有关会议。[E1]\n\n资料范围有限。",
        [_citation()],
    )

    assert result.valid is True
    assert result.used_evidence_ids == ("E1",)


def test_validation_rejects_uncited_core_fact_line() -> None:
    result = validate_grounded_answer(
        "- 1956年1月，周恩来参加有关会议。[E1]\n- 随后主持科学规划工作。",
        [_citation()],
    )

    assert result.valid is False
    assert result.error_code == "uncited_core_claim"
    assert result.uncited_claims == ("随后主持科学规划工作。",)


def test_validation_accepts_wrapped_fact_with_citation_in_same_list_item() -> None:
    result = validate_grounded_answer(
        "- 1956年1月，周恩来参加有关会议。\n  随后主持科学规划工作。[E1]",
        [_citation()],
    )

    assert result.valid is True


def test_validation_accepts_multiline_paragraph_with_trailing_citation() -> None:
    result = validate_grounded_answer(
        "1956年1月，周恩来参加有关会议。\n随后主持科学规划工作。相关情况见年谱记载。[E1]",
        [_citation()],
    )

    assert result.valid is True


def test_validation_rejects_fabricated_pdf_page() -> None:
    result = validate_grounded_answer(
        "《周恩来年谱》PDF第999页记载周恩来参加会议。[E1]",
        [_citation()],
    )

    assert result.valid is False
    assert result.error_code == "citation_metadata_mismatch"


def test_validation_accepts_matching_document_and_pdf_page() -> None:
    result = validate_grounded_answer(
        "《周恩来年谱》PDF第688页记载周恩来参加会议。[E1]",
        [_citation()],
    )

    assert result.valid is True


def test_validation_rejects_mismatched_document_name() -> None:
    result = validate_grounded_answer(
        "《林彪年谱》PDF第688页记载周恩来参加会议。[E1]",
        [_citation()],
    )

    assert result.valid is False
    assert result.error_code == "citation_metadata_mismatch"


def test_deepseek_falls_back_when_core_fact_has_no_citation(monkeypatch: Any) -> None:
    calls = 0

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "1956年1月，周恩来参加有关会议。[E1]\n\n随后主持科学规划工作。"
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    settings = Settings(_env_file=None, llm_api_key="sk-test")
    result = _llm_answer(
        settings=settings,
        request=QuestionRequest(question="周恩来在1956年做了什么？"),
        citations=[_citation()],
    )

    assert result.answer is None
    assert result.error_code == "citation_repair_uncited_core_claim"
    assert result.uncited_claims == ("随后主持科学规划工作。",)
    assert calls == 2


def test_deepseek_repairs_missing_core_fact_citation_once(monkeypatch: Any) -> None:
    responses = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "1956年1月，周恩来参加有关会议。[E1]\n\n随后主持科学规划工作。"
                            )
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "1956年1月，周恩来参加有关会议。[E1]\n\n随后主持科学规划工作。[E1]"
                            )
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 3,
                    "total_tokens": 18,
                },
            },
        ]
    )
    request_bodies: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        request_bodies.append(kwargs["json"])
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json=next(responses),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = _llm_answer(
        settings=Settings(_env_file=None, llm_api_key="sk-test"),
        request=QuestionRequest(question="周恩来在1956年做了什么？"),
        citations=[_citation()],
    )

    assert result.answer == ("1956年1月，周恩来参加有关会议。[E1]\n\n随后主持科学规划工作。[E1]")
    assert result.usage == {
        "prompt_tokens": 25,
        "completion_tokens": 5,
        "total_tokens": 30,
    }
    assert len(request_bodies) == 2
    repair_messages = request_bodies[1]["messages"]
    assert repair_messages[-2]["role"] == "assistant"
    assert "随后主持科学规划工作" in repair_messages[-1]["content"]
    assert "没有证据支持的事实必须删除" in repair_messages[-1]["content"]


@pytest.mark.parametrize(
    "answer",
    [
        "> 周恩来参加会议。\n> [E1]",
        "周恩来参加会议。\n\n[E1]",
        "- 周恩来参加会议。\n\n  [E1]",
        "- **共同参加会议**\n  - 周恩来参加会议。[E1]",
        "### 出席人员\n\n周恩来参加会议。[E1]",
        "### 三、参与者个例\n\n周恩来参加会议。[E1]",
        "### 筹备委员会成立\n\n周恩来参加会议。[E1]",
        "### 群众参与细节\n\n周恩来参加会议。[E1]",
        "| 参加人员 | 情况 |\n| --- | --- |\n| 周恩来 | 参加会议。[E1] |",
        "**周恩来参加会议。**\n**[E1]**",
        "周恩来参加会议。[E1]\n\n现有资料不足以确认两人是否共同参与了其他活动。",
        "周恩来参加会议。[E1]\n\n无法确认两人是否共同主持了其他会议。",
        (
            "周恩来参加会议。[E1]\n\n关于开国大典的阅兵细节、参加领导人名单、天安门广场"
            "布置等具体情况，证据包中无更多材料，无法进一步说明。"
        ),
        (
            "周恩来参加会议。[E1]\n\n证据中出现的周恩来在1945年、1946年涉及“国民大会”"
            "的内容，均与开国大典无直接关联，不予采入。"
        ),
    ],
)
def test_validation_accepts_markdown_citation_layouts_and_evidence_limits(answer: str) -> None:
    assert validate_grounded_answer(answer, [_citation()]).valid


@pytest.mark.parametrize(
    "uncited",
    [
        "现有资料不足以确认其他活动，但两人随后共同主持会议。",
        "现有资料不足以确认其他活动。两人随后共同主持会议。",
        "现有资料不足以确认其他活动，随后共同主持会议。",
        "现有资料记载两人共同主持会议。",
        "## 1956年周恩来主持会议",
        "## 1956年筹备委员会成立",
        "**周恩来主持会议。**",
        "- **周恩来主持会议**\n  - 另见材料。[E1]",
        "周恩来主持会议：",
        "<div>周恩来主持会议。</div>",
        "| 日期 | 活动 |\n| --- | --- |\n| 1956年 | 主持会议 |",
    ],
)
def test_validation_still_rejects_uncited_facts_in_any_format(uncited: str) -> None:
    result = validate_grounded_answer("参加有关会议。[E1]\n\n" + uncited, [_citation()])
    assert result.error_code == "uncited_core_claim"
    assert result.uncited_claims


@pytest.mark.parametrize(
    "answer",
    [
        "周恩来参加会议。\n\n另一件事。[E1]",
        "周恩来参加会议。\n\n## 出处\n\n[E1]",
        "周恩来参加会议。\n\n---\n\n[E1]",
        "- 周恩来参加会议。\n- [E1]",
        "> 周恩来参加会议。\n\n[E1]",
        "| 日期 | 活动 |\n| --- | --- |\n| 1956年 | 主持会议 |\n| 1957年 | 参加会议。[E1] |",
    ],
)
def test_citations_cannot_cover_separate_blocks(answer: str) -> None:
    result = validate_grounded_answer(answer, [_citation()])
    assert result.error_code == "uncited_core_claim"


def test_citation_on_separate_line_does_not_allow_fabricated_page() -> None:
    result = validate_grounded_answer(
        "> 《周恩来年谱》PDF第999页记载周恩来参加会议。\n> [E1]", [_citation()]
    )
    assert result.error_code == "citation_metadata_mismatch"


@pytest.mark.parametrize("marker", [r"\[E99\]", "&#91;E99&#93;"])
def test_decoded_markdown_marker_is_validated_before_metadata_lookup(marker: str) -> None:
    result = validate_grounded_answer(
        f"参加会议。[E1]\n\n《周恩来年谱》PDF第688页记载参加会议。{marker}", [_citation()]
    )
    assert result.error_code == "invalid_evidence_marker"
