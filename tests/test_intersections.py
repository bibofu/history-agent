from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from history_agent.cli import app
from history_agent.config import Settings
from history_agent.errors import ResearchDataError
from history_agent.evaluation.intersections import (
    build_intersection_review_packet,
    evaluate_intersections,
    finalize_intersection_review_packet,
)
from history_agent.research.intersections import (
    _match_adjacent_attendance,
    _match_grouped_attendance,
    get_person_intersections,
)
from history_agent.web.app import create_app
from test_timeline import _prepare_database, _prepare_timeline, _save_event
from typer.testing import CliRunner


@pytest.mark.parametrize(
    "description,expected",
    [
        ("毛泽东和周恩来出席会议并听取工作汇报。", True),
        ("毛泽东、林彪和周恩来在西柏坡共同出席会议。", True),
        ("同日 和毛泽东在西柏坡会见有关方面的代表。", True),
        ("出席毛泽东主持的中共中央政治局工作会议。", True),
        ("林彪参加活动，和毛泽东共同出席关于工作的会议。", False),
        ("周恩来、毛泽东共同签署关于工作的文件。", True),
        ("毛泽东会见周恩来，商谈有关工作安排。", True),
        ("10月1日 下午和毛泽东共同出席会议并听取汇报。", True),
        ("周恩来出席会议。毛泽东在另一地点讲话。", False),
        ("周恩来出席会议，毛泽东在另一地点讲话。", False),
        ("周恩来谈到毛泽东的讲话并出席会议。", False),
        ("周恩来致电毛泽东，报告关于会议的准备工作。", False),
        ("周恩来说：“毛泽东和周恩来出席会议。”", False),
        ("周恩来回忆：毛泽东和周恩来出席会议。", False),
        ("毛泽东和周恩来准备共同出席有关会议。", False),
        ("毛泽东和周恩来没有出席有关工作会议。", False),
    ],
    ids=[f"case-{index}" for index in range(16)],
)
def test_joint_action_not_name_cooccurrence(
    work_path: Path, description: str, expected: bool
) -> None:
    database = _prepare_database(work_path)
    _save_event(
        database,
        event_id="test",
        document_id="zhou_enlai_chronology_1949_1976",
        date_value="1956-01-01",
        description=description,
        event_type="meeting",
        review_status="confirmed",
        page=20,
        include_lin=False,
    )
    result = get_person_intersections(
        database, person_id="mao_zedong", other_person_id="zhou_enlai"
    )
    assert result.co_mention_total == 1
    assert result.total == int(expected)
    if expected:
        item = result.events[0]
        assert item.verification_status == "needs_review"
        assert item.joint_evidence[0].supporting_text in description
        assert set(item.joint_evidence[0].roles) == {"mao_zedong", "zhou_enlai"}
        reverse = get_person_intersections(
            database, person_id="zhou_enlai", other_person_id="mao_zedong"
        )
        assert reverse.events == result.events


def test_intersections_dedup_pagination_and_no_recipient_promotion(work_path: Path) -> None:
    database, canonical_id = _prepare_timeline(work_path)
    for index in range(3):
        _save_event(
            database,
            event_id=f"joint{index}",
            document_id="zhou_enlai_chronology_1949_1976",
            date_value=f"1943-03-0{index + 1}",
            description="周恩来和林彪共同出席会议并听取工作汇报。",
            event_type="meeting",
            review_status="needs_review",
            page=21,
            include_lin=True,
        )
    result = get_person_intersections(
        database, person_id="zhou_enlai", other_person_id="lin_biao", limit=1
    )
    assert result.total == 4
    assert result.events[0].event.event_id == canonical_id
    assert len(result.events[0].joint_evidence) == 2
    assert result.has_more
    page = get_person_intersections(
        database, person_id="zhou_enlai", other_person_id="lin_biao", offset=1, limit=2
    )
    assert [e.event.event_id for e in page.events] == ["joint0", "joint1"]
    assert page.has_more
    filtered = get_person_intersections(
        database, person_id="zhou_enlai", other_person_id="lin_biao", event_types=["meeting"]
    )
    assert filtered.total == 3
    no_recipient = get_person_intersections(
        database, person_id="mao_zedong", other_person_id="lin_biao"
    )
    assert no_recipient.total == 0
    with pytest.raises(ResearchDataError, match="different people"):
        get_person_intersections(database, person_id="lin_biao", other_person_id="lin_biao")
    with pytest.raises(ResearchDataError, match="unknown person_id"):
        get_person_intersections(database, person_id="lin_biao", other_person_id="missing")


def test_intersection_api_and_cli(work_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database, _ = _prepare_timeline(work_path)
    settings = Settings(
        _env_file=None,
        project_root=work_path,
        data_dir=work_path / "data",
        database_path=database.path,
    )
    client = TestClient(create_app(settings))
    url = "/api/people/zhou_enlai/intersections/lin_biao"
    response = client.get(url, params={"start_year": 1943, "end_year": 1943})
    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert client.get(url, params={"start_year": 1900}).status_code == 422
    assert client.get(url, params={"limit": 0}).status_code == 422
    assert client.get(url, params={"start_year": 1950, "end_year": 1940}).status_code == 400
    assert client.get("/api/people/zhou_enlai/intersections/missing").status_code == 404
    monkeypatch.setattr("history_agent.cli.get_settings", lambda: settings)
    runner = CliRunner()
    result = runner.invoke(
        app, ["research", "intersections", "周恩来", "林彪", "--year", "1943", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["total"] == 1
    assert (
        runner.invoke(
            app, ["research", "intersections", "周恩来", "林彪", "--year", "1900"]
        ).exit_code
        != 0
    )


def test_canonical_members_cannot_manufacture_joint_evidence(work_path: Path) -> None:
    database, _ = _prepare_timeline(work_path)
    with database.connect() as connection:
        connection.execute(
            "UPDATE evidence_records SET quote='周恩来出席会议并听取有关工作的详细汇报。' "
            "WHERE evidence_id='evidence_event_zhou_message'"
        )
        connection.execute(
            "UPDATE evidence_records SET quote='林彪出席会议并听取有关工作的详细汇报。' "
            "WHERE evidence_id='evidence_event_lin_message'"
        )
    result = get_person_intersections(database, person_id="zhou_enlai", other_person_id="lin_biao")
    assert result.co_mention_total == 1
    assert result.total == 0


def test_joint_pagination_crosses_candidate_batch_boundary(work_path: Path) -> None:
    database = _prepare_database(work_path)
    for index in range(202):
        _save_event(
            database,
            event_id=f"event{index:03}",
            document_id="zhou_enlai_chronology_1949_1976",
            date_value="1956-01-01",
            description=(
                "周恩来和毛泽东出席会议并听取详细汇报。"
                if index >= 199
                else "周恩来出席会议。毛泽东在其他地方进行工作。"
            ),
            event_type="meeting",
            review_status="needs_review",
            page=20,
            include_lin=False,
        )
    result = get_person_intersections(
        database, person_id="zhou_enlai", other_person_id="mao_zedong", offset=1, limit=1
    )
    assert result.total == 3
    assert result.co_mention_total == 202
    assert result.events[0].event.event_id == "event200"
    assert result.has_more


def test_recognized_chronology_retains_owner_when_explicit(work_path: Path) -> None:
    database = _prepare_database(work_path)
    _save_event(
        database,
        event_id="explicit_owner",
        document_id="zhou_enlai_chronology_1949_1976",
        date_value="1956-01-01",
        description="和毛泽东出席会议。周恩来随后作了报告。",
        event_type="meeting",
        review_status="needs_review",
        page=20,
        include_lin=False,
    )
    with database.connect() as connection:
        connection.execute("UPDATE event_participants SET mention_source='explicit'")
    assert (
        get_person_intersections(
            database, person_id="mao_zedong", other_person_id="zhou_enlai"
        ).total
        == 0
    )
    with database.connect() as connection:
        connection.execute("UPDATE historical_events SET extractor_version='chronology-rules-v1'")
    result = get_person_intersections(
        database, person_id="mao_zedong", other_person_id="zhou_enlai"
    )
    assert result.total == 1
    assert result.events[0].joint_evidence[0].subject_basis == "chronology_subject"


def test_intersection_evaluation_checks_sources_and_metrics(work_path: Path) -> None:
    database, _ = _prepare_timeline(work_path)
    cases_path = work_path / "cases.json"
    payload = {
        "cases": [
            {
                "id": "recipient",
                "source_event_id": "event_zhou_message",
                "document_id": "zhou_enlai_chronology_1949_1976",
                "pdf_page": 20,
                "year": 1943,
                "expected": False,
                "reason": "recipient is not co-sender",
            }
        ]
    }
    cases_path.write_text(json.dumps(payload), encoding="utf-8")
    result = evaluate_intersections(database, cases_path)
    assert result["true_negative"] == 1
    assert result["recall"] is None
    payload["cases"][0]["pdf_page"] = 99
    cases_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ResearchDataError, match="gold source/page missing"):
        evaluate_intersections(database, cases_path)


def test_blind_review_packet_excludes_predictions_and_finalizes(work_path: Path) -> None:
    database = _prepare_database(work_path)
    _save_event(
        database,
        event_id="blind_positive",
        document_id="zhou_enlai_chronology_1949_1976",
        date_value="1943-03-01",
        description="周恩来和林彪共同出席会议并听取汇报。",
        event_type="meeting",
        review_status="needs_review",
        page=22,
        include_lin=True,
    )
    _save_event(
        database,
        event_id="blind_negative",
        document_id="zhou_enlai_chronology_1949_1976",
        date_value="1943-03-02",
        description="周恩来致电毛泽东，林彪在重庆开展工作。",
        event_type="correspondence",
        review_status="needs_review",
        page=23,
        include_lin=True,
    )
    development_path = work_path / "development.json"
    development_path.write_text('{"cases": []}', encoding="utf-8")
    packet = build_intersection_review_packet(
        database,
        development_path,
        seed="fixed-seed",
        pair_limit=1,
        per_stratum=1,
        start_year=1943,
        end_year=1943,
    )
    assert packet == build_intersection_review_packet(
        database,
        development_path,
        seed="fixed-seed",
        pair_limit=1,
        per_stratum=1,
        start_year=1943,
        end_year=1943,
    )
    assert len(packet.cases) == 2
    assert {item.source_event_id for item in packet.cases} == {
        "blind_positive",
        "blind_negative",
    }
    serialized = packet.model_dump_json()
    assert "predicted" not in serialized
    assert all(item.annotation.expected is None for item in packet.cases)
    packet_path = work_path / "review.json"
    packet_path.write_text(packet.model_dump_json(indent=2), encoding="utf-8")
    with pytest.raises(ResearchDataError, match="review label missing"):
        finalize_intersection_review_packet(packet_path)
    original_name = packet.cases[0].event_name
    packet.cases[0].event_name = "changed"
    packet_path.write_text(packet.model_dump_json(indent=2), encoding="utf-8")
    with pytest.raises(ResearchDataError, match="review case content changed"):
        finalize_intersection_review_packet(packet_path)
    packet.cases[0].event_name = original_name
    for item in packet.cases:
        item.annotation.expected = item.source_event_id == "blind_positive"
        item.annotation.reason = "完整查看来源页后作出的独立判断。"
        item.annotation.reviewed_by = "reviewer-a"
        item.annotation.reviewed_at = "2026-09-06"
    packet.cases[0].annotation.reviewed_at = "09/06/2026"
    packet_path.write_text(packet.model_dump_json(indent=2), encoding="utf-8")
    with pytest.raises(ResearchDataError, match="review date must be ISO"):
        finalize_intersection_review_packet(packet_path)
    packet.cases[0].annotation.reviewed_at = "2026-09-06"
    packet_path.write_text(packet.model_dump_json(indent=2), encoding="utf-8")
    finalized = finalize_intersection_review_packet(packet_path)
    assert finalized["reviewers"] == ["reviewer-a"]
    assert {item["cohort"] for item in finalized["cases"]} == {"independent"}
    final_path = work_path / "independent.json"
    final_path.write_text(json.dumps(finalized), encoding="utf-8")
    result = evaluate_intersections(database, final_path)
    assert result["true_positive"] == 1
    assert result["true_negative"] == 1


@pytest.mark.parametrize(
    "text,subject,expected",
    [
        ("毛泽东在北京会见外国代表。参加会见的有周恩来、彭真等。", None, True),
        (
            "3月14日下午六时，在北京会见并宴请外国代表，同他们交谈。参加会见和宴请的有周恩来、彭真。",
            "毛泽东",
            True,
        ),
        (
            "2月14日下午，主持召开中央政治局扩大会议，讨论重要问题。出席会议的有：刘少奇、周恩来、彭真。",
            "毛泽东",
            True,
        ),
        ("周恩来主持会议，讨论工作。出席会议的有毛泽东、陈云。", None, True),
        ("毛泽东接见代表团成员。参加接见的还有周恩来、彭真等。", None, True),
        ("毛泽东准备会见外国代表。参加会见的有周恩来。", None, False),
        ("毛泽东没有会见外国代表。参加会见的有周恩来。", None, False),
        ("毛泽东委托林彪会见外国代表。参加会见的有周恩来。", None, False),
        ("林彪会见外国代表。参加会见的有周恩来。", "毛泽东", False),
        ("毛泽东会见外国代表。次日参加会见的有周恩来。", None, False),
        ("毛泽东会见外国代表。参加另一场会见的有周恩来。", None, False),
        ("毛泽东会见外国代表。参加会议的有周恩来。", None, False),
        ("毛泽东会见外国代表。参加会见的有周恩来的秘书。", None, False),
        ("毛泽东会见外国代表。参加会见的有林彪。", None, False),
        ("毛泽东会见外国代表。名单包括周恩来。", None, False),
        ("毛泽东会见外国代表。另召开会议。参加会见的有周恩来。", None, False),
        ("毛泽东说：会见外国代表。参加会见的有周恩来。", None, False),
        ("“毛泽东会见外国代表。参加会见的有周恩来。”", None, False),
        ("毛泽东会见外国代表。[注]参加会见的有周恩来。", None, False),
        ("毛泽东会见外国代表。参加会见的有周恩来，但未到场。", None, False),
        ("毛泽东会见外国代表。参加会见的有周恩来", None, False),
        ("会见外国代表。参加会见的有周恩来。", None, False),
    ],
    ids=[f"adjacent-{i}" for i in range(22)],
)
def test_adjacent_attendance_guards(text: str, subject: str | None, expected: bool) -> None:
    result = _match_adjacent_attendance(text, "毛泽东", "周恩来", subject)
    assert (result is not None) is expected
    if result:
        span, action, roles, basis = result
        assert span in text
        assert "。" in span[:-1]
        assert len(span) <= 420
        assert set(roles) == {"毛泽东", "周恩来"}
        assert _match_adjacent_attendance(text, "周恩来", "毛泽东", subject) == result
        if action == "主持":
            assert set(roles.values()) == {"主持者", "参会者"}
        else:
            assert "主持者" not in roles.values()


def test_adjacent_evidence_is_source_local_single_page_and_chat_complete(work_path: Path) -> None:
    database = _prepare_database(work_path)
    text = "在北京会见外国代表。参加会见的有毛泽东、彭真等。"
    _save_event(
        database,
        event_id="adjacent",
        document_id="zhou_enlai_chronology_1949_1976",
        date_value="1956-01-01",
        description=text,
        event_type="meeting",
        review_status="needs_review",
        page=20,
        include_lin=False,
    )
    result = get_person_intersections(
        database, person_id="mao_zedong", other_person_id="zhou_enlai"
    )
    proof = result.events[0].joint_evidence[0]
    assert proof.match_method == "adjacent_attendance"
    assert proof.supporting_text == text
    settings = Settings(
        _env_file=None,
        project_root=work_path,
        data_dir=work_path / "data",
        database_path=database.path,
    )
    response = TestClient(create_app(settings)).post(
        "/api/questions", json={"question": "毛泽东和周恩来在1956年有哪些交集"}
    )
    assert response.status_code == 200
    assert response.json()["citations"][0]["quote"] == text
    assert response.json()["citations"][0]["pdf_page"] == 20
    assert response.json()["evidence_status"] == "partial"
    with database.connect() as connection:
        connection.execute("UPDATE evidence_records SET pdf_page_end=21")
    assert (
        get_person_intersections(
            database, person_id="mao_zedong", other_person_id="zhou_enlai"
        ).total
        == 0
    )


def test_adjacent_does_not_build_oversized_proof() -> None:
    assert (
        _match_adjacent_attendance(
            "毛泽东会见外国代表，" + "进行工作交谈，" * 60 + "讨论事宜。参加会见的有周恩来。",
            "毛泽东",
            "周恩来",
            None,
        )
        is None
    )


def test_adjacent_roster_supports_two_non_subject_attendees() -> None:
    result = _match_adjacent_attendance(
        "主持召开中央政治局扩大会议。出席会议的有刘少奇、周恩来、陈云。",
        "刘少奇",
        "周恩来",
        "毛泽东",
    )
    assert result is not None
    assert result[2] == {"刘少奇": "参会者", "周恩来": "参会者"}


@pytest.mark.parametrize(
    "first,second,subject,expected",
    [
        ("毛泽东", "周恩来", "毛泽东", True),
        ("毛泽东", "邓小平", "毛泽东", True),
        ("周恩来", "邓小平", "毛泽东", True),
        ("毛泽东", "苏斯洛夫", "毛泽东", False),
        ("周恩来", "苏斯洛夫", "毛泽东", False),
    ],
)
def test_grouped_attendance_uses_only_domestic_roster(
    first: str,
    second: str,
    subject: str,
    expected: bool,
) -> None:
    text = (
        "10月1日上午，去天安门参加中华人民共和国成立十周年庆祝大会前，"
        "在中南海颐年堂会见赫鲁晓夫。"
        "参加会见的，中方有刘少奇、朱德、周恩来、邓小平，苏方有苏斯洛夫。"
    )
    result = _match_grouped_attendance(text, first, second, subject)
    assert (result is not None) is expected
    if result is not None:
        assert result[0] == text
        assert result[1] == "会见"
        assert set(result[2]) == {first, second}
        assert result[3] == "chronology_subject"


@pytest.mark.parametrize(
    "text",
    [
        "10月1日上午，准备去天安门参加庆祝大会前，在颐年堂会见赫鲁晓夫。"
        "参加会见的，中方有周恩来，苏方有苏斯洛夫。",
        "10月1日上午，去天安门参加庆祝大会前，在颐年堂会见赫鲁晓夫。"
        "参加接见的，中方有周恩来，苏方有苏斯洛夫。",
        "10月1日上午，去天安门参加庆祝大会前，在颐年堂会见赫鲁晓夫。"
        "参加会见的，中方有周恩来的秘书，苏方有苏斯洛夫。",
        "10月1日上午，去天安门参加庆祝大会前，在颐年堂会见赫鲁晓夫。"
        "随后参加会见的，中方有周恩来，苏方有苏斯洛夫。",
    ],
)
def test_grouped_attendance_guards(text: str) -> None:
    assert _match_grouped_attendance(text, "毛泽东", "周恩来", "毛泽东") is None
