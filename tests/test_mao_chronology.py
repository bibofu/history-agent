from pathlib import Path

from history_agent.research.catalog import load_person_catalog
from history_agent.research.chronology import PROFILES, PersonMentionResolver, _build_candidate
from history_agent.research.mao_chronology import extract_mao_entries, is_subject_entry
from test_chronology import _page

DOC = "mao_zedong_chronology_volume_4"


def test_mao_dates_footnotes_and_page_evidence() -> None:
    pages = {
        6: _page(DOC, 6, """1949年
五十六岁
10月
1
日下午，毛泽东与周恩来在北京召开会议讨论工作。
同日下午，出席人民群众举行的大会并宣读公告，
[1]邓小平，1954年担任某项职务。"""),
        7: _page(DOC, 7, """2
1 9 4 9 年
1
0 月
随后向到会群众讲话并接受代表致意。
10月2日—5日 在北京连续召开会议研究有关工作。
10月6日 致信有关部门研究下一阶段工作。"""),
    }
    entries = extract_mao_entries(pages, PROFILES[DOC], {6: 1949, 7: 1949})
    assert len(entries) == 4
    assert [e.temporal.start.value for e in entries] == [
        "1949-10-01", "1949-10-01", "1949-10-02", "1949-10-06",
    ]
    assert entries[2].temporal.end.value == "1949-10-05"
    assert entries[1].pages == [6, 7]
    assert "1954" not in entries[1].text
    assert "随后向到会群众" in entries[1].text
    resolver = PersonMentionResolver(load_person_catalog(Path("config/person_aliases.json")))
    candidate = _build_candidate(entries[1], resolver)
    assert candidate.event.review_status == "needs_review"
    assert [e.pdf_page_start for e in candidate.event.evidence] == [6, 7]
    assert not any(p.person_id == "deng_xiaoping" for p in candidate.event.participants)
    assert candidate.event.participants[0].mention_source == "chronology_subject"
    assert _build_candidate(entries[1], resolver).event == candidate.event


def test_mao_resets_at_year_boundaries_and_missing_pages() -> None:
    entries = extract_mao_entries({
        1: _page(DOC, 1, "12月31日 在北京召开会议部署下一阶段工作。"),
        2: _page(DOC, 2, "同日 在北京会见代表讨论工作安排。"),
        4: _page(DOC, 4, "同日 在上海与代表团举行工作会谈。"),
    }, PROFILES[DOC], {1: 1949, 2: 1950, 4: 1950})
    assert entries[0].temporal.start.value == "1949-12-31"
    assert entries[1].temporal.start.value == "1950"
    assert entries[2].temporal.start.value == "1950"
    assert "page_gap" in entries[1].rule_flags


def test_mao_relative_days_approximate_dates_and_outside_sections() -> None:
    entries = extract_mao_entries({
        1: _page(DOC, 1, "1月1日 目录中记载的日期不应生成正式事件。"),
        2: _page(DOC, 2, """7月初 同有关负责人讨论农村生产工作。
7月3日 在北京会见外地来访的代表团。
翌日 在北京再次会见同一代表团讨论安排。
夏衍来信讨论电影创作方面的工作。
本月 为有关部门题词并指导工作。"""),
    }, PROFILES[DOC], {2: 1961})
    assert len(entries) == 4
    assert entries[0].temporal.start.precision == "month"
    assert entries[2].temporal.start.value == "1961-07-04"
    assert "夏衍来信" in entries[2].text
    assert entries[3].temporal.start.value == "1961-07"


def test_mao_ocr_header_does_not_replace_previous_day_with_month() -> None:
    entries = extract_mao_entries({
        1: _page(DOC, 1, "8月9日 在延安主持会议讨论抗战形势，"),
        2: _page(DOC, 2, """620
l 9 4 5 年
8 月
并听取有关负责人的意见。
同日 和朱德致电有关方面表达欢迎。
ClJ 邓小平，当时担任某项职务。"""),
    }, PROFILES[DOC], {1: 1945, 2: 1945})
    assert len(entries) == 2
    assert entries[1].temporal.start.value == "1945-08-09"
    assert "邓小平" not in entries[1].text


def test_mao_background_entries_are_not_implicit_subject_actions() -> None:
    entries = extract_mao_entries({1: _page(DOC, 1, """
7月7日 夜，日本侵略军发动进攻，全国抗日战争开始。
7月8日 下午，在延安主持会议讨论抗战形势。
同日 毛泽东和朱德致电有关方面表达意见。""")}, PROFILES[DOC], {1: 1937})
    assert [is_subject_entry(e) for e in entries] == [False, True, True]


def test_identical_same_day_wording_on_different_dates_has_distinct_ids() -> None:
    entries = extract_mao_entries({1: _page(DOC, 1, """
7月23日 在北京主持工作会议讨论有关工作。
同日 下午三时，出席第一届全国人大二次会议。
7月25日 在北京主持工作会议讨论有关工作。
同日 下午三时，出席第一届全国人大二次会议。
""")}, PROFILES[DOC], {1: 1955})
    resolver = PersonMentionResolver(load_person_catalog(Path("config/person_aliases.json")))
    a, b = [_build_candidate(entries[i], resolver) for i in [1, 3]]
    assert a.event.description == b.event.description
    assert a.event.event_id != b.event.event_id
