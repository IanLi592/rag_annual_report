"""下载环节（src/download_reports.py）的回归守卫。

这一批守卫全部离线：零网络、零密钥、只碰 tmp_path。
判据刻意落在「产物」和「返回值语义」上，而不是 mock 巨潮返回的字段形状 ——
接口结果不可复现，mock 得再细也证明不了线上是对的。
"""
import re
from pathlib import Path

import pytest

import download_reports as dr

REPO = Path(dr.__file__).resolve().parents[1]


def make_pdf_bytes(body=1600):
    """一份结构上完整的最小 PDF：头部魔数 + 尾部 %%EOF。"""
    return b"%PDF-1.4\n" + b"x" * body + b"\n%%EOF\n"


def ann(title, url="static/xxx.PDF", code="000858", t=None):
    item = {
        "announcementTitle": title,
        "adjunctUrl": url,
        "secCode": code,
        "announcementId": "1213222211",
    }
    if t is not None:
        item["announcementTime"] = t
    return item


# ── ① 查询失败 ≠ 没有这份年报 ─────────────────────────────────────────────────

def test_query_exception_returns_none_not_empty(monkeypatch):
    class Boom:
        def post(self, *a, **k):
            raise ConnectionError("cninfo 断连")

    monkeypatch.setattr(dr.requests, "post", Boom().post)
    assert dr._do_query({"searchkey": "x"}) is None


def test_query_empty_result_returns_empty_list(monkeypatch):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"announcements": []}

    monkeypatch.setattr(dr.requests, "post", lambda *a, **k: Resp())
    assert dr._do_query({"searchkey": "x"}) == []


# ── ② 同一年的更正稿 / 摘要 / 英文版 不能被随手取第一条 ──────────────────────

def test_select_report_prefers_corrected_version():
    reports = [
        ann("2021年年度报告", t=1000),
        ann("2021年年度报告（更正后）", t=900),   # 时间更早也不能压过更正稿
    ]
    assert dr.select_report(reports, "2021")["announcementTitle"] == "2021年年度报告（更正后）"


def test_select_report_picks_newest_when_no_correction():
    reports = [ann("2021年年度报告", t=1000), ann("2021年年度报告", t=2000)]
    assert dr.select_report(reports, "2021")["announcementTime"] == 2000


def test_select_report_order_irrelevant_without_time():
    """接口没给可比较的时间戳时退回第一条，不许崩。"""
    reports = [ann("2021年年度报告"), ann("2021年年度报告")]
    assert dr.select_report(reports, "2021") is reports[0]


@pytest.mark.parametrize("title", [
    "2021年年度报告摘要",
    "2021年年度报告（英文）",
    "2021年年度报告（已取消）",
    "2021年年度报告补充公告",
    "2021年度利润分配的问询回复",
    "2022年年度报告",
])
def test_select_report_rejects_non_target_titles(title):
    assert dr.select_report([ann(title)], "2021") is None


def test_select_report_rejects_non_pdf_attachment():
    reports = [ann("2021年年度报告", url="static/xxx.DOCX")]
    assert dr.select_report(reports, "2021") is None


# ── ③ secCode 归属复核：stock_code 参数必须真的生效 ───────────────────────────

def test_restrict_to_stock_drops_other_company():
    reports = [ann("2021年年度报告", code="600519"), ann("2021年年度报告", code="000858")]
    kept = dr._restrict_to_stock(reports, "000858")
    assert [r["secCode"] for r in kept] == ["000858"]


def test_restrict_to_stock_passthrough_when_field_absent():
    """接口没返回 secCode 时不许把结果清空 —— 那等于用严格换来全量下载失败。"""
    reports = [ann("2021年年度报告")]
    for r in reports:
        r.pop("secCode")
    assert dr._restrict_to_stock(reports, "000858") == reports


def test_query_annual_reports_uses_stock_code_argument(monkeypatch):
    """守卫 docstring 与实现曾经不一致那个问题：形参 stock_code 必须参与筛选。"""
    seen = []

    def fake(payload):
        seen.append(payload["searchkey"])
        # 策略1 命中，但公告其实属于另一家公司
        return [ann("贵州茅台2021年年度报告", code="600519")]

    monkeypatch.setattr(dr, "_do_query", fake)
    out = dr.query_annual_reports("000858", "sz", "五粮液", "2021")
    assert out == []
    assert seen and seen[0] == "五粮液2021年年度报告"


def test_query_annual_reports_falls_back_to_spaced_name(monkeypatch):
    calls = []

    def fake(payload):
        calls.append(payload["searchkey"])
        return [ann("2021年年度报告", code="000858")] if len(calls) == 2 else []

    monkeypatch.setattr(dr, "_do_query", fake)
    out = dr.query_annual_reports("000858", "sz", "五粮液", "2021")
    assert calls[1] == "五 粮 液"
    assert len(out) == 1


def test_query_annual_reports_propagates_none(monkeypatch):
    monkeypatch.setattr(dr, "_do_query", lambda payload: None)
    assert dr.query_annual_reports("000858", "sz", "五粮液", "2021") is None


# ── ④ 断流不能把半截 PDF 留在正式路径上 ───────────────────────────────────────

def test_is_valid_pdf(tmp_path):
    good = tmp_path / "good.pdf"
    good.write_bytes(make_pdf_bytes())
    assert dr.is_valid_pdf(good) is True

    cut = tmp_path / "cut.pdf"          # 断流：头部对、尾部没有 %%EOF
    cut.write_bytes(b"%PDF-1.4\n" + b"x" * 2000)
    assert dr.is_valid_pdf(cut) is False

    tiny = tmp_path / "tiny.pdf"        # 404 页面存下来的碎片
    tiny.write_bytes(b"%PDF-1.4")
    assert dr.is_valid_pdf(tiny) is False

    html = tmp_path / "html.pdf"
    html.write_bytes(b"<html>" + b"x" * 2000)
    assert dr.is_valid_pdf(html) is False

    assert dr.is_valid_pdf(tmp_path / "nope.pdf") is False


def test_download_failure_leaves_no_final_file(tmp_path, monkeypatch):
    """核心守卫：失败后正式路径上不许出现任何文件。"""
    class HalfStream:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=1):
            yield make_pdf_bytes()[:500]
            raise ConnectionError("断流")

    monkeypatch.setattr(dr.requests, "get", lambda *a, **k: HalfStream())
    monkeypatch.setattr(dr.time, "sleep", lambda s: None)

    target = tmp_path / "600519_2021_贵州茅台_贵州茅台2021年年度报告.pdf"
    assert dr.download_pdf("http://x/y.PDF", target) is False
    assert target.exists() is False, "半截 PDF 被写成正式文件名，下次重跑会被当作已完成"


def test_existing_corrupt_pdf_is_not_trusted(tmp_path, monkeypatch):
    """旧实现在这里直接 return True —— 中毒入口。"""
    target = tmp_path / "a.pdf"
    target.write_bytes(b"%PDF-1.4\n" + b"x" * 2000)   # 有头无尾

    def fake_get(*a, **k):
        raise ConnectionError("仍然连不上")

    monkeypatch.setattr(dr.requests, "get", fake_get)
    monkeypatch.setattr(dr.time, "sleep", lambda s: None)

    assert dr.download_pdf("http://x/y.PDF", target) is False, \
        "残缺的既有文件被当成下载成功"


def test_download_success_replaces_part(tmp_path, monkeypatch):
    class Full:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=1):
            yield make_pdf_bytes()

    monkeypatch.setattr(dr.requests, "get", lambda *a, **k: Full())
    monkeypatch.setattr(dr.time, "sleep", lambda s: None)

    target = tmp_path / "b.pdf"
    assert dr.download_pdf("http://x/y.PDF", target) is True
    assert dr.is_valid_pdf(target) is True
    assert list(tmp_path.glob("*.part")) == [], "临时文件没被 replace 收干净"


def test_existing_complete_pdf_skipped_without_network(tmp_path, monkeypatch):
    """已存在且完整的文件仍然直接跳过 —— 修中毒不能把幂等性一起修没。"""
    target = tmp_path / "c.pdf"
    target.write_bytes(make_pdf_bytes())

    def boom(*a, **k):
        raise AssertionError("不该发起网络请求")

    monkeypatch.setattr(dr.requests, "get", boom)
    assert dr.download_pdf("http://x/y.PDF", target) is True


# ── ⑤ manifest 不许因为当次查询失败而凭空缩水 ────────────────────────────────

@pytest.fixture
def corpus(tmp_path, monkeypatch):
    raw = tmp_path / "raw_pdf"
    raw.mkdir()
    monkeypatch.setattr(dr, "RAW_DIR", raw)
    monkeypatch.setattr(dr, "MANIFEST_PATH", tmp_path / "manifest.json")
    return tmp_path, raw


def _full_entries(raw):
    entries = {}
    for code, plate, name in dr.TARGET_STOCKS:
        for year in dr.TARGET_YEARS:
            fn = f"{code}_{year}_{name}_{year}年年度报告.pdf"
            (raw / fn).write_bytes(make_pdf_bytes())
            entries[(code, year)] = {
                "stock_code": code, "plate": plate, "company_name": name,
                "year": year, "title": f"{year}年年度报告", "filename": fn,
                "source_url": f"http://static.cninfo.com.cn/{code}.PDF",
                "announce_id": "x",
            }
    return entries


def test_save_manifest_keeps_previous_entries(corpus):
    _, raw = corpus
    entries = _full_entries(raw)
    kept = dr.save_manifest(entries)
    assert len(kept) == 15
    assert dr.verify_corpus() == []


def test_save_manifest_drops_entries_without_valid_file(corpus):
    """查询失败当次不写新条目，但已有条目必须留着 —— 只丢磁盘上真没了的。"""
    _, raw = corpus
    entries = _full_entries(raw)
    (raw / entries[("000858", "2021")]["filename"]).write_bytes(b"%PDF-1.4")  # 断流残片
    kept = dr.save_manifest(entries)
    assert len(kept) == 14
    assert ("000858", "2021") not in {(e["stock_code"], e["year"]) for e in kept}


def test_load_manifest_roundtrip(corpus):
    _, raw = corpus
    entries = _full_entries(raw)
    dr.save_manifest(entries)
    loaded = dr.load_manifest()
    assert loaded[("000858", "2021")]["filename"] == entries[("000858", "2021")]["filename"]
    assert set(dr.MANIFEST_FIELDS) == set(loaded[("600519", "2023")].keys()), \
        "manifest 字段集合是 parse_pdf.py:557 的对外契约，不许漂移"


def test_verify_corpus_flags_missing_entry(corpus):
    _, raw = corpus
    entries = _full_entries(raw)
    entries.pop(("300750", "2022"))
    dr.save_manifest(entries)
    problems = dr.verify_corpus()
    assert any("缺少年报条目" in p and "300750" in p for p in problems)


def test_verify_corpus_flags_orphan_and_part(corpus):
    _, raw = corpus
    dr.save_manifest(_full_entries(raw))
    (raw / "外来的.pdf").write_bytes(make_pdf_bytes())
    (raw / "半成品.pdf.part").write_bytes(b"%PDF-1.4")
    problems = dr.verify_corpus()
    assert any("清单外的 PDF" in p for p in problems)
    assert any("残留下载临时文件" in p for p in problems)


def test_verify_corpus_reports_absent_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "MANIFEST_PATH", tmp_path / "nope.json")
    assert any("manifest 不存在" in p for p in dr.verify_corpus())


def test_verify_corpus_passes_on_real_artifacts():
    """仓库里有真实语料时，守卫必须承认它是好的（防止校验逻辑自己写歪）。"""
    if not dr.MANIFEST_PATH.exists():
        pytest.skip("本仓库没有 data/ 产物，跳过真实语料校验")
    assert dr.verify_corpus() == []


def test_load_manifest_absent_file_creates_nothing(tmp_path, monkeypatch):
    """读清单不许有写副作用 —— 校验路径（--verify）必须是纯只读的。"""
    ghost = tmp_path / "manifest.json"
    monkeypatch.setattr(dr, "MANIFEST_PATH", ghost)
    assert dr.load_manifest() == {}
    assert not ghost.exists()


# ── ⑥ --verify 的退出码必须对齐 CI 三档契约 ──────────────────────────────────

def test_verify_exit_prereq_when_corpus_absent(tmp_path, monkeypatch):
    """新克隆还没下载过 → 前置缺失(2)，不许报成代码回归(1)。"""
    monkeypatch.setattr(dr, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(dr, "RAW_DIR", tmp_path / "raw_pdf")
    assert dr.cli(["--verify"]) == dr.EXIT_PREREQ == 2


def test_verify_exit_pass_on_clean_corpus(corpus):
    _, raw = corpus
    dr.save_manifest(_full_entries(raw))
    assert dr.cli(["--verify"]) == dr.EXIT_PASS == 0


def test_verify_exit_assert_on_corrupt_pdf(corpus):
    _, raw = corpus
    entries = _full_entries(raw)
    dr.save_manifest(entries)
    (raw / entries[("000858", "2021")]["filename"]).write_bytes(b"%PDF-1.4\n" + b"x" * 2000)
    assert dr.cli(["--verify"]) == dr.EXIT_ASSERT == 1


def test_exit_code_constants_agree_with_ci_lib_sh():
    """Python 侧三档和 scripts/ci/ci_lib.sh 必须同名同值。

    两边各写一份是刻意的（用户在 CI 外面手敲 --verify 也要同一种语义），
    但重复就有漂移风险，所以把"不许漂移"本身做成断言。
    """
    lib = (REPO / "scripts" / "ci" / "ci_lib.sh").read_text(encoding="utf-8")
    for name, val in (("EXIT_PASS", 0), ("EXIT_ASSERT", 1), ("EXIT_PREREQ", 2)):
        assert getattr(dr, name) == val
        assert re.search(rf"^{name}={val}\s", lib, re.M), f"ci_lib.sh 里 {name} 不是 {val}"
