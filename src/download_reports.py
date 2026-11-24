"""
数据下载脚本：从巨潮资讯网批量下载上市公司年报

巨潮资讯网（cninfo.com.cn）是中国证监会指定的上市公司信息披露平台
所有年报均为公开信息，下载合法合规

两种运行模式：

    python src/download_reports.py            # 联网下载（磁盘上已存在且完整则跳过）
    python src/download_reports.py --verify   # 零网络零密钥，只校验本地语料完整性

退出码（与 scripts/ci/ci_lib.sh 的 P0-5 三档契约一致）：
    0 == 全部通过
    1 == 断言失败：查询失败 / 未命中 / 下载失败 / 语料真有缺陷
    2 == 前置缺失：还没下载过语料（新克隆的正常状态，不算回归）

这一条分档必须硬：本脚本是整条流水线唯一的联网环节，它一旦"半成功但返回 0"，
下游 parse_pdf 就会拿着一份少了两家公司、或者装着半截 PDF 的 manifest 继续跑，
而所有检索指标看起来都完全正常。反过来，新克隆还没建语料时也不能报 1 ——
那会让 CI 把"正常的第一次"当成代码回归。
"""

import os
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 退出码三档，与 scripts/ci/ci_lib.sh 里的同名常量严格一致。
# 为什么要在这里重复一份而不是只让外壳脚本翻译：`python src/download_reports.py
# --verify` 也会被新克隆的用户手工敲，语义必须和 CI 里跑是同一种。
EXIT_PASS   = 0    # 全部通过
EXIT_ASSERT = 1    # 断言失败：真回归
EXIT_PREREQ = 2    # 前置缺失：产物/网络没到位，不代表代码有问题

RAW_DIR = Path(__file__).parent.parent / "data" / "raw_pdf"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 股票代码、所属板块、公司简称
TARGET_STOCKS = [
    ("600519", "sh", "贵州茅台"),   # 上交所主板
    ("000858", "sz", "五粮液"),     # 深交所主板
    ("601318", "sh", "中国平安"),   # 上交所主板
    ("300750", "sz", "宁德时代"),   # 创业板
    ("002415", "sz", "海康威视"),   # 深交所（2021 年中小板已并入主板）
]

TARGET_YEARS = ["2021", "2022", "2023"]

CNINFO_QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_BASE_URL  = "http://static.cninfo.com.cn/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "http://www.cninfo.com.cn/",
    "Content-Type": "application/x-www-form-urlencoded",
}

# 标题口径：只要「年度报告全文」。同一年的同类公告有好几种，
# 摘要不含完整报表、英文版是另一套科目排版、已取消/补充/问询回复都不是取数对象。
TITLE_EXCLUDE = ("摘要", "英文", "取消", "补充", "问询", "回复")
# 出现这些词说明是更正/修订稿 —— 更正后才是权威文本，优先级高于原始稿。
TITLE_CORRECTION = ("更正", "修订")


# ── 标题与归属筛选（纯函数，可离线断言）───────────────────────────────────────

def _is_full_report_title(title: str, year: str) -> bool:
    """标题是否指向 `year` 年的「年度报告全文」。"""
    t = (title or "").strip()
    if "年度报告" not in t:
        return False
    if any(k in t for k in TITLE_EXCLUDE):
        return False
    return year in t


def _restrict_to_stock(reports: list[dict], stock_code: str) -> list[dict]:
    """按接口返回的 secCode 复核公告归属。

    为什么需要：策略 2 的 searchkey 只有公司名、不带年份，全靠标题筛；一旦
    同名或近似名的公司混进结果，下游拿到的就是**另一家公司的年报**，后面
    再精确的表格取数也救不回来 —— 这是"跨公司串数"在语料层的同一类问题。

    只在字段真实存在时才生效：拿一个不存在的字段去过滤会把结果清空，
    等于用"更严格"换来一次全量下载失败。全部被筛掉时返回空列表，
    让调用方明确报"未找到"，也不要把可疑公告当成品下下来。
    """
    if not any(r.get("secCode") for r in reports):
        return reports
    code = stock_code.upper()
    keep = [r for r in reports if not r.get("secCode") or str(r["secCode"]).upper() == code]
    dropped = len(reports) - len(keep)
    if dropped:
        logger.warning(f"secCode 复核筛掉 {dropped} 条不属于 {stock_code} 的公告")
    return keep


def select_report(reports: list[dict], year: str) -> dict | None:
    """从同一年的公告堆里挑出唯一该下载的那一份。

    为什么需要：接口返回的是一类公告的列表，同一年的年报可能同时存在原始稿
    和更正稿。旧实现直接取 `candidates[0]`，选中哪一份取决于巨潮的返回顺序
    （未文档化、也不保证稳定），等于把语料权威性交给运气。

    口径（优先级从高到低）：
      1. 标题含「更正/修订」的版本
      2. `announcementTime` 最新的一份（字段缺失或类型不对则跳过这一层）
      3. 列表里的第一份
    """
    candidates = [
        r for r in reports
        if _is_full_report_title(r.get("announcementTitle", ""), year)
        and str(r.get("adjunctUrl", "")).upper().endswith(".PDF")
    ]
    if not candidates:
        return None

    corrected = [
        r for r in candidates
        if any(k in r.get("announcementTitle", "") for k in TITLE_CORRECTION)
    ]
    if corrected:
        candidates = corrected

    times = [r.get("announcementTime") for r in candidates]
    if times and all(isinstance(t, (int, float)) for t in times):
        best = max(range(len(candidates)), key=lambda i: candidates[i]["announcementTime"])
        return candidates[best]
    return candidates[0]


# ── 结构完整性（纯函数，可离线断言）───────────────────────────────────────────

def is_valid_pdf(path: Path) -> bool:
    """PDF 结构是否完整：头部 `%PDF-` 魔数 + 尾部 2 KB 内有 `%%EOF`。

    只做结构校验、不解析内容 —— 判断"下载有没有断流"够了；
    判断"表格会不会解析坏"是 parse_pdf 那一层的事。
    """
    try:
        size = path.stat().st_size
        if size < 1024:
            return False
        with open(path, "rb") as f:
            if not f.read(8).startswith(b"%PDF-"):
                return False
            f.seek(max(0, size - 2048))
            return b"%%EOF" in f.read()
    except OSError:
        return False


# ── 巨潮接口 ──────────────────────────────────────────────────────────────────

def _do_query(payload: dict) -> list[dict] | None:
    """查询一次。返回 `None` == 请求失败，`[]` == 查到了但没有结果。

    为什么必须区分：旧实现把所有异常收敛成空列表，而且只打 debug 级日志
    （默认级别是 INFO，根本看不见）。于是一次断网和一份真不存在的年报在日志里
    长得一模一样，`main` 照样打「完成！共下载 N 份」并以 0 退出 —— 半成功语料
    就是这么静默流进下游的。
    """
    try:
        resp = requests.post(CNINFO_QUERY_URL, data=payload, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.json().get("announcements") or []
    except Exception as e:
        logger.warning(f"查询失败（记为错误，不等于「没有这份年报」）: {e}")
        return None


def query_annual_reports(stock_code: str, plate: str, company_name: str, year: str) -> list[dict] | None:
    """
    搜索年报，两级策略：

      策略 1  searchkey = "公司名+年份+年年度报告"
              适合标题里带公司名的情况（贵州茅台、中国平安）
      策略 2  searchkey = 公司名逐字加空格（"五粮液" → "五 粮 液"）
              巨潮部分深市股票的 secName 就是带空格存的，策略 1 的拼接词匹配不上；
              这一级的年份与"全文"口径交给 select_report 按标题筛

    两种口径下都用 secCode 复核一遍归属（见 `_restrict_to_stock`）。
    返回 `None` == 两级查询都请求失败；`[]` == 确实没搜到。
    """
    pub_year  = str(int(year) + 1)     # 2021 年报发布于 2022 年，别搜错窗口
    plate_col = "sse" if plate == "sh" else "szse"

    base_payload = {
        "stock":     "",               # 接口已不支持按代码过滤，保留空值仅为兼容旧 payload
        "tabName":   "fulltext",
        "pageSize":  30,
        "pageNum":   1,
        "column":    plate_col,
        "category":  "category_ndbg_szsh",   # "年度报告"类别码，比关键词可靠
        "plate":     plate,
        "seDate":    f"{pub_year}-01-01~{pub_year}-06-30",  # 法定 4/30 前披露，留 2 个月缓冲
        "secid":     "",
        "sortName":  "",
        "sortType":  "",
        "isHLtitle": True,
    }

    # 策略 1：公司名 + 年份关键词
    results = _do_query({**base_payload, "searchkey": f"{company_name}{year}年年度报告"})
    if results is None:
        return None
    if results:
        return _restrict_to_stock(results, stock_code)

    # 策略 2：公司名每字之间加空格
    logger.info(f"  策略1无结果，改用拆字关键词: {' '.join(list(company_name))}")
    results2 = _do_query({**base_payload, "searchkey": " ".join(list(company_name))})
    if results2 is None:
        return None
    return _restrict_to_stock(results2, stock_code)


def download_pdf(pdf_url: str, save_path: Path) -> bool:
    """原子下载：先写 `.part`，结构校验通过才 `os.replace` 成正式文件名。

    为什么不能直接在正式路径上流式写：旧实现三次重试全失败时半成品留在原地，
    下次重跑 `save_path.exists()` 判定「已存在，跳过」并返回 True，于是**截断的
    PDF 被正式写进 manifest**，下游把它当完整语料用到底。
    改成 `.part` + replace 后，正式路径上只可能出现校验通过的文件。
    这里刻意不删任何东西：残留的 `.part` 无下游可见性，下次以 wb 打开直接截断。
    """
    if save_path.exists():
        if is_valid_pdf(save_path):
            logger.info(f"已存在且完整，跳过: {save_path.name}")
            return True
        # 只有"确认残缺"才会被覆盖，完整文件永远走上面那条跳过分支
        logger.warning(f"已存在但结构不完整，重新下载覆盖: {save_path.name}")

    tmp_path = save_path.with_name(save_path.name + ".part")
    for attempt in range(3):
        try:
            resp = requests.get(pdf_url, headers=HEADERS, timeout=120, stream=True)
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                f.flush()
                os.fsync(f.fileno())
            if not is_valid_pdf(tmp_path):
                raise OSError(f"下载内容不是完整 PDF（{tmp_path.stat().st_size} 字节）")
            os.replace(tmp_path, save_path)
            logger.info(f"下载成功: {save_path.name}  ({save_path.stat().st_size // 1024} KB)")
            return True
        except Exception as e:
            logger.warning(f"第{attempt+1}次失败: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)

    logger.error(f"下载失败: {pdf_url}")
    return False


def sanitize(name: str) -> str:
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, "_")
    return name.strip()


# ── 语料清单 ──────────────────────────────────────────────────────────────────

MANIFEST_PATH = RAW_DIR.parent / "manifest.json"

# 下游 parse_pdf.py:557 直接按这份清单驱动整个流水线：它读 filename /
# stock_code / year，并把整条记录原样写进 parsed JSON 的 meta。
# 所以字段集合是对外契约 —— 这里只允许"更新"，不允许增删字段，
# 也不允许因为本次没跑到某家公司就把条目删掉。
MANIFEST_FIELDS = ("stock_code", "plate", "company_name", "year",
                   "title", "filename", "source_url", "announce_id")


def load_manifest() -> dict[tuple, dict]:
    """读上一版 manifest，键为 (stock_code, year)。文件坏或缺失都当空表。"""
    if not MANIFEST_PATH.exists():
        return {}
    try:
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            entries = json.load(f)
        return {(e["stock_code"], e["year"]): e for e in entries}
    except Exception as e:
        logger.warning(f"manifest 读取失败，按空清单重建: {e}")
        return {}


def save_manifest(entries: dict[tuple, dict]) -> list[dict]:
    """落盘 manifest：只保留磁盘上真实存在且结构完整的条目。

    为什么起点是"上一版"而不是"本次成功列表"：旧实现每次 `manifest = []` 从头
    攒，于是一旦某几家公司的查询当次失败，manifest 就**凭空少掉那几条**，
    parse_pdf 只遍历 manifest → 语料无声缩到 13 份，指标照样能跑出一堆好看的数字。
    """
    kept = []
    for item in entries.values():
        path = RAW_DIR / item["filename"]
        if path.exists() and is_valid_pdf(path):
            kept.append({k: item.get(k, "") for k in MANIFEST_FIELDS})
        else:
            logger.warning(f"磁盘上没有完整 PDF，从清单剔除: {item['filename']}")
    kept.sort(key=lambda x: (x["stock_code"], x["year"]))
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    return kept


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> int:
    entries = load_manifest()
    problems: list[str] = []

    for stock_code, plate, company_name in TARGET_STOCKS:
        for year in TARGET_YEARS:
            logger.info(f"── {company_name}({stock_code}) {year}年报 ──")
            reports = query_annual_reports(stock_code, plate, company_name, year)

            if reports is None:
                problems.append(f"{company_name} {year}: 查询失败（网络/接口）")
                continue

            report = select_report(reports, year)
            if report is None:
                problems.append(f"{company_name} {year}: 未找到年报全文（候选 {len(reports)} 条）")
                for r in reports[:3]:
                    logger.debug(f"    候选: {r.get('announcementTitle')} | {r.get('adjunctUrl','')[:50]}")
                continue

            title    = report["announcementTitle"]
            pdf_url  = CNINFO_BASE_URL + report["adjunctUrl"]
            filename = sanitize(f"{stock_code}_{year}_{company_name}_{title}.pdf")
            save_path = RAW_DIR / filename

            if download_pdf(pdf_url, save_path):
                entries[(stock_code, year)] = {
                    "stock_code":   stock_code,
                    "plate":        plate,
                    "company_name": company_name,
                    "year":         year,
                    "title":        title,
                    "filename":     filename,
                    "source_url":   pdf_url,
                    "announce_id":  report.get("announcementId"),
                }
            else:
                problems.append(f"{company_name} {year}: 下载失败")

            time.sleep(random.uniform(1.5, 3.0))

    kept = save_manifest(entries)

    logger.info(f"\n完成！清单内 {len(kept)} 份年报（期望 {len(TARGET_STOCKS) * len(TARGET_YEARS)}）")
    for item in kept:
        logger.info(f"  {item['company_name']} {item['year']}: {item['filename']}")
    for p in problems:
        logger.error(f"  问题: {p}")
    if len(kept) < len(TARGET_STOCKS) * len(TARGET_YEARS) and not problems:
        logger.error("  问题: 清单条数少于预期，但本轮没有报错 —— 请跑 --verify 复查")

    return EXIT_ASSERT if problems or len(kept) != len(TARGET_STOCKS) * len(TARGET_YEARS) else EXIT_PASS


# ── 离线校验（零网络、零密钥）─────────────────────────────────────────────────

def corpus_prereq_missing() -> str:
    """语料校验的前置是否具备。缺则返回一句说明，具备则返回空串。

    为什么单拎出来：CI 契约里"产物还没建"是退出码 2（前置缺失），不是 1
    （断言失败）。新克隆跑第一次 CI 时 data/ 是空的，这道门必须说"跳过"，
    不能把"正常的第一次"报成代码回归。
    """
    if not MANIFEST_PATH.exists():
        return f"清单不存在（还没跑过下载）: {MANIFEST_PATH}"
    if not RAW_DIR.exists():
        return f"语料目录不存在: {RAW_DIR}"
    return ""


def verify_corpus() -> list[str]:
    """校验本地语料：manifest ↔ 磁盘 PDF ↔ 结构完整 ↔ 5 公司 × 3 年齐全。

    返回问题列表，空列表 == 语料可用。这一段是下载环节唯一该有、
    而此前没有的守卫：下载逻辑本身不可离线复现（要碰外网、结果会变），
    mock 掉 requests 之后什么都没验证到；**该固化的其实是产物校验**。
    """
    problems: list[str] = []
    expected = {(c, y) for c, _, _ in TARGET_STOCKS for y in TARGET_YEARS}

    if not MANIFEST_PATH.exists():
        return [f"manifest 不存在: {MANIFEST_PATH}"]

    entries = load_manifest()
    for (code, year), item in sorted(entries.items()):
        path = RAW_DIR / item["filename"]
        if not path.exists():
            problems.append(f"清单有记录但文件缺失: {item['filename']}")
        elif not is_valid_pdf(path):
            problems.append(f"PDF 结构不完整（疑似断流）: {item['filename']}")

    for code, year in sorted(expected - set(entries)):
        problems.append(f"缺少年报条目: {code} {year}")

    known = {item["filename"] for item in entries.values()}
    for path in sorted(RAW_DIR.glob("*.pdf")):
        if path.name not in known:
            problems.append(f"磁盘上有清单外的 PDF（下游不会处理它）: {path.name}")
    for path in sorted(RAW_DIR.glob("*.pdf.part")):
        problems.append(f"残留下载临时文件: {path.name}")

    for (code, year), item in entries.items():
        if item["filename"].startswith(f"{code}_{year}_"):
            continue
        problems.append(f"清单与文件名不匹配: {code}/{year} → {item['filename']}")

    return problems


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="巨潮年报下载 / 本地语料校验")
    parser.add_argument("--verify", action="store_true",
                        help="只校验本地 data/raw_pdf 与 manifest 是否齐全完整，不联网、不需要密钥")
    args = parser.parse_args(argv)

    if args.verify:
        missing = corpus_prereq_missing()
        if missing:
            logger.warning(f"前置缺失，语料门跳过: {missing}")
            return EXIT_PREREQ
        problems = verify_corpus()
        if problems:
            for p in problems:
                logger.error(f"语料缺陷: {p}")
            logger.error(f"共 {len(problems)} 项问题")
            return EXIT_ASSERT
        logger.info(f"语料校验通过：{len(TARGET_STOCKS) * len(TARGET_YEARS)} 份年报齐全且结构完整")
        return EXIT_PASS

    return main()


if __name__ == "__main__":
    sys.exit(cli())
