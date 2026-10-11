"""Live source probe; PASS means returned data passed basic checks, not full coverage."""

import hashlib
import json
import os
import platform
import signal
import subprocess
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from tempfile import mkdtemp
from zoneinfo import ZoneInfo

SYMBOL = "600036"
# Keep requested dates explicit: never silently substitute a different date.
TRANSACTION_DATES = ("20261009", "20240920")
SERVERS = (("110.41.147.114", 7709), ("124.70.176.52", 7709),
           ("119.97.185.59", 7709))
ATTEMPT_SECONDS = 60
REPORT_DIR = Path("test_reports")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_report(report):
    """Checkpoint every attempt; preserve completed evidence if later work fails."""
    path = REPORT_DIR / "report.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)
    lines = ["# 酉巳无悔 · mootdxPlus 实测", "", f"Commit: `{report['commit']}`", "",
             "| 项目 | 状态 | 返回条数 | 时间范围 / 原因 |", "|---|---|---:|---|"]
    for name, result in report["results"].items():
        detail = result.get("time_range") or result.get("error", "尚未执行")
        detail = str(detail).replace("|", "/").replace("\n", " ")
        lines.append(f"| {name} | {result['status']} | {result.get('rows', 0)} | {detail} |")
    lines += ["", "PASS 仅表示本次返回非空数据且通过基本结构校验；不证明全历史、全市场或完整逐笔覆盖。",
              "分笔日期由 mootdxPlus 按请求参数补入，不能独立证明服务器返回日期；分笔不等于交易所逐笔。",
              "每次线路尝试、异常、返回样本和原始 CSV 见 report.json 与附件。"]
    (REPORT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def deadline(_signum, _frame):
    raise TimeoutError(f"attempt exceeded {ATTEMPT_SECONDS}s")


def inspect_data(df, kind, requested_date, info, pd):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("接口没有返回非空 DataFrame")
    info.update(rows=len(df), columns=list(map(str, df.columns)),
                sample=json.loads(df.head(3).to_json(orient="records", date_format="iso")))
    required = {"price", "vol", "time", "datetime"} if kind == "transactions" else {
        "open", "high", "low", "close", "vol", "datetime"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"缺少必要字段: {sorted(missing)}")
    timestamps = pd.to_datetime(df["datetime"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("时间字段存在无法解析的值")
    info["time_range"] = [str(timestamps.min()), str(timestamps.max())]
    info["trading_dates_returned"] = int(timestamps.dt.date.nunique())
    info["duplicate_rows"] = int(df.duplicated().sum())
    price_cols = ["price"] if kind == "transactions" else ["open", "high", "low", "close"]
    values = df[price_cols + ["vol"]].apply(pd.to_numeric, errors="coerce")
    if (values.isna().any().any() or values.isin([float("inf"), float("-inf")]).any().any()
            or (values[price_cols] <= 0).any().any() or (values["vol"] < 0).any()):
        raise ValueError("价格/成交量字段无效")
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    if timestamps.dt.date.max() > today:
        raise ValueError("返回数据包含未来日期")
    if kind == "transactions":
        if not (timestamps.dt.strftime("%Y%m%d") == requested_date).all():
            raise ValueError("返回日期字段与请求日期不一致")
        # Equal timestamps and identical prints can be legitimate; do not deduplicate.
        info["date_provenance"] = "mootdxPlus attaches requested date; server date not independently verified"
    else:
        if timestamps.duplicated().any():
            raise ValueError("K 线时间重复，可能存在重复分页")
        if timestamps.dt.date.nunique() < 2 or timestamps.dt.date.min() >= today:
            raise ValueError("未获取跨交易日历史 K 线")
        if ((df["low"] > df[price_cols].min(axis=1)).any()
                or (df["high"] < df[price_cols].max(axis=1)).any()):
            raise ValueError("OHLC 高低价关系异常")
    info["coverage"] = "returned data only; full historical completeness not independently verified"


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cases = [("daily", "bars", 9, None), ("minute_5", "bars", 0, None),
             ("minute_1", "bars", 8, None)]
    cases += [(f"transactions_{date}", "transactions", None, date) for date in TRANSACTION_DATES]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    report = {"project": "酉巳无悔", "source": "mootdxPlus", "symbol": SYMBOL,
              "timestamp_utc": utc_now(), "commit": commit,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "python": platform.python_version(), "run_id": os.getenv("GITHUB_RUN_ID"),
              "run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
              "results": {name: {"status": "NOT_RUN", "attempts": []} for name, *_ in cases}}
    save_report(report)
    try:
        # Isolate the probe from user config and avoid the library's automatic scan
        # of every HQ/EX/GP host. Use only the three explicit HQ endpoints above.
        os.environ["MOOTDX_HOME"] = mkdtemp(prefix="yousi-mootdx-")
        import pandas as pd
        import mootdx
        from mootdx.consts import CONFIG
        from mootdx.quotes import Quotes
        config_file = Path(os.environ["MOOTDX_HOME"]) / "config.json"
        config_file.write_text(json.dumps(CONFIG), encoding="utf-8")
        report["module_path"] = str(Path(mootdx.__file__).resolve())
        report["versions"] = {name: version(name) for name in ("mootdx", "tdxpy", "pandas")}
        signal.signal(signal.SIGALRM, deadline)
    except Exception as exc:
        for result in report["results"].values():
            result.update(status="ERROR", error=f"初始化失败: {type(exc).__name__}: {exc}")
        report["initialization_traceback"] = traceback.format_exc()
        save_report(report)
        return 1

    for name, kind, frequency, date in cases:
        result = report["results"][name]
        result.update(status="RUNNING", request={"symbol": SYMBOL, "frequency": frequency,
                                                 "date": date, "page_size": 800, "max_pages": 200})
        save_report(report)
        for number, server in enumerate(SERVERS, start=1):
            info = {"server": list(server), "started_utc": utc_now(), "stage": "connect"}
            started = time.monotonic()
            client = None
            signal.alarm(ATTEMPT_SECONDS)
            print(f"{name}: attempt {number}, server={server}", flush=True)
            try:
                client = Quotes.factory(market="std", server=server, bestip=False, timeout=5,
                                        heartbeat=False, auto_retry=False, raise_exception=True)
                info["stage"] = "fetch"
                if kind == "bars":
                    df = client.bars_all(symbol=SYMBOL, frequency=frequency, max_pages=200)
                else:
                    df = client.transactions_all(symbol=SYMBOL, date=date, max_pages=200)
                info["stage"] = "validate"
                if isinstance(df, pd.DataFrame):
                    filename = f"{name}-attempt-{number}.csv"
                    df.to_csv(REPORT_DIR / filename, index=True, encoding="utf-8-sig")
                    info.update(csv=filename, rows=len(df))
                inspect_data(df, kind, date, info, pd)
                info["status"] = "PASS"
            except Exception as exc:
                info.update(status="FAIL", error=f"{type(exc).__name__}: {exc}",
                            traceback=traceback.format_exc())
            finally:
                signal.alarm(0)
                if client is not None:
                    try:
                        client.close()
                    except Exception as exc:
                        info["close_error"] = str(exc)
                info["elapsed_seconds"] = round(time.monotonic() - started, 3)
                result["attempts"].append(info)
                print(json.dumps(info, ensure_ascii=False), flush=True)
                if info["status"] == "PASS":
                    result.update(status="PASS", rows=info["rows"], time_range=info["time_range"],
                                  server=info["server"])
                save_report(report)
            if info["status"] == "PASS":
                break
        if result["status"] != "PASS":
            result.update(status="FAIL", error=result["attempts"][-1]["error"])
        save_report(report)
    report["finished_utc"] = utc_now()
    save_report(report)
    print((REPORT_DIR / "summary.md").read_text(encoding="utf-8"), flush=True)
    return int(any(item["status"] != "PASS" for item in report["results"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
