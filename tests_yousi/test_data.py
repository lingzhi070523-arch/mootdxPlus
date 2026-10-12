"""Live source probe; PASS means returned data passed basic checks, not full coverage."""

import hashlib
import importlib
import inspect
import struct
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
        if timestamps.dt.date.min() > today:
            raise ValueError("未获取跨交易日历史 K 线")
        if ((df["low"] > df[price_cols].min(axis=1)).any()
                or (df["high"] < df[price_cols].max(axis=1)).any()):
            raise ValueError("OHLC 高低价关系异常")
    info["coverage"] = "returned data only; full historical completeness not independently verified"


def _exception_chain(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def install_bar_response_diagnostic():
    """Observe parser input without changing response bytes or parsing behavior."""
    module = importlib.import_module("tdxpy.parser.std.get_security_bars")
    patched = []
    for _, cls in inspect.getmembers(module, inspect.isclass):
        method = cls.__dict__.get("parseResponse")
        if method is None or method.__module__ != module.__name__:
            continue
        original = method

        def traced(self, body_buf, _original=original, _class=cls.__name__):
            buf = bytes(body_buf)
            diagnostic = {"parser_class": _class, "body_length": len(buf),
                          "prefix_hex": buf[:16].hex(),
                          "declared_count_le16": struct.unpack("<H", buf[:2])[0] if len(buf) >= 2 else None}
            print("BAR_RESPONSE_DIAGNOSTIC " + json.dumps(diagnostic), flush=True)
            try:
                return _original(self, body_buf)
            except Exception as exc:
                print("BAR_PARSE_FAILURE " + json.dumps({
                    **diagnostic, "exception": type(exc).__name__, "message": str(exc)
                }), flush=True)
                raise

        cls.parseResponse = traced
        patched.append(cls.__name__)
    print("BAR_DIAGNOSTIC_INSTALLED " + json.dumps(patched), flush=True)
    if not patched:
        raise RuntimeError("No bar parser class was instrumented")


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cases = [("daily_1", "bars", 9, None), ("minute_1", "bars", 8, None)]  # One daily and one 1-minute bar; minimal protocol probes.
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
        install_bar_response_diagnostic()
        # Diagnostic only: inspect the outgoing request count without altering bytes.
        request_module = importlib.import_module("tdxpy.parser.std.get_security_bars")
        for _, request_cls in inspect.getmembers(request_module, inspect.isclass):
            if request_cls.__module__ != request_module.__name__:
                continue
            for method_name in ("setParams", "getRequest", "get_request"):
                original_method = request_cls.__dict__.get(method_name)
                if not callable(original_method):
                    continue
                def traced_request(self, *args, _orig=original_method, _method=method_name, **kwargs):
                    result = _orig(self, *args, **kwargs)
                    print("BAR_REQUEST_DIAGNOSTIC " + json.dumps({
                        "method": _method,
                        "args": [repr(x) for x in args],
                        "kwargs": {k: repr(v) for k, v in kwargs.items()},
                        "result_length": len(result) if isinstance(result, (bytes, bytearray)) else None,
                        "result_hex": bytes(result).hex() if isinstance(result, (bytes, bytearray)) else None,
                    }), flush=True)
                    return result
                setattr(request_cls, method_name, traced_request)
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

    # Independent implementation control: one bar per frequency on one endpoint.
    # Never treat a successful independent client as mootdxPlus success.
    report["independent_pytdx"] = {}
    try:
        from pytdx.hq import TdxHq_API as PytdxAPI
        for label, frequency in (("daily_1", 9), ("minute_1", 8)):
            probe = {"server": list(SERVERS[0]), "symbol": SYMBOL,
                     "frequency": frequency, "start": 0, "count": 1}
            api = PytdxAPI(heartbeat=False, auto_retry=False)
            try:
                connected = api.connect(SERVERS[0][0], SERVERS[0][1], time_out=5)
                probe["connected"] = bool(connected)
                if not connected:
                    raise ConnectionError("pytdx connect returned false")
                bars = api.get_security_bars(frequency, 1, SYMBOL, 0, 1)
                probe["rows"] = len(bars) if bars is not None else 0
                probe["sample"] = bars[:1] if bars else []
                probe["status"] = "PASS" if probe["rows"] == 1 else "FAIL"
            except Exception as exc:
                probe.update(status="FAIL", error=f"{type(exc).__name__}: {exc}",
                             traceback=traceback.format_exc())
            finally:
                try:
                    api.disconnect()
                except Exception:
                    pass
            report["independent_pytdx"][label] = probe
            print("INDEPENDENT_PYTDX " + json.dumps(probe, ensure_ascii=False, default=str), flush=True)
            save_report(report)
    except Exception as exc:
        report["independent_pytdx"]["setup_error"] = f"{type(exc).__name__}: {exc}"
        print("INDEPENDENT_PYTDX_SETUP_ERROR " + traceback.format_exc(), flush=True)
        save_report(report)

    for name, kind, frequency, date in cases:
        result = report["results"][name]
        result.update(status="RUNNING", request={"symbol": SYMBOL, "frequency": frequency,
                                                 "date": date, "start": 0, "offset": 1, "mode": "single_page"})
        save_report(report)
        for number, server in enumerate(SERVERS, start=1):
            info = {"server": list(server), "started_utc": utc_now(), "stage": "connect", "diagnostic": "single_page_1_bar"}
            started = time.monotonic()
            client = None
            signal.alarm(ATTEMPT_SECONDS)
            print(f"{name}: attempt {number}, server={server}", flush=True)
            try:
                client = Quotes.factory(market="std", server=server, bestip=False, timeout=5,
                                        heartbeat=False, auto_retry=False, raise_exception=True)
                info["connected"] = True
                info["stage"] = "fetch"
                if kind == "bars":
                    df = client.bars(symbol=SYMBOL, frequency=frequency, start=0, offset=1)
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
                            exception_chain=[{"type": type(e).__name__, "message": str(e)} for e in _exception_chain(exc)],
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
