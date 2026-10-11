
import json
import traceback
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from mootdx.quotes import Quotes


SYMBOL = "600036"
MARKET = 1  # 上海证券交易所
REPORT_DIR = Path("test_reports")
REPORT_DIR.mkdir(exist_ok=True)

results = {}


def record(name, func):
    print(f"\n===== {name} =====", flush=True)

    try:
        df = func()

        if df is None:
            raise ValueError("接口返回 None")

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        if df.empty:
            raise ValueError("接口返回空数据")

        info = {
            "status": "PASS",
            "rows": len(df),
            "columns": list(map(str, df.columns)),
            "sample": json.loads(
                df.head(3).to_json(
                    orient="records",
                    date_format="iso"
                )
            )
        }

        print(df.head(3).to_string())
        print(f"记录数量: {len(df)}")

        # 保存原始返回结果，不做业务计算
        df.to_csv(
            REPORT_DIR / f"{name}.csv",
            index=True,
            encoding="utf-8-sig"
        )

    except Exception as exc:
        info = {
            "status": "FAIL",
            "error": str(exc)
        }
        traceback.print_exc()

    results[name] = info


def main():
    print("酉巳无悔 - mootdxPlus 数据源测试")
    print("股票:", SYMBOL)

    client = Quotes.factory(
        market="std",
        multithread=False,
        heartbeat=True
    )

    # 日线：通达信 frequency=9
    record(
        "daily",
        lambda: client.bars_all(
            symbol=SYMBOL,
            frequency=9
        )
    )

    # 5分钟线：通达信 frequency=0
    record(
        "minute_5",
        lambda: client.bars_all(
            symbol=SYMBOL,
            frequency=0
        )
    )

    # 历史分笔：测试最近一个已完成交易日
    record(
        "transactions",
        lambda: client.transactions_all(
            symbol=SYMBOL,
            date="20261009"
        )
    )

    report = {
        "project": "酉巳无悔",
        "source": "mootdxPlus",
        "symbol": SYMBOL,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "results": results
    }

    with open(
        REPORT_DIR / "report.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n===== 测试总结 =====")
    for name, info in results.items():
        print(name, info["status"])

    if any(
        info["status"] != "PASS"
        for info in results.values()
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
