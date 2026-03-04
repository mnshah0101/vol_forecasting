import lakeapi
import datetime
import boto3
import pandas as pd
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed


USER_TMP = '/tmp'

AWS_ACCESS_KEY_ID = ''
AWS_SECRET_ACCESS_KEY = ''
AWS_REGION = 'eu-west-1'


def download_instrument(symbol: str, exchange: str, start: datetime.datetime, end: datetime.datetime) -> str:
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )

    print(f"[{exchange}/{symbol}] Downloading...")

    df = lakeapi.load_data(
        table="level_1",
        start=start,
        end=end,
        symbols=[symbol],
        exchanges=[exchange],
        boto3_session=session,
        cached=False
    )

    df = df[["received_time", "bid_0_price", "bid_0_size", "ask_0_price", "ask_0_size"]].rename(
        columns={
            "bid_0_price": "p_bid_0_price",
            "bid_0_size": "p_bid_0_size",
            "ask_0_price": "p_ask_0_price",
            "ask_0_size": "p_ask_0_size",
        }
    )

    out_path = os.path.join(USER_TMP, f"{symbol.replace('-', '_')}_{exchange}.parquet")
    df.to_parquet(out_path, index=False)
    print(f"[{exchange}/{symbol}] Saved to {out_path}  ({len(df)} rows)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download L1 data from LakeAPI")
    parser.add_argument("--exchange", required=True, help="Exchange name (e.g. BINANCE)")
    parser.add_argument("--instruments", required=True, nargs="+", help="Instrument symbols (e.g. BTC-USDT-PERP ETH-USDT-PERP)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    start = datetime.datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.datetime.strptime(args.end, "%Y-%m-%d")

    os.makedirs(USER_TMP, exist_ok=True)

    with ProcessPoolExecutor(max_workers=len(args.instruments)) as pool:
        futures = {
            pool.submit(download_instrument, sym, args.exchange, start, end): sym
            for sym in args.instruments
        }
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"[{args.exchange}/{sym}] FAILED: {e}")
