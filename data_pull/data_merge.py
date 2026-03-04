import polars as pl
from pathlib import Path
import math


def process_orderbook_data(
    input_file: str,
    output_file: str,
    prefix: str,
    gap_threshold_minutes: int = 30,
):
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gap_threshold_ns = gap_threshold_minutes * 60 * 1_000_000_000
    cols_to_drop = ["origin_time", "exchange", "symbol"]
    one_sec_ns = 1_000_000_000

    # --- Step 1: Read ONLY received_time to find segment boundaries ---
    print("Scanning received_time to find segment boundaries...")
    times = (
        pl.scan_parquet(input_path)
        .select("received_time")
        .sort("received_time")
        .collect()
    )

    diffs = times["received_time"].diff().dt.total_nanoseconds()
    gap_mask = (diffs > gap_threshold_ns).fill_null(False)
    gap_indices = gap_mask.arg_true().to_list()

    n = len(times)
    boundaries = [0] + gap_indices + [n]
    segments = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    print(f"Found {len(segments)} segment(s)")

    del times, diffs, gap_mask

    # --- Step 2: Process each segment and collect resampled frames ---
    resampled_parts: list[pl.DataFrame] = []

    for seg_id, (row_start, row_end) in enumerate(segments):
        print(f"Reading segment {seg_id} (rows {row_start}:{row_end})...")

        seg_df = (
            pl.scan_parquet(input_path)
            .sort("received_time")
            .slice(row_start, row_end - row_start)
            .collect()
        )
        seg_df = seg_df.drop([c for c in cols_to_drop if c in seg_df.columns])

        start_ns = seg_df["received_time"].min().timestamp() * 1e9
        end_ns = seg_df["received_time"].max().timestamp() * 1e9
        start_ceil_ns = int(math.ceil(start_ns / one_sec_ns) * one_sec_ns)
        end_ceil_ns = int(math.ceil(end_ns / one_sec_ns) * one_sec_ns)

        grid = pl.DataFrame({
            "received_time": pl.Series(
                range(start_ceil_ns, end_ceil_ns + one_sec_ns, one_sec_ns),
                dtype=pl.Int64,
            ).cast(pl.Datetime("ns"))
        })

        seg_df = seg_df.sort("received_time")
        merged = grid.join_asof(seg_df, on="received_time", strategy="backward")

        print(f"  Segment {seg_id}: {len(merged)} rows")
        resampled_parts.append(merged)
        del seg_df

    # --- Step 3: Concatenate, add mid price, rename, and write ---
    print("Concatenating segments...")
    result = pl.concat(resampled_parts).sort("received_time")
    del resampled_parts

    # Add mid price (adjust column names if yours differ)
    if "best_bid_price" in result.columns and "best_ask_price" in result.columns:
        result = result.with_columns(
            ((pl.col("best_bid_price") + pl.col("best_ask_price")) / 2).alias("mid_price")
        )
    else:
        # Try common L2 naming: bids[0].price / asks[0].price style — adapt as needed
        bid_col = next((c for c in result.columns if "bid" in c.lower() and "price" in c.lower()), None)
        ask_col = next((c for c in result.columns if "ask" in c.lower() and "price" in c.lower()), None)
        if bid_col and ask_col:
            result = result.with_columns(
                ((pl.col(bid_col) + pl.col(ask_col)) / 2).alias("mid_price")
            )
        else:
            print(f"  Warning: could not find bid/ask price columns for mid_price. Columns: {result.columns}")

    rename_map = {col: f"{prefix}_{col}" for col in result.columns if col != "received_time"}
    result = result.rename(rename_map)

    result.write_parquet(output_path)
    print(f"Wrote {len(result)} rows to {output_path}")


if __name__ == "__main__":
    process_orderbook_data(
        input_file="/tmp/BTC_USDT_PERP.parquet",
        output_file="/tmp/1s_downsampled_BTC_USDT_PERP.parquet",
        prefix="p",
    )
