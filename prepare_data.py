#!/usr/bin/env python3
"""
월별 위성 CSV + KODC 정선해양 CSV  ->  ocean_predictor.py가 읽는 하나의 ts_data.csv

사용 예
-------
python prepare_data.py 저장소폴더            # 저장소폴더 안의 'N월_*.csv'와 '정선해양*.csv'를 읽음
python prepare_data.py 저장소폴더 -o data/ts_data.csv

출력 열: time, depth, lat, lon, temp, sal, source
  - 월별 파일 중 10열짜리(염분 5열 + 수온 5열)는 염분과 수온의 관측 위치·날짜·수심이 서로 달라서
    행을 나눠 담습니다(한쪽 값은 비어 있음). 예측 프로그램은 수온과 염분을 따로 학습하므로 문제없습니다.
  - KODC 파일은 위쪽 설명 줄을 건너뛰고 '관측일시' 줄을 머리글로 읽습니다.
  - 물리적으로 불가능한 값은 비웁니다(수온 -2~35, 염분 20~40). 비운 개수는 아래에 출력합니다.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

TEMP_RANGE = (-2.0, 35.0)
SAL_RANGE = (20.0, 40.0)


def read_csv(path, **kw):
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path}: 인코딩을 읽지 못했습니다 (utf-8, cp949 시도)")


def monthly_frames(path):
    """월별 위성 CSV 하나 -> 통합 형식 DataFrame 리스트."""
    df = read_csv(path)
    cols = [str(c) for c in df.columns]
    src = f"위성_{os.path.basename(path).split('월')[0]}월"
    meta = ["time", "depth", "lat", "lon"]
    if len(cols) == 6 and "염분" in cols[4] and "수온" in cols[5]:
        # 날짜, 깊이, 위도, 경도, 염분, 수온
        out = df.iloc[:, [0, 1, 2, 3, 5, 4]].copy()
        out.columns = meta + ["temp", "sal"]
        return [out.assign(source=src)]
    if len(cols) == 10 and "염분" in cols[4] and "수온" in cols[9]:
        # 염분(날짜, 깊이, 위도, 경도, 값) + 수온(날짜, 깊이, 위도, 경도, 값): 서로 다른 관측점
        sal = df.iloc[:, 0:5].copy()
        sal.columns = meta + ["sal"]
        sal["temp"] = np.nan
        tmp = df.iloc[:, 5:10].copy()
        tmp.columns = meta + ["temp"]
        tmp["sal"] = np.nan
        return [sal.assign(source=src), tmp.assign(source=src)]
    print(f"[건너뜀] {os.path.basename(path)}: 예상한 열 구성이 아닙니다 {cols}", file=sys.stderr)
    return []


def kodc_frame(path):
    """KODC 정선해양 관측자료: 설명 줄을 건너뛰고 '관측일시' 줄을 머리글로 사용."""
    header_row = None
    for enc in ("utf-8-sig", "cp949"):
        try:
            with open(path, encoding=enc) as f:
                for i, line in enumerate(f):
                    if "관측일시" in line:
                        header_row = i
                        break
            break
        except UnicodeDecodeError:
            continue
    if header_row is None:
        print(f"[건너뜀] {os.path.basename(path)}: '관측일시' 머리글 줄을 찾지 못했습니다.", file=sys.stderr)
        return None
    df = read_csv(path, header=header_row)
    need = ["위도", "경도", "관측일시(KST)", "관측수심(m)", "수온(°C)", "염분(psu)"]
    if any(c not in df.columns for c in need):
        print(f"[건너뜀] {os.path.basename(path)}: 필요한 열 {need} 중 일부가 없습니다.", file=sys.stderr)
        return None
    out = pd.DataFrame({
        "time": df["관측일시(KST)"], "depth": df["관측수심(m)"], "lat": df["위도"],
        "lon": df["경도"], "temp": df["수온(°C)"], "sal": df["염분(psu)"],
    })
    out["source"] = "KODC"
    return out


def build(folder, out, verbose=True):
    """folder의 CSV들을 읽어 out으로 저장하고, 요약 문장 리스트를 반환한다."""
    frames = []
    for p in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        name = os.path.basename(p)
        if re.match(r"^\d+월_", name):
            frames += monthly_frames(p)
        elif "정선해양" in name or "KODC" in name.upper():
            f = kodc_frame(p)
            if f is not None:
                frames.append(f)
    if not frames:
        raise ValueError("읽을 수 있는 CSV가 없습니다. (파일명: 'N월_*.csv' 또는 '정선해양*.csv')")

    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    for c in ("depth", "lat", "lon", "temp", "sal"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # 값 범위 점검: 불가능한 값은 비움
    bad_t = (df["temp"] < TEMP_RANGE[0]) | (df["temp"] > TEMP_RANGE[1])
    bad_s = (df["sal"] < SAL_RANGE[0]) | (df["sal"] > SAL_RANGE[1])
    df.loc[bad_t, "temp"] = np.nan
    df.loc[bad_s, "sal"] = np.nan

    # 위치·시간이 없거나 값이 둘 다 비면 제거
    ok = df["lat"].between(-90, 90) & df["lon"].between(-180, 180) & df["time"].notna()
    ok &= df[["temp", "sal"]].notna().any(axis=1)
    dropped = int((~ok).sum())
    df = df[ok].drop_duplicates().sort_values(["time", "lat", "lon", "depth"]).reset_index(drop=True)
    df["time"] = df["time"].dt.strftime("%Y-%m-%d")
    df[["time", "depth", "lat", "lon", "temp", "sal", "source"]].to_csv(out, index=False, encoding="utf-8")

    yrs = pd.to_datetime(df["time"]).dt.year.value_counts().sort_index()
    lines = [
        f"읽은 행 {n0} -> 저장 {len(df)}행  ({out})",
        f"  제거: 위치·시간·값이 없는 행 {dropped}개, 값이 불가능해 비운 것 수온 {int(bad_t.sum())}개 / 염분 {int(bad_s.sum())}개",
        f"  수온 유효 {int(df['temp'].notna().sum())}행, 염분 유효 {int(df['sal'].notna().sum())}행",
        "  연도별 행수: " + ", ".join(f"{int(y)}년 {n}" for y, n in yrs.items()),
        "  출처별 행수: " + ", ".join(f"{k} {v}" for k, v in df["source"].value_counts().items()),
    ]
    if verbose:
        print("\n".join(lines))
    return lines


def main():
    ap = argparse.ArgumentParser(description="월별 위성 CSV + KODC CSV를 하나로 통합")
    ap.add_argument("folder", help="CSV 파일들이 있는 폴더")
    ap.add_argument("-o", "--out", default="ts_data.csv", help="출력 파일 (기본 ts_data.csv)")
    args = ap.parse_args()
    try:
        build(args.folder, args.out)
    except ValueError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
