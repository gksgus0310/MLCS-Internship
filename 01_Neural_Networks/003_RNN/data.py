#==========================================#
# Title:  01-003 RNN & Regression — 데이터 준비
#
#         ohlcv(시가·고가·저가·종가·거래량)를 받아
#         LSTM 이 먹을 수 있는 (샘플, 시퀀스, 특징) 텐서로 만든다.
#
# Author: Hyun Han
# Date:   2026-09-09
#
# 이 파일에서 신경 쓴 것 세 가지 (전부 시계열이라서 생기는 함정):
#   1. 분할을 셔플하지 않는다        — 미래로 학습해 과거를 맞히면 누수다
#   2. 스케일러는 train 으로만 fit   — val/test 통계를 쓰면 그것도 누수다
#   3. 타깃을 '가격'이 아니라 '수익률'로 — 이유는 아래 주석에 길게 적었다
#==========================================#
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# 폴백용 공개 CSV (yfinance 가 막힌 환경에서 코드 검증용, 약 500일치)
FALLBACK_URL = ("https://raw.githubusercontent.com/plotly/datasets/master/"
                "finance-charts-apple.csv")


# ---------------------------------------------------------------- #
#  1. ohlcv 받아오기 (한 번 받으면 CSV 로 캐시)
# ---------------------------------------------------------------- #
def load_ohlcv(ticker="AAPL", start="2010-01-01", end="2025-01-01", force=False):
    """ohlcv 5개 컬럼을 가진 DataFrame 을 돌려준다 (index = 날짜)."""
    cache = DATA / f"{ticker}_ohlcv.csv"
    if cache.exists() and not force:
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        print(f"[data] 캐시 사용: {cache.name}  {len(df)}일  "
              f"({df.index[0].date()} ~ {df.index[-1].date()})")
        return df

    df = None
    try:
        import yfinance as yf
        raw = yf.download(ticker, start=start, end=end,
                          progress=False, auto_adjust=True)
        if len(raw) > 0:
            if isinstance(raw.columns, pd.MultiIndex):     # yfinance 신버전
                raw.columns = raw.columns.get_level_values(0)
            df = raw[OHLCV].copy()
            print(f"[data] yfinance 로 {ticker} {len(df)}일 수신")
    except Exception as e:
        print(f"[data] yfinance 실패 ({type(e).__name__}) — 폴백 CSV 사용")

    if df is None or len(df) == 0:
        raw = pd.read_csv(FALLBACK_URL, parse_dates=["Date"]).set_index("Date")
        df = raw[["AAPL.Open", "AAPL.High", "AAPL.Low",
                  "AAPL.Close", "AAPL.Volume"]].copy()
        df.columns = OHLCV
        print(f"[data] 폴백 CSV 사용: {len(df)}일 "
              f"(네트워크가 되는 곳에서 다시 돌리면 yfinance 로 더 긴 구간을 받는다)")

    df = df.dropna()
    df.to_csv(cache)
    print(f"[data] 캐시 저장: {cache.name}")
    return df


# ---------------------------------------------------------------- #
#  2. 특징 만들기
#
#  ohlcv 원값을 그대로 넣으면 안 되는 이유:
#  주가는 추세를 가진 비정상(non-stationary) 시계열이다. 학습 구간이 100달러대,
#  테스트 구간이 200달러대면 모델은 한 번도 본 적 없는 입력 범위를 받는다.
#  그래서 전부 '변화율'로 바꾼다. 변화율은 구간이 달라도 분포가 비슷하다.
# ---------------------------------------------------------------- #
def make_features(df):
    """ohlcv -> 정상성 있는 6개 특징. 첫 행은 차분 때문에 사라진다."""
    C = df["Close"].values
    feats = pd.DataFrame(index=df.index)
    feats["r_close"]  = np.log(df["Close"] / df["Close"].shift(1))   # 종가 수익률
    feats["r_open"]   = np.log(df["Open"]  / df["Close"].shift(1))   # 갭 (전일 종가 대비 시가)
    feats["hl_range"] = np.log(df["High"]  / df["Low"])              # 당일 변동폭
    feats["co"]       = np.log(df["Close"] / df["Open"])             # 장중 방향
    feats["r_volume"] = np.log(df["Volume"] / df["Volume"].shift(1)) # 거래량 변화
    feats["vol_20"]   = feats["r_close"].rolling(20).std()           # 최근 변동성
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
    close = pd.Series(C, index=df.index).loc[feats.index]
    return feats, close


# ---------------------------------------------------------------- #
#  3. 슬라이딩 윈도우
#
#  many-to-one   : 과거 SEQ 일  ->  다음 1일 수익률
#  many-to-many  : 과거 SEQ 일  ->  다음 HORIZON 일 수익률 (여러 스텝)
#  aligned m2m   : 과거 SEQ 일  ->  각 시점의 '다음날' 수익률 SEQ 개
# ---------------------------------------------------------------- #
def make_windows(X, y, seq_len, horizon=1, aligned=False):
    """(N, seq_len, F) 입력과 타깃을 만든다.

    aligned=True 면 타깃이 (N, seq_len) — 입력 각 시점마다 다음날 값.
    aligned=False 면 타깃이 (N, horizon) — 윈도우 뒤쪽 horizon 일.
    """
    xs, ys, idx = [], [], []
    last = len(X) - seq_len - (seq_len if aligned else horizon) + 1
    for i in range(last):
        xs.append(X[i:i + seq_len])
        if aligned:
            ys.append(y[i + 1:i + seq_len + 1])
        else:
            ys.append(y[i + seq_len:i + seq_len + horizon])
        idx.append(i + seq_len - 1)          # 예측 기준이 되는 마지막 관측일
    return np.array(xs, np.float32), np.array(ys, np.float32), np.array(idx)


# ---------------------------------------------------------------- #
#  4. 시간순 분할 + train 으로만 스케일링
# ---------------------------------------------------------------- #
def prepare(ticker="AAPL", seq_len=60, horizon=5,
            train_ratio=0.70, val_ratio=0.15, verbose=True):
    """학습에 바로 쓸 수 있는 dict 를 돌려준다."""
    df = load_ohlcv(ticker)
    feats, close = make_features(df)
    X_all = feats.values.astype(np.float32)
    y_all = feats["r_close"].values.astype(np.float32)      # 타깃 = 종가 로그수익률

    n = len(X_all)
    n_tr = int(n * train_ratio)
    n_va = int(n * (train_ratio + val_ratio))

    # 스케일러는 train 구간 통계로만 만든다
    mu, sd = X_all[:n_tr].mean(0), X_all[:n_tr].std(0) + 1e-8
    Xs = (X_all - mu) / sd
    y_sd = y_all[:n_tr].std() + 1e-8
    ys = y_all / y_sd                                       # 타깃도 스케일 맞춤

    out = {"close": close.values, "dates": feats.index,
           "y_sd": float(y_sd), "n_train": n_tr, "n_val": n_va,
           "feature_names": list(feats.columns)}

    for tag, aligned, H in [("m2o", False, 1),
                            ("m2m", False, horizon),
                            ("m2m_aligned", True, 1)]:
        Xw, yw, iw = make_windows(Xs, ys, seq_len, H, aligned)
        # 윈도우의 '마지막 관측일'이 어느 구간에 속하는지로 나눈다
        tr = iw < n_tr - seq_len
        va = (iw >= n_tr) & (iw < n_va - seq_len)
        te = iw >= n_va
        out[tag] = dict(
            train=(Xw[tr], yw[tr], iw[tr]),
            val=(Xw[va], yw[va], iw[va]),
            test=(Xw[te], yw[te], iw[te]))

    if verbose:
        print(f"[data] 특징 {X_all.shape[1]}개: {list(feats.columns)}")
        print(f"[data] 전체 {n}일 -> train {n_tr} / val {n_va-n_tr} / test {n-n_va} "
              f"(시간순, 셔플 안 함)")
        for tag in ("m2o", "m2m", "m2m_aligned"):
            a, b, c = (len(out[tag][k][0]) for k in ("train", "val", "test"))
            print(f"       {tag:<12} 윈도우  train {a} / val {b} / test {c}"
                  f"   타깃 shape {out[tag]['train'][1].shape[1:]}")
    return out


if __name__ == "__main__":
    d = prepare()
    print("\n확인용:", d["m2o"]["train"][0].shape, d["m2m"]["train"][1].shape)
