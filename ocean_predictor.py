#!/usr/bin/env python3
"""
위성 사진 -> 위치 파악 -> 수온/염분 예측

사용 예
-------
# 1) 지리참조된 위성 사진(GeoTIFF)을 넣으면 사진 범위 전체의 수온·염분 지도를 예측
python ocean_predictor.py --image scene.tif --data ts_data.csv --depth 10 --date 2026-11-14

# 2) GPS 정보가 든 사진(EXIF) -> 그 한 지점 예측
python ocean_predictor.py --image photo.jpg --data ts_data.csv

# 3) 사진에 위치 정보가 없을 때: 범위나 좌표를 직접 지정
python ocean_predictor.py --image scene.png --bbox 126.0 34.0 129.0 37.0 --data ts_data.csv
python ocean_predictor.py --lat 35.2 --lon 127.5 --data ts_data.csv

데이터 CSV 컬럼 (이름은 아래 별칭 중 아무거나 가능)
  위도: lat, latitude          경도: lon, lng, longitude
  수온: temp, temperature, temp_c, sst
  염분: sal, salinity, psal, psu
  (선택) 수심/압력: depth, pres, pressure     (선택) 시간: time, date, datetime

위치 파악 방식
  사진 픽셀만 보고 "여기가 어디"라고 인식하는 것은 신뢰할 수 없어서, 사진에 들어 있는
  좌표 정보를 읽습니다. 순서: GeoTIFF 지리참조 -> EXIF GPS -> --bbox/--lat/--lon 직접 입력.
"""
import argparse
import sys
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from PIL.ExifTags import GPSTAGS  # noqa: F401  (EXIF GPS 태그 참조용)

from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

Image.MAX_IMAGE_PIXELS = None  # 큰 위성 영상 허용

COLUMN_ALIASES = {
    "lat": ["lat", "latitude", "위도"],
    "lon": ["lon", "lng", "long", "longitude", "경도"],
    "temp": ["temp", "temperature", "temp_c", "sst", "수온"],
    "sal": ["sal", "salinity", "psal", "psu", "염분"],
    "depth": ["depth", "pres", "pressure", "수심", "압력"],
    "time": ["time", "date", "datetime", "시간", "날짜"],
}

MAX_TRAIN_ROWS = 1500  # GP는 O(n^3)이라 그 이상이면 무작위로 줄임


# --------------------------------------------------------------------------
# 1. 위치 파악
# --------------------------------------------------------------------------
def geotiff_extent(img):
    """GeoTIFF 태그에서 (lon_min, lat_min, lon_max, lat_max)를 읽는다. 없으면 None."""
    tags = getattr(img, "tag_v2", None)
    if tags is None:
        return None
    scale = tags.get(33550)      # ModelPixelScaleTag
    tie = tags.get(33922)        # ModelTiepointTag
    if not scale or not tie or len(tie) < 6:
        return None
    sx, sy = float(scale[0]), float(scale[1])
    i, j, x, y = float(tie[0]), float(tie[1]), float(tie[3]), float(tie[4])
    w, h = img.size
    lon_min = x - i * sx
    lat_max = y + j * sy
    lon_max = lon_min + w * sx
    lat_min = lat_max - h * sy
    ok = -180 <= lon_min <= 180 and -180 <= lon_max <= 180 \
        and -90 <= lat_min <= 90 and -90 <= lat_max <= 90
    if not ok:
        print("[경고] GeoTIFF가 위경도 좌표계가 아닙니다(UTM 등). "
              "--bbox로 위경도 범위를 직접 지정하세요.", file=sys.stderr)
        return None
    return (lon_min, lat_min, lon_max, lat_max)


def exif_gps_point(img):
    """EXIF GPS에서 (lat, lon)을 읽는다. 없으면 None."""
    try:
        gps = img.getexif().get_ifd(0x8825)
    except Exception:
        return None
    if not gps or 2 not in gps or 4 not in gps:
        return None

    def to_deg(dms):
        d, m, s = (float(v) for v in dms)
        return d + m / 60 + s / 3600

    lat = to_deg(gps[2]) * (-1 if gps.get(1) == "S" else 1)
    lon = to_deg(gps[4]) * (-1 if gps.get(3) == "W" else 1)
    return lat, lon


def locate(args):
    """(bbox 또는 None, point 또는 None, 출처 설명) 반환."""
    img = None
    if args.image:
        img = Image.open(args.image)
        ext = geotiff_extent(img)
        if ext:
            return ext, None, "GeoTIFF 지리참조"
        pt = exif_gps_point(img)
        if pt:
            return None, pt, "EXIF GPS"
        print("[알림] 사진에서 위치 정보를 찾지 못했습니다.", file=sys.stderr)
    if args.bbox:
        return tuple(args.bbox), None, "--bbox 수동 입력"
    if args.lat is not None and args.lon is not None:
        return None, (args.lat, args.lon), "--lat/--lon 수동 입력"
    sys.exit("위치를 알 수 없습니다. GeoTIFF/EXIF GPS가 있는 사진을 쓰거나 "
             "--bbox 또는 --lat/--lon을 지정하세요.")


# --------------------------------------------------------------------------
# 2. 데이터 로딩
# --------------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    low = {c.lower().strip(): c for c in df.columns}
    out = {}
    for key, names in COLUMN_ALIASES.items():
        for n in names:
            if n.lower() in low:
                out[key] = df[low[n.lower()]]
                break
    for need in ("lat", "lon"):
        if need not in out:
            sys.exit(f"CSV에 {need} 컬럼이 없습니다. 컬럼: {list(df.columns)}")
    if "temp" not in out and "sal" not in out:
        sys.exit("CSV에 수온(temp)이나 염분(sal) 컬럼이 필요합니다.")
    data = pd.DataFrame(out)
    if "time" in data:
        data["time"] = pd.to_datetime(data["time"], errors="coerce")
    data = data.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return data


def build_features(df, use_depth, use_season, depth=None, doy=None):
    """모델 입력 행렬 생성. df는 lat/lon(/depth/time) 포함."""
    cols = [df["lat"].to_numpy(float), df["lon"].to_numpy(float)]
    if use_depth:
        d = df["depth"].to_numpy(float) if depth is None else np.full(len(df), depth, float)
        cols.append(np.log1p(np.clip(d, 0, None)))
    if use_season:
        if doy is None:
            doy_arr = df["time"].dt.dayofyear.to_numpy(float)
        else:
            doy_arr = np.full(len(df), doy, float)
        ang = 2 * np.pi * doy_arr / 365.25
        cols += [np.sin(ang), np.cos(ang)]
    return np.column_stack(cols)


# --------------------------------------------------------------------------
# 3. 예측 모델 (가우시안 프로세스)
# --------------------------------------------------------------------------
class OceanModel:
    """변수 하나(수온 또는 염분)를 위치(+수심+계절)로 예측하는 GP 회귀."""

    def __init__(self, name):
        self.name = name

    def _new_gp(self, n_feat):
        kernel = C(1.0, (1e-2, 1e2)) * RBF([1.0] * n_feat, (0.05, 50.0)) \
            + WhiteKernel(0.05, (1e-4, 1.0))
        return GaussianProcessRegressor(kernel, normalize_y=True,
                                        n_restarts_optimizer=2, random_state=0)

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.gp = self._new_gp(X.shape[1]).fit(Xs, y)
        self._Xs, self._y = Xs, y
        return self

    def predict(self, X):
        mean, std = self.gp.predict(self.scaler.transform(X), return_std=True)
        return mean, std

    def cv_rmse(self, k=5):
        n = len(self._y)
        if n < 10:
            return None
        errs = []
        for tr, te in KFold(min(k, n), shuffle=True, random_state=0).split(self._Xs):
            gp = clone(self.gp.kernel_)
            m = GaussianProcessRegressor(gp, normalize_y=True, optimizer=None).fit(
                self._Xs[tr], self._y[tr])
            errs.append(self._y[te] - m.predict(self._Xs[te]))
        return float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def nearest_data_km(data, lat, lon):
    return float(np.min(haversine_km(lat, lon, data["lat"].to_numpy(), data["lon"].to_numpy())))


# --------------------------------------------------------------------------
# 4. 지도 그리기
# --------------------------------------------------------------------------
def image_preview(path, max_px=1500):
    img = Image.open(path)
    arr = np.asarray(img).astype(float)
    if arr.ndim == 3:
        arr = arr[..., :3]
    step = max(1, max(arr.shape[:2]) // max_px)
    arr = arr[::step, ::step]
    lo, hi = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
    arr = np.clip((arr - lo) / (hi - lo + 1e-12), 0, 1)
    return arr


def plot_maps(args, bbox, grids, data, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon_min, lat_min, lon_max, lat_max = bbox
    extent = (lon_min, lon_max, lat_min, lat_max)
    names = {"temp": ("수온 (°C)", "viridis"), "sal": ("염분 (PSU)", "plasma")}
    n = 1 + len(grids)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.8), squeeze=False)
    axes = axes[0]

    ax = axes[0]
    if args.image:
        try:
            ax.imshow(image_preview(args.image), extent=extent, origin="upper",
                      cmap="gray" if Image.open(args.image).mode in ("L", "I;16", "I", "F") else None)
        except Exception as e:  # 미리보기가 안 되어도 예측 지도는 그린다
            ax.text(0.5, 0.5, f"미리보기 실패\n{e}", transform=ax.transAxes, ha="center")
    ax.set_title("위성 사진")

    for ax, (key, (mean, std)) in zip(axes[1:], grids.items()):
        label, cmap = names[key]
        im = ax.imshow(mean, extent=extent, origin="lower", cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, label=label)
        ax.scatter(data["lon"], data["lat"], s=8, c="white", edgecolors="k",
                   linewidths=0.4, label="관측점")
        ax.set_title(f"예측 {label}")
    for ax in axes:
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel("경도")
        ax.set_ylabel("위도")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def setup_korean_font():
    import matplotlib
    from matplotlib import font_manager
    for f in ("NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "Noto Sans KR"):
        if any(f == x.name for x in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = f
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------------------------------
# 5. 메인
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="위성 사진 위치 기반 수온·염분 예측")
    ap.add_argument("--image", help="위성 사진 (GeoTIFF 권장, EXIF GPS 사진도 가능)")
    ap.add_argument("--data", required=True, help="수온·염분 CSV")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                    help="위치 정보가 없는 사진용 수동 범위")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--depth", type=float, help="예측 수심(m). 기본: 데이터의 최소 수심(표층)")
    ap.add_argument("--date", help="예측 날짜 YYYY-MM-DD (데이터에 시간 컬럼이 있을 때 계절성 반영)")
    ap.add_argument("--grid", type=int, default=40, help="지도 격자 해상도(기본 40x40)")
    ap.add_argument("--out", default="prediction", help="출력 파일 접두어")
    ap.add_argument("--no-cv", action="store_true", help="교차검증 생략")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    bbox, point, how = locate(args)
    data = load_data(args.data)
    print(f"[위치] {how}: " + (f"범위 lon {bbox[0]:.3f}~{bbox[2]:.3f}, lat {bbox[1]:.3f}~{bbox[3]:.3f}"
                              if bbox else f"lat {point[0]:.4f}, lon {point[1]:.4f}"))
    print(f"[데이터] {len(data)}행 로드")

    use_depth = "depth" in data and data["depth"].nunique() > 1
    use_season = "time" in data and data["time"].notna().any() and args.date is not None
    if "time" in data and args.date is None:
        print("[알림] --date가 없어 시간(계절) 정보는 쓰지 않습니다.")
    doy = pd.Timestamp(args.date).dayofyear if args.date else None
    depth = args.depth
    if use_depth and depth is None:
        depth = float(data["depth"].min())
        print(f"[알림] --depth 미지정 -> 표층 {depth:g} m로 예측")
    if not use_depth and args.depth is not None:
        print("[알림] 데이터에 수심 변화가 없어 --depth는 무시됩니다.")

    if use_season:
        data = data.dropna(subset=["time"]).reset_index(drop=True)
    variables = [k for k in ("temp", "sal") if k in data]
    models = {}
    for key in variables:
        sub = data.dropna(subset=[key] + (["depth"] if use_depth else []))
        if len(sub) < 5:
            sys.exit(f"{key} 유효 데이터가 5개 미만이라 예측할 수 없습니다.")
        if len(sub) > MAX_TRAIN_ROWS:
            sub = sub.sample(MAX_TRAIN_ROWS, random_state=0)
            print(f"[알림] {key}: 학습 데이터를 {MAX_TRAIN_ROWS}행으로 무작위 축소")
        X = build_features(sub, use_depth, use_season)
        m = OceanModel(key).fit(X, sub[key].to_numpy(float))
        if not args.no_cv:
            m.rmse = m.cv_rmse()
        models[key] = m

    label = {"temp": "수온(°C)", "sal": "염분(PSU)"}
    for key, m in models.items():
        r = getattr(m, "rmse", None)
        if r is not None:
            print(f"[검증] {label[key]} 5-fold 교차검증 RMSE = {r:.3f}")

    def make_query(lats, lons):
        q = pd.DataFrame({"lat": lats, "lon": lons})
        return build_features(q, use_depth, use_season, depth=depth, doy=doy)

    # --- 한 지점 예측
    if point:
        lat, lon = point
        Xq = make_query([lat], [lon])
        d_km = nearest_data_km(data, lat, lon)
        print(f"\n[예측] lat {lat:.4f}, lon {lon:.4f}"
              + (f", 수심 {depth:g} m" if use_depth else "")
              + (f", {args.date}" if use_season else ""))
        rows = []
        for key, m in models.items():
            mu, sd = m.predict(Xq)
            print(f"  {label[key]}: {mu[0]:.3f} ± {sd[0]:.3f}")
            rows.append({"variable": key, "mean": mu[0], "std": sd[0]})
        print(f"  가장 가까운 관측점까지 {d_km:.1f} km")
        if d_km > 200:
            print("  [주의] 관측점에서 많이 멀어 신뢰하기 어렵습니다.")
        pd.DataFrame(rows).to_csv(f"{args.out}_point.csv", index=False)
        return

    # --- 범위(지도) 예측
    lon_min, lat_min, lon_max, lat_max = bbox
    gx = np.linspace(lon_min, lon_max, args.grid)
    gy = np.linspace(lat_min, lat_max, args.grid)
    GX, GY = np.meshgrid(gx, gy)
    Xq = make_query(GY.ravel(), GX.ravel())
    grids, table = {}, pd.DataFrame({"lat": GY.ravel(), "lon": GX.ravel()})
    for key, m in models.items():
        mu, sd = m.predict(Xq)
        grids[key] = (mu.reshape(GX.shape), sd.reshape(GX.shape))
        table[f"{key}_pred"] = mu
        table[f"{key}_std"] = sd
        print(f"[예측] {label[key]}: 범위 {mu.min():.2f} ~ {mu.max():.2f}, 평균 불확실성 ±{sd.mean():.2f}")

    inside = data[(data.lon.between(lon_min, lon_max)) & (data.lat.between(lat_min, lat_max))]
    print(f"[참고] 사진 범위 안의 관측점 {len(inside)}개"
          + (" — 0개면 주변 관측점으로 외삽한 값이라 신뢰도가 낮습니다." if len(inside) == 0 else ""))

    setup_korean_font()
    table.to_csv(f"{args.out}_grid.csv", index=False)
    plot_maps(args, bbox, grids, data, f"{args.out}_map.png")
    print(f"[저장] {args.out}_grid.csv, {args.out}_map.png")


if __name__ == "__main__":
    main()
