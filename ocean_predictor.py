#!/usr/bin/env python3
"""
위성 사진 -> 위치 파악 -> 수온/염분 예측

사용 예
-------
# 1) 지리참조된 위성 사진(GeoTIFF): 사진 범위 전체의 수온·염분 지도를 예측
python ocean_predictor.py --image scene.tif --data ts_data.csv --depth 10 --date 2026-11-14

# 2) 위치 정보가 없는 사진(지도 캡처 등): 기준점 3개 이상으로 위치를 계산
#    --gcp 픽셀X 픽셀Y 위도 경도   (도시, 섬, 곶 등 지도에서 위치를 아는 지점)
python ocean_predictor.py --image capture.png --data ts_data.csv --mask-land \\
    --gcp 387 79 37.5665 126.9780  --gcp 717 514 35.1796 129.0756  --gcp 289 591 34.8118 126.3922

# 3) 육지를 가리고 바다만 예측: --mask-land (어둡고 푸른 픽셀을 바다로 판정)

# 4) GPS 정보가 든 사진(EXIF) 또는 좌표 직접 입력: 한 지점 예측
python ocean_predictor.py --image photo.jpg --data ts_data.csv
python ocean_predictor.py --lat 35.2 --lon 127.5 --data ts_data.csv

# 5) 위경도 범위를 직접 지정 (사진이 위경도 격자일 때: 등간격 위경도 가정)
python ocean_predictor.py --image scene.png --bbox 126.0 34.0 129.0 37.0 --data ts_data.csv

데이터 CSV 컬럼 (이름은 아래 별칭 중 아무거나 가능)
  위도: lat, latitude          경도: lon, lng, longitude
  수온: temp, temperature, temp_c, sst
  염분: sal, salinity, psal, psu
  (선택) 수심/압력: depth, pres, pressure     (선택) 시간: time, date, datetime

위치 파악 순서 (먼저 걸리는 것을 사용)
  --gcp -> --bbox -> --lat/--lon -> 사진의 GeoTIFF 지리참조 -> 사진의 EXIF GPS
  사진 픽셀만 보고 위치를 인식하는 것은 신뢰할 수 없어서 좌표 정보나 기준점이 필요합니다.
"""
import argparse
import sys
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

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
# 1. 위치 파악: 픽셀 <-> 위경도 변환
# --------------------------------------------------------------------------
def _merc_y(lat):
    return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


class Georef:
    """픽셀(x, y) <-> (lat, lon) 변환.

    kind='linear'   : 위경도가 픽셀에 비례 (GeoTIFF, --bbox)
    kind='mercator' : 웹 지도(메르카토르) 투영 (--gcp)
    내부적으로 u = ax*x + bx, v = ay*y + by 로 투영 좌표를 얻는다.
    """

    def __init__(self, W, H, kind, ax, bx, ay, by):
        self.W, self.H, self.kind = W, H, kind
        self.ax, self.bx, self.ay, self.by = ax, bx, ay, by

    def to_geo(self, x, y):
        u = self.ax * np.asarray(x, float) + self.bx
        v = self.ay * np.asarray(y, float) + self.by
        if self.kind == "linear":
            return v, u  # lat, lon
        lon = np.degrees(u)
        lat = np.degrees(2 * np.arctan(np.exp(v)) - np.pi / 2)
        return lat, lon

    def to_pixel(self, lat, lon):
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        if self.kind == "linear":
            u, v = lon, lat
        else:
            u, v = np.radians(lon), _merc_y(lat)
        return (u - self.bx) / self.ax, (v - self.by) / self.ay

    def bbox(self):
        lat0, lon0 = self.to_geo(0, 0)
        lat1, lon1 = self.to_geo(self.W, self.H)
        return (float(min(lon0, lon1)), float(min(lat0, lat1)),
                float(max(lon0, lon1)), float(max(lat0, lat1)))

    @classmethod
    def from_bbox(cls, W, H, bbox):
        lon_min, lat_min, lon_max, lat_max = bbox
        return cls(W, H, "linear", (lon_max - lon_min) / W, lon_min,
                   -(lat_max - lat_min) / H, lat_max)

    @classmethod
    def from_gcps(cls, W, H, gcps):
        """기준점 [(x, y, lat, lon), ...] -> 메르카토르 변환. (변환, 잔차 px, 축척비) 반환."""
        x, y, lat, lon = np.asarray(gcps, float).T
        u, v = np.radians(lon), _merc_y(lat)
        if np.ptp(u) == 0 or np.ptp(v) == 0 or np.ptp(x) == 0 or np.ptp(y) == 0:
            sys.exit("기준점이 한 줄 위에 있습니다. 가로·세로로 떨어진 지점을 고르세요.")
        px = np.polyfit(u, x, 1)  # x = px0*u + px1  (측정 오차가 있는 쪽이 픽셀)
        py = np.polyfit(v, y, 1)
        geo = cls(W, H, "mercator", 1 / px[0], -px[1] / px[0], 1 / py[0], -py[1] / py[0])
        xf, yf = geo.to_pixel(lat, lon)
        res = np.hypot(xf - x, yf - y)
        ratio = abs(px[0] / py[0])
        return geo, res, ratio


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
              "--gcp나 --bbox로 위치를 직접 지정하세요.", file=sys.stderr)
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
    """(Georef 또는 None, point 또는 None, 출처 설명) 반환."""
    img = Image.open(args.image) if args.image else None
    W, H = img.size if img is not None else (None, None)

    if args.gcp:
        if img is None:
            sys.exit("--gcp는 --image와 함께 써야 합니다.")
        if len(args.gcp) < 3:
            sys.exit(f"--gcp는 최소 3개가 필요합니다 (지금 {len(args.gcp)}개).")
        geo, res, ratio = Georef.from_gcps(W, H, args.gcp)
        print(f"[기준점] {len(args.gcp)}개로 변환 계산: 잔차 평균 {res.mean():.1f}px, 최대 {res.max():.1f}px, "
              f"가로/세로 축척비 {ratio:.2f}")
        if res.max() > 0.015 * max(W, H) or not 0.8 <= ratio <= 1.25:
            print("[경고] 기준점이 서로 잘 맞지 않습니다. 픽셀 위치나 위경도를 다시 확인하세요 "
                  "(웹 지도 캡처가 아니면 변환이 맞지 않을 수 있습니다).", file=sys.stderr)
        out_of = [i + 1 for i, g in enumerate(args.gcp) if not (0 <= g[0] <= W and 0 <= g[1] <= H)]
        if out_of:
            print(f"[경고] 기준점 {out_of}번의 픽셀 좌표가 사진 크기({W}x{H}) 밖입니다.", file=sys.stderr)
        return geo, None, "기준점(--gcp) 변환"

    if args.bbox:
        bb = tuple(args.bbox)
        if img is None:  # 사진이 없으면 가상 격자(위경도 비율에 맞춤)
            mid = np.radians((bb[1] + bb[3]) / 2)
            W = 1000
            H = int(np.clip(1000 * (bb[3] - bb[1]) / ((bb[2] - bb[0]) * np.cos(mid)), 200, 3000))
        return Georef.from_bbox(W, H, bb), None, "--bbox 수동 입력"

    if args.lat is not None and args.lon is not None:
        return None, (args.lat, args.lon), "--lat/--lon 수동 입력"

    if img is not None:
        ext = geotiff_extent(img)
        if ext:
            return Georef.from_bbox(W, H, ext), None, "GeoTIFF 지리참조"
        pt = exif_gps_point(img)
        if pt:
            return None, pt, "EXIF GPS"
        print("[알림] 사진에서 위치 정보를 찾지 못했습니다.", file=sys.stderr)
    sys.exit("위치를 알 수 없습니다. GeoTIFF/EXIF GPS가 있는 사진을 쓰거나 "
             "--gcp(기준점 3개 이상), --bbox, --lat/--lon을 지정하세요.")


# --------------------------------------------------------------------------
# 2. 영상 처리: 미리보기, 바다 판별
# --------------------------------------------------------------------------
def load_rgb(path, max_px=1500):
    """8비트 RGB 배열과 축소 간격(step)을 반환."""
    img = Image.open(path)
    if img.mode in ("P", "PA", "LA", "1", "CMYK"):
        img = img.convert("RGB")
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        a = arr.astype(float)
        lo, hi = np.nanpercentile(a, 2), np.nanpercentile(a, 98)
        arr = (np.clip((a - lo) / (hi - lo + 1e-12), 0, 1) * 255).astype(np.uint8)
    step = max(1, int(np.ceil(max(arr.shape[:2]) / max_px)))
    return arr[::step, ::step], step


def water_mask(rgb, max_brightness=110):
    """바다(깊고 푸른 물)를 True로. 어둡고 푸른 픽셀 기준이며 얕은 연안·갯벌은 육지로 취급."""
    f = rgb.astype(float)
    R, G, B = f[..., 0], f[..., 1], f[..., 2]
    m = (B >= G - 3) & (B > R + 8) & (f.mean(axis=2) < max_brightness)
    k = np.ones((5, 5), bool)
    m = ndi.binary_opening(ndi.binary_closing(m, k), k)   # 글자·잡음 정리
    lab, n = ndi.label(m)
    if n:
        sizes = ndi.sum(m, lab, np.arange(1, n + 1))
        m = np.isin(lab, 1 + np.flatnonzero(sizes >= 0.002 * m.size))  # 작은 웅덩이 제거
    return m


def cell_fraction(mask, step, cx, cy, nx, ny):
    """격자 칸(가로 cx, 세로 cy 픽셀)별로 mask가 True인 비율."""
    csx, csy = cx / step, cy / step
    out = np.zeros((ny, nx))
    for j in range(ny):
        y0 = int(j * csy)
        y1 = max(int((j + 1) * csy), y0 + 1)
        for i in range(nx):
            x0 = int(i * csx)
            x1 = max(int((i + 1) * csx), x0 + 1)
            blk = mask[y0:y1, x0:x1]
            out[j, i] = blk.mean() if blk.size else 0.0
    return out


# --------------------------------------------------------------------------
# 3. 데이터 로딩
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
# 4. 예측 모델 (가우시안 프로세스)
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
# 5. 지도 그리기 (픽셀 좌표계에 그리고 눈금만 위경도로 표시)
# --------------------------------------------------------------------------
def _nice_step(span):
    for s in (0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10):
        if span / s <= 7:
            return s
    return 20


def set_geo_ticks(ax, geo):
    lon_min, lat_min, lon_max, lat_max = geo.bbox()
    mid_lat, mid_lon = (lat_min + lat_max) / 2, (lon_min + lon_max) / 2
    s = _nice_step(lon_max - lon_min)
    lons = np.arange(np.ceil(lon_min / s) * s, lon_max + 1e-9, s)
    xs, _ = geo.to_pixel(np.full_like(lons, mid_lat), lons)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{abs(round(v, 4)):g}°{'E' if v >= 0 else 'W'}" for v in lons], fontsize=8)
    s = _nice_step(lat_max - lat_min)
    lats = np.arange(np.ceil(lat_min / s) * s, lat_max + 1e-9, s)
    _, ys = geo.to_pixel(lats, np.full_like(lats, mid_lon))
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{abs(round(v, 4)):g}°{'N' if v >= 0 else 'S'}" for v in lats], fontsize=8)


def setup_korean_font():
    import matplotlib
    from matplotlib import font_manager
    for f in ("NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "Noto Sans KR"):
        if any(f == x.name for x in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = f
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def plot_maps(geo, rgb, step, grids, data, gcps, wmask, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    W, H = geo.W, geo.H
    extent = (0, W, H, 0)
    names = {"temp": ("수온 (°C)", "viridis"), "sal": ("염분 (PSU)", "plasma")}
    panels = (["image"] if rgb is not None else []) + list(grids)
    fig, axes = plt.subplots(1, len(panels), squeeze=False,
                             figsize=(5.4 * len(panels), 5.4 * H / W + 1.3))
    axes = axes[0]

    ox, oy = geo.to_pixel(data["lat"].to_numpy(), data["lon"].to_numpy())
    inside = (ox >= 0) & (ox <= W) & (oy >= 0) & (oy <= H)

    for ax, name in zip(axes, panels):
        if name == "image":
            ax.imshow(rgb, extent=extent)
            ax.set_title("위성 사진" + (" (노랑선: 바다 경계)" if wmask is not None else ""))
            if wmask is not None:
                xs = (np.arange(wmask.shape[1]) + 0.5) * step
                ys = (np.arange(wmask.shape[0]) + 0.5) * step
                ax.contour(xs, ys, wmask.astype(float), levels=[0.5], colors="yellow", linewidths=0.7)
            if gcps:
                g = np.asarray(gcps, float)
                ax.plot(g[:, 0], g[:, 1], "+", color="cyan", ms=12, mew=2, label="기준점")
                ax.legend(loc="lower right", fontsize=8)
        else:
            label, cmap = names[name]
            mean, _ = grids[name]
            if rgb is not None:
                ax.imshow(rgb, extent=extent, alpha=0.55)
            im = ax.imshow(np.ma.masked_invalid(mean), extent=extent, cmap=cmap,
                           interpolation="nearest", alpha=0.92 if rgb is not None else 1.0)
            plt.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)
            ax.scatter(ox[inside], oy[inside], s=8, c="white", edgecolors="k",
                       linewidths=0.4, label="관측점")
            ax.set_title(f"예측 {label}")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect("equal")
        set_geo_ticks(ax, geo)
    if len(panels) > 1:
        axes[-1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 6. 메인
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="위성 사진 위치 기반 수온·염분 예측")
    ap.add_argument("--image", help="위성 사진 (GeoTIFF 권장, 지도 캡처는 --gcp 필요)")
    ap.add_argument("--data", required=True, help="수온·염분 CSV")
    ap.add_argument("--gcp", type=float, nargs=4, action="append",
                    metavar=("X", "Y", "LAT", "LON"),
                    help="기준점: 사진 픽셀 X Y와 그 지점의 위도 경도. 3개 이상, 여러 번 사용")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                    help="사진 범위 직접 지정(등간격 위경도 가정)")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--mask-land", action="store_true", help="육지를 가리고 바다만 예측 (사진 필요)")
    ap.add_argument("--water-brightness", type=float, default=110,
                    help="바다로 볼 최대 밝기(0~255). 얕은 바다가 빠지면 올리고 육지가 섞이면 내림")
    ap.add_argument("--depth", type=float, help="예측 수심(m). 기본: 데이터의 최소 수심(표층)")
    ap.add_argument("--date", help="예측 날짜 YYYY-MM-DD (데이터에 시간 컬럼이 있을 때 계절성 반영)")
    ap.add_argument("--grid", type=int, default=60, help="긴 변 기준 격자 칸 수(기본 60)")
    ap.add_argument("--out", default="prediction", help="출력 파일 접두어")
    ap.add_argument("--no-cv", action="store_true", help="교차검증 생략")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    geo, point, how = locate(args)
    data = load_data(args.data)
    if geo is not None:
        lon0, lat0, lon1, lat1 = geo.bbox()
        print(f"[위치] {how}: 범위 lon {lon0:.3f}~{lon1:.3f}, lat {lat0:.3f}~{lat1:.3f}")
    else:
        print(f"[위치] {how}: lat {point[0]:.4f}, lon {point[1]:.4f}")
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
        q = pd.DataFrame({"lat": np.asarray(lats, float), "lon": np.asarray(lons, float)})
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

    # --- 범위(지도) 예측: 사진 픽셀 격자 위에서 예측
    W, H = geo.W, geo.H
    cell = max(W, H) / args.grid
    nx, ny = max(1, int(round(W / cell))), max(1, int(round(H / cell)))
    cx, cy = W / nx, H / ny          # 칸이 사진 크기에 딱 맞게 나뉘도록
    xs = (np.arange(nx) + 0.5) * cx
    ys = (np.arange(ny) + 0.5) * cy
    X, Y = np.meshgrid(xs, ys)
    LAT, LON = geo.to_geo(X, Y)

    rgb, step = (load_rgb(args.image) if args.image else (None, 1))
    wmask, water_cell = None, np.ones((ny, nx), bool)
    if args.mask_land:
        if rgb is None:
            sys.exit("--mask-land는 --image가 필요합니다.")
        wmask = water_mask(rgb, args.water_brightness)
        water_cell = cell_fraction(wmask, step, cx, cy, nx, ny) >= 0.5
        n_water = int(water_cell.sum())
        print(f"[육지 가리기] 격자 {nx * ny}칸 중 바다 {n_water}칸 ({100 * n_water / (nx * ny):.0f}%)")
        if n_water == 0:
            sys.exit("바다로 판정된 영역이 없습니다. --water-brightness 값을 올려 보세요.")

    sel = water_cell.ravel()
    Xq = make_query(LAT.ravel()[sel], LON.ravel()[sel])
    grids = {}
    table = pd.DataFrame({"x_px": X.ravel(), "y_px": Y.ravel(), "lat": LAT.ravel(),
                          "lon": LON.ravel(), "water": water_cell.ravel()})
    for key, m in models.items():
        mu, sd = m.predict(Xq)
        mean_full = np.full(nx * ny, np.nan)
        std_full = np.full(nx * ny, np.nan)
        mean_full[sel], std_full[sel] = mu, sd
        grids[key] = (mean_full.reshape(ny, nx), std_full.reshape(ny, nx))
        table[f"{key}_pred"], table[f"{key}_std"] = mean_full, std_full
        print(f"[예측] {label[key]}: 범위 {mu.min():.2f} ~ {mu.max():.2f}, 평균 불확실성 ±{sd.mean():.2f}")

    ox, oy = geo.to_pixel(data["lat"].to_numpy(), data["lon"].to_numpy())
    n_in = int(((ox >= 0) & (ox <= W) & (oy >= 0) & (oy <= H)).sum())
    print(f"[참고] 사진 범위 안의 관측점 {n_in}개"
          + (" — 0개면 주변 관측점으로 외삽한 값이라 신뢰도가 낮습니다." if n_in == 0 else ""))

    setup_korean_font()
    table.to_csv(f"{args.out}_grid.csv", index=False)
    plot_maps(geo, rgb, step, grids, data, args.gcp, wmask, f"{args.out}_map.png")
    print(f"[저장] {args.out}_grid.csv, {args.out}_map.png")


if __name__ == "__main__":
    main()
