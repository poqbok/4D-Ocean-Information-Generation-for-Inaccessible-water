"""
Render 배포용 웹 앱: 위성 사진 + 수온·염분 CSV를 올리면 ocean_predictor.py로 예측하고
결과 지도와 로그를 페이지에 보여준다.

로컬 실행:  python app.py   ->  http://localhost:5000
Render:     gunicorn app:app
"""
import base64
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from flask import Flask, render_template_string, request

BASE = Path(__file__).parent
DEFAULT_CSV = BASE / "data" / "ts_data.csv"   # 서버에 기본 데이터를 두고 싶으면 여기에 넣기
TIMEOUT_SEC = 150

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024  # 업로드 합계 40MB 제한

PAGE = """
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>위성 사진 수온·염분 예측</title>
<style>
  body { font-family: system-ui, "Noto Sans KR", sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
  h1 { font-size: 1.4rem; }
  fieldset { border: 1px solid #ccc; border-radius: 8px; margin: 1rem 0; padding: .8rem 1rem; }
  legend { font-weight: 600; padding: 0 .4rem; }
  label { display: block; margin: .5rem 0 .15rem; font-size: .9rem; }
  input[type=number], input[type=text], input[type=date] { width: 9rem; padding: .3rem; }
  .row { display: flex; gap: 1rem; flex-wrap: wrap; }
  button { padding: .6rem 1.4rem; font-size: 1rem; cursor: pointer; }
  pre { background: #f4f4f4; padding: .8rem; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; }
  .err { color: #b00020; }
  img { max-width: 100%; }
  small { color: #666; }
</style>
</head>
<body>
<h1>위성 사진 → 수온·염분 예측</h1>
<form method="post" enctype="multipart/form-data">
  <fieldset>
    <legend>1. 파일</legend>
    <label>위성 사진 (GeoTIFF 권장, EXIF GPS 사진도 가능)</label>
    <input type="file" name="image" accept=".tif,.tiff,.jpg,.jpeg,.png">
    <label>수온·염분 CSV {% if has_default %}<small>(비우면 서버 기본 데이터 사용)</small>{% endif %}</label>
    <input type="file" name="data" accept=".csv" {% if not has_default %}required{% endif %}>
  </fieldset>
  <fieldset>
    <legend>2. 위치 (사진에 위치 정보가 없을 때만)</legend>
    <div class="row">
      <div><label>경도 최소</label><input type="number" step="any" name="lon_min"></div>
      <div><label>위도 최소</label><input type="number" step="any" name="lat_min"></div>
      <div><label>경도 최대</label><input type="number" step="any" name="lon_max"></div>
      <div><label>위도 최대</label><input type="number" step="any" name="lat_max"></div>
    </div>
    <small>또는 한 지점:</small>
    <div class="row">
      <div><label>위도</label><input type="number" step="any" name="lat"></div>
      <div><label>경도</label><input type="number" step="any" name="lon"></div>
    </div>
  </fieldset>
  <fieldset>
    <legend>3. 조건 (선택)</legend>
    <div class="row">
      <div><label>수심 (m)</label><input type="number" step="any" name="depth"></div>
      <div><label>날짜</label><input type="date" name="date"></div>
    </div>
  </fieldset>
  <button type="submit">예측하기</button>
</form>

{% if error %}<p class="err"><b>오류:</b> {{ error }}</p>{% endif %}
{% if log %}<h2>결과</h2><pre>{{ log }}</pre>{% endif %}
{% if png %}<img alt="예측 지도" src="data:image/png;base64,{{ png }}">{% endif %}
</body>
</html>
"""


def _form_float(name):
    v = request.form.get(name, "").strip()
    return v if v else None


@app.route("/health")
def health():
    return "ok"


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = dict(has_default=DEFAULT_CSV.exists(), error=None, log=None, png=None)
    if request.method == "GET":
        return render_template_string(PAGE, **ctx)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cmd = [sys.executable, str(BASE / "ocean_predictor.py"), "--out", str(tmp / "result")]

        # 데이터
        data_file = request.files.get("data")
        if data_file and data_file.filename:
            data_path = tmp / "data.csv"
            data_file.save(data_path)
        elif DEFAULT_CSV.exists():
            data_path = DEFAULT_CSV
        else:
            ctx["error"] = "수온·염분 CSV를 올려 주세요."
            return render_template_string(PAGE, **ctx)
        cmd += ["--data", str(data_path)]

        # 사진
        img_file = request.files.get("image")
        if img_file and img_file.filename:
            img_path = tmp / "upload_image"
            img_file.save(img_path)
            cmd += ["--image", str(img_path)]

        # 위치 (사진에 위치 정보가 없을 때 사용됨)
        bbox = [_form_float(k) for k in ("lon_min", "lat_min", "lon_max", "lat_max")]
        try:
            if all(bbox):
                cmd += ["--bbox"] + [str(float(v)) for v in bbox]
            lat, lon = _form_float("lat"), _form_float("lon")
            if lat and lon:
                cmd += ["--lat", str(float(lat)), "--lon", str(float(lon))]
            depth = _form_float("depth")
            if depth:
                cmd += ["--depth", str(float(depth))]
        except ValueError:
            ctx["error"] = "숫자 입력 형식이 올바르지 않습니다."
            return render_template_string(PAGE, **ctx)
        date = _form_float("date")
        if date:
            cmd += ["--date", date]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=TIMEOUT_SEC, cwd=tmp)
        except subprocess.TimeoutExpired:
            ctx["error"] = f"{TIMEOUT_SEC}초 안에 끝나지 않았습니다. 데이터 행 수나 사진 크기를 줄여 보세요."
            return render_template_string(PAGE, **ctx)

        log = (proc.stdout + proc.stderr).strip()
        if proc.returncode != 0:
            ctx["error"] = log or "예측 중 오류가 발생했습니다."
            return render_template_string(PAGE, **ctx)

        ctx["log"] = log
        png = tmp / "result_map.png"
        if png.exists():
            ctx["png"] = base64.b64encode(png.read_bytes()).decode()
        return render_template_string(PAGE, **ctx)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
