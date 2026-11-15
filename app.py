"""
Render 배포용 웹 앱: 위성 사진 + 수온·염분 CSV를 올리면 ocean_predictor.py로 예측하고
결과 지도와 로그를 페이지에 보여준다.

로컬 실행:  python app.py   ->  http://localhost:5000
Render:     gunicorn app:app
"""
import base64
import datetime
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from flask import Flask, render_template_string, request

BASE = Path(__file__).parent
DEFAULT_CSV = BASE / "data" / "ts_data.csv"   # 통합 자료를 직접 두고 싶으면 여기에 넣기
CACHE_CSV = Path(tempfile.gettempdir()) / "ocean_ts_data_cache.csv"


def default_data_path():
    """기본 자료 경로. data/ts_data.csv가 없으면 저장소의 월별 CSV·KODC CSV를 통합해 캐시로 쓴다."""
    if DEFAULT_CSV.exists():
        return DEFAULT_CSV
    if CACHE_CSV.exists():
        return CACHE_CSV
    if any(BASE.glob("*월_*.csv")) or any(BASE.glob("정선해양*.csv")):
        try:
            import prepare_data
            prepare_data.build(str(BASE), str(CACHE_CSV), verbose=False)
            return CACHE_CSV
        except Exception as e:  # 통합 실패 시 업로드 방식으로 대체
            print("기본 자료 통합 실패:", e, file=sys.stderr)
    return None


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
  body { font-family: system-ui, "Noto Sans KR", sans-serif; max-width: 920px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
  h1 { font-size: 1.4rem; }
  fieldset { border: 1px solid #ccc; border-radius: 8px; margin: 1rem 0; padding: .8rem 1rem; }
  legend { font-weight: 600; padding: 0 .4rem; }
  label { display: block; margin: .5rem 0 .15rem; font-size: .9rem; }
  label.inline { display: inline-flex; align-items: center; gap: .4rem; margin: .3rem 0; }
  input[type=number], input[type=text], input[type=date] { width: 8.5rem; padding: .3rem; }
  .row { display: flex; gap: 1rem; flex-wrap: wrap; }
  button { padding: .6rem 1.4rem; font-size: 1rem; cursor: pointer; }
  button.small { padding: .25rem .7rem; font-size: .85rem; }
  pre { background: #f4f4f4; padding: .8rem; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; }
  .err { color: #b00020; }
  img.result { max-width: 100%; }
  small, .hint { color: #666; font-size: .85rem; }
  #pvwrap { position: relative; display: none; margin: .6rem 0; max-width: 100%; cursor: crosshair; line-height: 0; }
  #pv { max-width: 100%; border: 1px solid #ccc; }
  .dot { position: absolute; width: 16px; height: 16px; margin: -8px 0 0 -8px; border-radius: 50%;
         background: rgba(0,200,255,.9); border: 2px solid #fff; color: #000; font: 700 10px/12px sans-serif;
         text-align: center; pointer-events: none; }
  table.gcp { border-collapse: collapse; margin-top: .5rem; }
  table.gcp td, table.gcp th { padding: .2rem .4rem; font-size: .85rem; text-align: left; }
  table.gcp input { width: 7rem; }
</style>
</head>
<body>
<h1>위성 사진 → 수온·염분 예측</h1>
<form method="post" enctype="multipart/form-data">
  <fieldset>
    <legend>1. 파일</legend>
    <label>위성 사진 (GeoTIFF 권장, EXIF GPS 사진·지도 캡처도 가능)</label>
    <input type="file" name="image" id="imgfile" accept=".tif,.tiff,.jpg,.jpeg,.png">
    <label>수온·염분 CSV {% if has_default %}<small>(비우면 저장소에 올린 자료 사용)</small>{% endif %}</label>
    <input type="file" name="data" accept=".csv" {% if not has_default %}required{% endif %}>
  </fieldset>

  <fieldset>
    <legend>2. 위치 (GeoTIFF나 GPS 사진이면 비워 두세요)</legend>
    <b>방법 A. 기준점</b> <span class="hint">— 지도 캡처처럼 위치 정보가 없는 사진용. 위치를 아는 지점(도시·섬·곶) 3개 이상</span>
    <div class="hint">사진을 고르면 아래에 미리보기가 나옵니다. 아는 지점을 클릭하면 픽셀 좌표가 기록되니 그 지점의 위도·경도를 입력하세요. (TIFF는 브라우저가 미리보기를 못 보여 주므로 좌표를 직접 입력)</div>
    <div id="pvwrap"><img id="pv" alt="미리보기"></div>
    <table class="gcp" id="gcptable">
      <thead><tr><th>#</th><th>픽셀 X</th><th>픽셀 Y</th><th>위도</th><th>경도</th><th></th></tr></thead>
      <tbody></tbody>
    </table>
    <button type="button" class="small" id="addrow">기준점 행 추가</button>

    <div style="margin-top:1rem"><b>방법 B. 위경도 범위</b> <span class="hint">— 사진이 위경도 등간격 격자일 때</span></div>
    <div class="row">
      <div><label>경도 최소</label><input type="number" step="any" name="lon_min"></div>
      <div><label>위도 최소</label><input type="number" step="any" name="lat_min"></div>
      <div><label>경도 최대</label><input type="number" step="any" name="lon_max"></div>
      <div><label>위도 최대</label><input type="number" step="any" name="lat_max"></div>
    </div>

    <div style="margin-top:1rem"><b>방법 C. 한 지점만</b></div>
    <div class="row">
      <div><label>위도</label><input type="number" step="any" name="lat"></div>
      <div><label>경도</label><input type="number" step="any" name="lon"></div>
    </div>
  </fieldset>

  <fieldset>
    <legend>3. 조건 (선택)</legend>
    <div class="row">
      <div><label>수심 (m)</label><input type="number" step="any" name="depth"></div>
      <div><label>날짜</label><input type="date" name="date" value="{{ today }}"></div>
      <div><label>자료 기간 (±일)</label><input type="number" step="1" min="0" name="window_days" value="60"></div>
      <div><label>관측점 최대 거리 (km)</label><input type="number" step="any" min="0" name="max_km" value="100"></div>
    </div>
    <div class="hint">날짜를 넣으면 그 날짜 ±N일(연도 무관)의 자료만 씁니다(0이면 전체 기간). 가장 가까운 관측점이 최대 거리보다 먼 칸은 예측하지 않고 비워 둡니다(0이면 끔). 날짜를 비우면 모든 시기의 자료가 섞입니다.</div>
    <label class="inline"><input type="checkbox" name="mask_land" value="1" checked> 육지를 가리고 바다만 예측</label>
    <div class="hint">어둡고 푸른 곳을 바다로 봅니다. 얕은 연안·갯벌은 육지로 처리될 수 있어요. 결과 그림의 노란 선이 바다 경계입니다.</div>
  </fieldset>
  <button type="submit">예측하기</button>
</form>

{% if error %}<p class="err"><b>오류:</b> {{ error }}</p>{% endif %}
{% if log %}<h2>결과</h2><pre>{{ log }}</pre>{% endif %}
{% if png %}<img class="result" alt="예측 지도" src="data:image/png;base64,{{ png }}">{% endif %}

<script>
(function () {
  const tbody = document.querySelector('#gcptable tbody');
  const wrap = document.getElementById('pvwrap');
  const pv = document.getElementById('pv');
  const fileInput = document.getElementById('imgfile');

  function renumber() {
    [...tbody.rows].forEach((tr, i) => {
      tr.cells[0].textContent = i + 1;
      const dot = tr._dot; if (dot) dot.textContent = i + 1;
    });
  }
  function addRow(x, y) {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td></td>' +
      '<td><input type="number" step="any" name="gcp_x"></td>' +
      '<td><input type="number" step="any" name="gcp_y"></td>' +
      '<td><input type="number" step="any" name="gcp_lat"></td>' +
      '<td><input type="number" step="any" name="gcp_lon"></td>' +
      '<td><button type="button" class="small">삭제</button></td>';
    tr.querySelector('button').onclick = () => { if (tr._dot) tr._dot.remove(); tr.remove(); renumber(); };
    tbody.appendChild(tr);
    if (x != null) {
      tr.querySelector('[name=gcp_x]').value = x;
      tr.querySelector('[name=gcp_y]').value = y;
      const d = document.createElement('div'); d.className = 'dot';
      d.style.left = (x / pv.naturalWidth * pv.clientWidth) + 'px';
      d.style.top = (y / pv.naturalHeight * pv.clientHeight) + 'px';
      wrap.appendChild(d); tr._dot = d;
      tr.querySelector('[name=gcp_lat]').focus();
    }
    renumber();
  }
  document.getElementById('addrow').onclick = () => addRow();
  for (let i = 0; i < 3; i++) addRow();

  pv.addEventListener('click', (e) => {
    const r = pv.getBoundingClientRect();
    const x = Math.round((e.clientX - r.left) / r.width * pv.naturalWidth);
    const y = Math.round((e.clientY - r.top) / r.height * pv.naturalHeight);
    // 비어 있는 첫 행을 쓰고, 없으면 새 행
    const empty = [...tbody.rows].find(tr => !tr.querySelector('[name=gcp_x]').value && !tr.querySelector('[name=gcp_y]').value);
    if (empty) {
      empty.querySelector('[name=gcp_x]').value = x;
      empty.querySelector('[name=gcp_y]').value = y;
      const d = document.createElement('div'); d.className = 'dot';
      d.style.left = (x / pv.naturalWidth * pv.clientWidth) + 'px';
      d.style.top = (y / pv.naturalHeight * pv.clientHeight) + 'px';
      wrap.appendChild(d); empty._dot = d; renumber();
      empty.querySelector('[name=gcp_lat]').focus();
    } else { addRow(x, y); }
  });

  fileInput.addEventListener('change', () => {
    [...tbody.rows].forEach(tr => { if (tr._dot) { tr._dot.remove(); tr._dot = null; } });
    const f = fileInput.files[0];
    if (!f) { wrap.style.display = 'none'; return; }
    pv.onload = () => { wrap.style.display = 'block'; };
    pv.onerror = () => { wrap.style.display = 'none'; };
    pv.src = URL.createObjectURL(f);
  });
})();
</script>
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
    default_path = default_data_path()
    ctx = dict(has_default=default_path is not None, error=None, log=None, png=None,
               today=datetime.date.today().isoformat())
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
        elif default_path is not None:
            data_path = default_path
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

        try:
            # 기준점: 네 칸이 모두 찬 행만 사용. 일부만 찬 행이 있으면 알려 준다.
            gx, gy = request.form.getlist("gcp_x"), request.form.getlist("gcp_y")
            gla, glo = request.form.getlist("gcp_lat"), request.form.getlist("gcp_lon")
            rows = list(zip(gx, gy, gla, glo))
            full = [r for r in rows if all(v.strip() for v in r)]
            partial = [r for r in rows if any(v.strip() for v in r) and not all(v.strip() for v in r)]
            if partial:
                ctx["error"] = "기준점 행 중 일부 칸이 비어 있습니다. 픽셀 X·Y와 위도·경도를 모두 채우거나 그 행을 삭제하세요."
                return render_template_string(PAGE, **ctx)
            if full:
                if not (img_file and img_file.filename):
                    ctx["error"] = "기준점을 쓰려면 위성 사진을 올려야 합니다."
                    return render_template_string(PAGE, **ctx)
                if len(full) < 3:
                    ctx["error"] = f"기준점은 최소 3개가 필요합니다 (지금 {len(full)}개)."
                    return render_template_string(PAGE, **ctx)
                for r in full:
                    cmd += ["--gcp"] + [str(float(v)) for v in r]

            bbox = [_form_float(k) for k in ("lon_min", "lat_min", "lon_max", "lat_max")]
            if all(bbox):
                cmd += ["--bbox"] + [str(float(v)) for v in bbox]
            lat, lon = _form_float("lat"), _form_float("lon")
            if lat and lon:
                cmd += ["--lat", str(float(lat)), "--lon", str(float(lon))]
            depth = _form_float("depth")
            if depth:
                cmd += ["--depth", str(float(depth))]
            window_days = _form_float("window_days")
            if window_days:
                cmd += ["--window-days", str(int(float(window_days)))]
            max_km = _form_float("max_km")
            if max_km:
                cmd += ["--max-km", str(float(max_km))]
        except ValueError:
            ctx["error"] = "숫자 입력 형식이 올바르지 않습니다."
            return render_template_string(PAGE, **ctx)
        date = _form_float("date")
        if date:
            cmd += ["--date", date]
        if request.form.get("mask_land") and img_file and img_file.filename:
            cmd += ["--mask-land"]

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
