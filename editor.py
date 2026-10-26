"""동작 편집기 서버. 브라우저에서 동작을 재생하며 뼈를 조절하고, 지금 쓰는 동작 JSON 형식 그대로 저장한다.

사용법:
    python editor.py [캐릭터폴더] [--port 8765] [--open]
    → 브라우저에서 출력된 주소(기본 http://127.0.0.1:8765)를 연다. --open 이면 브라우저를 저절로 연다.
    그 포트를 다른 프로그램이 쓰고 있으면 다음 포트(8766, 8767, …)를 쓴다.

편집기(editor/editor.js)는 animate.py 와 똑같은 규칙(뼈 행렬, 열쇠 곡선, view_swap)으로 자세를 계산해 그린다.
그림 조각은 parts.PartModel 이 쓰는 것과 같은 텍스처·자리 행렬을 넘겨주므로, 편집기에서 본 자세와 픽셀 결과가 어긋나지 않는다.
"""
import argparse
import io
import json
import os
import subprocess
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # 내장 Python은 스크립트 폴더를 자동으로 찾지 않는다
import animate as an  # noqa: E402
import parts  # noqa: E402

STATIC = HERE / "editor"
ANIMS = HERE / "anims"


def straight_png(tex):
    """PartModel 텍스처(알파를 곱한 0~1 실수)를 보통 RGBA PNG 바이트로."""
    a = tex[..., 3:4]
    rgb = np.divide(tex[..., :3], a, out=np.zeros_like(tex[..., :3]), where=a > 1e-6)
    img = Image.fromarray((np.dstack([rgb, a]).clip(0, 1) * 255).round().astype(np.uint8), "RGBA")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


class Scenes:
    """방향마다 PartModel 을 한 번 만들어 두고 편집기용 장면 정보로 바꿔 준다."""

    def __init__(self, char_dir, cfg_path):
        self.char_dir, self.cfg_path = char_dir, cfg_path
        self.models, self.textures = {}, {}

    def reload(self):
        self.models, self.textures = {}, {}

    def model(self, view):
        if view not in self.models:
            rig_path = self.char_dir / f"rig_{view}.json"
            if not (self.char_dir / "parts" / view / "parts.json").exists():
                raise FileNotFoundError(f"{view}: 조각이 없습니다. 먼저 animate.bat parts 를 돌리세요")
            cfg = an.load_config(self.cfg_path)
            rig = json.loads(rig_path.read_text(encoding="utf-8"))
            self.models[view] = parts.PartModel(self.char_dir, rig, cfg)
        return self.models[view]

    def views(self):
        return [v for v in an.VIEWS if (self.char_dir / f"rig_{v}.json").exists()
                and (self.char_dir / "parts" / v / "parts.json").exists()]

    def scene(self, view):
        m = self.model(view)
        cfg = an.load_config(self.cfg_path)
        pieces = []
        for i, p in enumerate(m.pieces):
            self.textures[(view, i)] = p["tex"]
            pieces.append({"bone": p["bone"], "depth": p["depth"], "layer": p["layer"],
                           "rest": np.asarray(p["rest"]).tolist(), "tex": f"/api/tex?view={view}&i={i}",
                           "w": int(p["tex"].shape[1]), "h": int(p["tex"].shape[0])})
        return {
            "view": view, "size": m.size, "k": m.k, "unit": m.unit,
            "canvas": cfg["canvas"], "foot_line": m.size - cfg["foot_margin"] * m.k,
            "bones": [list(b) for b in m.bones],
            "joints": {n: [float(p[0]), float(p[1])] for n, p in m.joints.items()},
            "base_pose": m.base_pose, "weapon_bones": m.weapon_bones,
            "default_layers": m.default_layers, "pieces": pieces,
        }


def list_anims():
    out = {}
    for view in (*an.VIEWS, "common"):
        out[view] = sorted(p.stem for p in (ANIMS / view).glob("*.json"))
    return out


def make_handler(scenes):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # 요청마다 찍히는 줄은 끈다
            pass

        def send(self, body, ctype="application/json", code=200):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            elif isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def fail(self, msg, code=400):
            self.send({"error": msg}, code=code)

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    return self.send((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                if url.path.startswith("/static/"):
                    f = (STATIC / url.path[len("/static/"):]).resolve()
                    if STATIC.resolve() not in f.parents or not f.exists():
                        return self.fail("없는 파일", 404)
                    ctype = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}.get(
                        f.suffix, "application/octet-stream")
                    return self.send(f.read_bytes(), ctype)
                if url.path == "/api/info":
                    return self.send({"character": scenes.char_dir.name, "views": scenes.views(),
                                      "anims": list_anims()})
                if url.path == "/api/scene":
                    return self.send(scenes.scene(q["view"]))
                if url.path == "/api/tex":
                    return self.send(straight_png(scenes.textures[(q["view"], int(q["i"]))]), "image/png")
                if url.path == "/api/anim":
                    path = ANIMS / f"{q['key']}.json"
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    # 공통 동작은 방향마다 편집기가 바꿔 계산하므로 그대로 넘긴다
                    resolved = raw if q["key"].startswith("common/") else an.load_anim(path, ANIMS)
                    return self.send({"raw": raw, "resolved": resolved})
                if url.path == "/api/output":
                    f = (HERE / q["path"]).resolve()
                    if (HERE / "output").resolve() not in f.parents or not f.exists():
                        return self.fail("없는 파일", 404)
                    return self.send(f.read_bytes(), "image/png")
                if url.path == "/api/frame":
                    # Python 이 그린 큰 그림 (편집기 계산이 같은지 비교하는 데 쓴다)
                    view = q.get("view") or q["key"].split("/")[0]
                    anim = an.load_anim(ANIMS / f"{q['key']}.json", ANIMS, view)
                    models = {v: scenes.model(v) for v in scenes.views()}
                    t = an.frame_times(anim)[int(q["i"])]
                    m, pose = an.frame_setup(anim, models, view, t)
                    buf = io.BytesIO()
                    m.render(pose).save(buf, "PNG")
                    return self.send(buf.getvalue(), "image/png")
                return self.fail("없는 주소", 404)
            except (KeyError, FileNotFoundError, ValueError) as e:
                return self.fail(str(e))

        def do_POST(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                if url.path == "/api/anim":
                    key = q["key"]
                    view, name = key.split("/")
                    if view not in (*an.VIEWS, "common") or not name.replace("_", "").isalnum():
                        return self.fail("동작 이름은 영문·숫자·_ 만 됩니다")
                    data = json.loads(body.decode("utf-8"))
                    path = ANIMS / view / f"{name}.json"
                    path.write_text(format_anim(data), encoding="utf-8")
                    return self.send({"saved": str(path.relative_to(HERE))})
                if url.path == "/api/render":
                    key = q["key"]
                    out = HERE / "output" / scenes.char_dir.name
                    view, name = key.split("/")
                    # 공통 동작은 네 방향을 모두 만들고, 지금 보는 방향의 결과를 보여 준다
                    only = [f"{v}/{name}" for v in scenes.views()] if view == "common" else [key]
                    if view == "common":
                        key = f"{q.get('view') or scenes.views()[0]}/{name}"
                    r = subprocess.run([sys.executable, str(HERE / "animate.py"), "render", str(scenes.char_dir),
                                        "-o", str(out), "-c", str(scenes.cfg_path), "--only", *only],
                                       capture_output=True, text=True, encoding="utf-8", errors="replace")
                    name = key.replace("/", "_")
                    rel = lambda p: str(p.relative_to(HERE)).replace("\\", "/")
                    return self.send({"ok": r.returncode == 0, "log": (r.stdout + r.stderr)[-4000:],
                                      "sheet": rel(out / f"{name}.png"),
                                      "frames": rel(out / "check" / f"{name}_frames.png")})
                if url.path == "/api/reload":
                    scenes.reload()
                    return self.send({"ok": True})
                return self.fail("없는 주소", 404)
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                return self.fail(str(e))

    return Handler


def format_anim(data):
    """사람이 읽기 좋게: 바깥은 들여쓰기, 열쇠 목록은 한 줄로. 공통 동작의 views 는 방향마다 한 줄."""
    head = {k: v for k, v in data.items() if k not in ("keys", "views")}
    lines = ["{"]
    for k, v in head.items():
        lines.append(f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)},")
    if data.get("views"):
        lines.append('  "views": {')
        views = list(data["views"].items())
        for vi, (view, v) in enumerate(views):
            lines.append(f"    {json.dumps(view)}: {json.dumps(v, ensure_ascii=False)}" + ("," if vi < len(views) - 1 else ""))
        lines.append("  },")
    lines.append('  "keys": {')
    bones = list(data.get("keys", {}).items())
    for bi, (bone, chans) in enumerate(bones):
        inner = ", ".join(f"{json.dumps(ch)}: {json.dumps(keys, ensure_ascii=False)}" for ch, keys in chans.items())
        lines.append(f"    {json.dumps(bone)}: {{{inner}}}" + ("," if bi < len(bones) - 1 else ""))
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


class EditorServer(ThreadingHTTPServer):
    # Windows 에서는 주소 재사용을 켜면 다른 프로그램이 이미 쓰는 포트도 오류 없이 열려서,
    # 브라우저가 그 프로그램에 붙는다. 꺼서 이미 쓰는 포트면 오류가 나게 한다.
    allow_reuse_address = False


def open_server(handler, port, tries=20):
    for p in range(port, port + tries):
        try:
            return EditorServer(("127.0.0.1", p), handler), p
        except OSError:
            print(f"포트 {p} 는 다른 프로그램이 쓰고 있어 다음 포트를 찾습니다", flush=True)
    sys.exit(f"{port}~{port + tries - 1} 포트를 모두 쓰고 있습니다. --port 로 다른 번호를 주세요")


def main():
    parser = argparse.ArgumentParser(description="동작 편집기 서버")
    parser.add_argument("character", type=Path, nargs="?", default=HERE / "characters" / "warrior")
    parser.add_argument("-c", "--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    parser.add_argument("--open", action="store_true", help="브라우저를 저절로 연다")
    args = parser.parse_args()
    scenes = Scenes(args.character.resolve(), args.config.resolve())
    server, port = open_server(make_handler(scenes), args.port)
    url = f"http://127.0.0.1:{port}"
    print(f"동작 편집기: {url}  (끝내려면 Ctrl+C)", flush=True)
    if args.open:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
