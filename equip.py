"""몸틀 도구: 기본 몸 시트와 편집본들로 몸틀(몸·장비 층·캐릭터 머리 층)을 만든다.

사용법:
    python equip.py 입력폴더 -o 출력폴더 [-t 몸틀폴더] [-c config_128.json]

입력폴더:
    base.png            장비 없는 기본 몸 시트 (방향들을 가로로 나란히). 몸틀의 몸이 된다
    <장비이름>.png      base.png 를 ChatGPT 로 편집해 장비 하나만 입힌 시트 (같은 크기).
                        slot 은 이름의 첫 '_' 앞 (armor_iron → armor). 같은 slot 끼리 바꿔 끼운다
    heads/<이름>.png    base.png 를 편집해 얼굴·머리카락만 바꾼 시트 = 캐릭터 머리
    기본 몸과 크게 다른 그림(따로 그린 무기 시트 등)은 편집본이 아니므로 건너뛴다.

하는 일:
    1. 떼어 내기: 편집본과 기본 몸을 비교해 바뀐 곳 + "기본 몸에서는 배경인데 편집본에서는 배경이 아닌 곳"을
       층으로 본다. 외곽선 1~3픽셀 흔들림은 지우고, 조각난 것은 잇고, 안쪽 구멍은 메운다.
    2. 방향 나누기: 기본 몸의 빈 세로줄로 방향을 나누고, 두 방향 사이 빈 곳의 가운데에서 자른다
       (5개면 front, front_left, left, back_left, back / 3개면 front, front_left, left).
       기본 몸만으로 정하므로 장비·머리를 나중에 더해도 자르는 자리가 바뀌지 않는다.
    3. 캐릭터 머리: 몸틀의 머리 조각(parts/<방향>/head.png)에서 바뀐 곳을 지우고 편집본의 바뀐 곳을 얹는다.
       머리 조각에 닿지 않는 바뀐 곳(몸 쪽 색 흔들림)은 버린다. 몸틀 조각이 있어야 하므로
       처음에는 -t 로 몸틀을 만들고 animate rig·parts 를 마친 뒤 다시 돌린다.
    4. 픽셀화: 기본 몸의 키를 char_height 에 맞춘 같은 칸 나누기로 모든 층을 줄인다. 팔레트는 전체에서 하나.
       shade / face 설정을 그대로 따른다 (얼굴은 기본 몸에만).

결과 (출력폴더):
    base/<방향>.png, items/<장비>/<방향>.png, heads/<이름>/<방향>.png
                                          캔버스 크기의 픽셀 층 (외곽선 없음 — 다 쌓은 뒤 두른다)
    preview/<방향>_<조합>_x<배율>.png     기본 몸 / 장비 하나 / 머리 하나를 합치고 외곽선을 두른 미리 보기
    palette.hex
결과 (몸틀폴더, -t):
    <방향>.png                몸틀 몸 (이미 있으면 그대로 둔다 — 뼈대·조각이 이 그림에 맞춰져 있다)
    items/<장비>/<방향>.png   원본 해상도 장비 층 (몸 그림과 같은 크기)
    heads/<이름>/<방향>.png   원본 해상도 머리 층
    장비·머리 층은 왼쪽을 보는 방향을 뒤집어 오른쪽 방향(right, front_right, back_right)도 쓴다.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # 내장 Python은 스크립트 폴더를 자동으로 찾지 않는다

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import pixelize as px  # noqa: E402

VIEW_NAMES = {5: ["front", "front_left", "left", "back_left", "back"], 3: ["front", "front_left", "left"],
              1: ["front"]}
MIRROR = {"left": "right", "front_left": "front_right", "back_left": "back_right"}  # animate mirror 와 같은 짝
MIN_SAME = 0.6   # 기본 몸의 몸 픽셀 중 이만큼은 그대로여야 편집본으로 본다


def foreground(img, cfg):
    return np.asarray(px.remove_background(img, cfg["bg_tolerance"]))[..., 3] > 0


def extract(base_rgb, base_fg, eq_img, cfg):
    """편집본에서 바뀐 곳: (RGBA 시트, 바뀐 곳 마스크). 알파는 바뀐 곳 중 편집본에서 배경이 아닌 곳
    (머리를 짧게 바꾸면 예전 머리 자리는 바뀐 곳이지만 투명하다). 편집본이 아니면 None."""
    eq_rgb = np.asarray(eq_img.convert("RGB"))
    if eq_rgb.shape != base_rgb.shape:
        return None
    d = np.abs(base_rgb.astype(np.int16) - eq_rgb.astype(np.int16)).max(2)
    if (d[base_fg] <= 40).mean() < MIN_SAME:
        return None
    unit = d.shape[0] / 1024   # 1024 높이 기준 두께

    def k(r):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, round(r * unit)) + 1,) * 2)
    # 바뀐 곳 + 기본 몸에서는 배경인데 편집본에서는 배경이 아닌 곳 (배경과 밝기가 비슷한 은색 장비도 잡히게)
    eq_fg = foreground(eq_img, cfg)
    m = ((d > 40) | (eq_fg & ~base_fg)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k(2))    # 외곽선 흔들림 지우기
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k(6))   # 장비 안쪽 선 때문에 조각난 것을 잇기
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > 400 * unit * unit:
            keep[lab == i] = 1
    inv = np.pad(1 - keep, 1, constant_values=1)
    cv2.floodFill(inv, None, (0, 0), 2)
    keep |= (inv[1:-1, 1:-1] == 1).astype(np.uint8)   # 안쪽 구멍 메우기
    edge = cv2.dilate(keep, k(4)).astype(bool) & ~keep.astype(bool)
    keep = keep.astype(bool) | (edge & (d > 25))       # 장비 자신의 외곽선까지 (몸·배경 그대로인 곳은 빼고)
    return np.dstack([eq_rgb, np.where(keep & eq_fg, 255, 0).astype(np.uint8)]), keep


def head_layer(head_part, edit, changed):
    """캐릭터 머리 층: 몸틀 머리 조각에서 바뀐 곳을 지우고 편집본의 바뀐 곳을 얹는다.
    바뀐 곳 중 머리 조각에 닿지 않는 덩어리(몸 쪽 색 흔들림)는 버린다. (머리 층, 버린 비율)."""
    out = np.asarray(head_part).copy()
    head = (out[..., 3] > 0).astype(np.uint8)
    _, lab = cv2.connectedComponents(changed.astype(np.uint8), connectivity=8)
    touch = np.unique(lab[cv2.dilate(head, np.ones((5, 5), np.uint8)) > 0])
    keep = np.isin(lab, touch[touch > 0])
    out[keep] = np.asarray(edit)[keep]
    return Image.fromarray(out), (changed & ~keep).sum() / max(changed.sum(), 1)


def merged_palette(views, cfg):
    """기본 몸에서 colors 색, 장비·머리마다 그 층에서 colors/2 색을 따로 뽑아 합친다.
    한꺼번에 뽑으면 넓이가 작은 층의 색(은색 갑옷의 밝은 면 등)이 빠져 피부색 같은 엉뚱한 색으로 바뀐다."""
    import oklab
    thr = cfg["alpha_threshold"]
    first = next(iter(views.values()))
    parts = [px.build_palette([v["base"] for v in views.values()], cfg["colors"], thr)]
    for group in ("items", "heads"):
        for key in first[group]:
            parts.append(px.build_palette([v[group][key] for v in views.values()], max(8, cfg["colors"] // 2), thr))
    out = []
    for c in np.concatenate(parts):
        lab = oklab.srgb_to_oklab(c)
        if all(np.linalg.norm(lab - oklab.srgb_to_oklab(o)) > 0.02 for o in out):   # 거의 같은 색은 하나만
            out.append(c)
    return np.array(out, np.int32)


def layer_outline(layer, base, color):
    """장비 층 둘레 중 기본 몸 위에 놓이는 칸에 1칸 외곽선을 더한다 (몸 밖 둘레는 다 쌓은 뒤 두르는 외곽선이 맡는다).
    장비가 몸에 묻혀 보이지 않게 한다."""
    a = np.asarray(layer).copy()
    m = a[..., 3] > 0
    near = np.zeros_like(m)   # 위아래좌우로 장비에 닿은 칸 (대각선은 빼서 선이 1칸 굵기로 남게)
    near[1:] |= m[:-1]
    near[:-1] |= m[1:]
    near[:, 1:] |= m[:, :-1]
    near[:, :-1] |= m[:, 1:]
    a[near & ~m & (np.asarray(base)[..., 3] > 0)] = [*color, 255]
    return Image.fromarray(a)


def view_ranges(solid):
    """세로로 빈 줄을 경계로 나눈 방향 구간 (x0, x1). 자르는 자리는 두 방향 사이 빈 곳의 가운데."""
    cols = solid.any(0)
    runs, start = [], None
    for x, c in enumerate(list(cols) + [False]):
        if c and start is None:
            start = x
        elif not c and start is not None:
            runs.append((start, x))
            start = None
    runs = [r for r in runs if r[1] - r[0] > solid.shape[1] * 0.03]
    cuts = [0] + [(a[1] + b[0]) // 2 for a, b in zip(runs, runs[1:])] + [solid.shape[1]]
    return list(zip(cuts[:-1], cuts[1:]))


def to_canvas(src, box, cfg):
    """src(방향 그림, RGBA)를 기본 몸 상자 box 기준의 같은 칸 나누기로 캔버스 전체에 맞춰 줄일 준비:
    캔버스 칸 하나가 원본의 몇 픽셀인지(sx, sy)와 캔버스 원점에 맞춘 원본 그림을 돌려준다."""
    bx0, by0, bx1, by1 = box
    base_crop = src.crop(box)
    w, h, ox, oy = px.placement(base_crop, cfg, "")
    sx, sy = (bx1 - bx0) / w, (by1 - by0) / h
    c = cfg["canvas"]
    W, H = round(c * sx), round(c * sy)
    pad = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    pad.paste(src, (round(ox * sx - bx0), round(oy * sy - by0)))
    return pad, (sx, sy, ox, oy)


def save_view(img, folder, view):
    """몸틀 층 그림을 쓴다. 왼쪽을 보는 방향이면 뒤집어 오른쪽 방향도 (animate mirror 로 만든 몸과 맞는다)."""
    folder.mkdir(parents=True, exist_ok=True)
    img.save(folder / f"{view}.png")
    if view in MIRROR:
        img.transpose(Image.FLIP_LEFT_RIGHT).save(folder / f"{MIRROR[view]}.png")


def main():
    parser = argparse.ArgumentParser(description="기본 몸과 편집본으로 몸틀(몸·장비 층·캐릭터 머리 층)을 만듭니다.")
    parser.add_argument("input", type=Path, help="base.png 와 편집본이 든 폴더")
    parser.add_argument("-o", "--output", type=Path, required=True, help="픽셀 층과 미리 보기를 넣을 폴더")
    parser.add_argument("-t", "--template", type=Path, help="원본 해상도 층을 쓸 몸틀 폴더 (예: templates\\male)")
    parser.add_argument("-c", "--config", type=Path, help="설정 파일 (없으면 기본값)")
    args = parser.parse_args()
    cfg = px.load_config(args.config)
    thr = cfg["alpha_threshold"]

    base_path = args.input / "base.png"
    if not base_path.exists():
        sys.exit(f"{base_path} 가 없습니다 (장비 없는 기본 몸 시트)")
    base_img = Image.open(base_path)
    base_rgba = np.asarray(px.remove_background(base_img, cfg["bg_tolerance"])).copy()
    base_fg = base_rgba[..., 3] > 0
    base_rgb = np.asarray(base_img.convert("RGB"))

    edits = {"items": {}, "heads": {}}
    files = {"items": sorted(args.input.iterdir()), "heads": sorted((args.input / "heads").glob("*"))}
    for group, paths in files.items():
        for f in paths:
            if f.suffix.lower() not in px.IMAGE_EXTS or f.name == base_path.name:
                continue
            got = extract(base_rgb, base_fg, Image.open(f), cfg)
            if got is None:
                print(f"  건너뜀 {f.name}: 기본 몸의 편집본이 아닙니다 (크기가 다르거나 몸이 많이 바뀜)")
                continue
            edits[group][f.stem] = got
            print(f"  {'장비' if group == 'items' else '머리'} {f.stem}: 시트의 {(got[0][..., 3] > 0).mean() * 100:.1f}%")

    ranges = view_ranges(base_rgba[..., 3] >= thr)
    names = VIEW_NAMES.get(len(ranges), [f"view{i}" for i in range(len(ranges))])
    print(f"방향 {len(ranges)}개: {', '.join(names)}")

    views = {}
    for name, (x0, x1) in zip(names, ranges):
        base_v = px.remove_small_parts(Image.fromarray(base_rgba[:, x0:x1].copy()), thr, cfg["min_part_ratio"])
        ys, xs = np.nonzero(np.asarray(base_v)[..., 3] >= thr)
        box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)   # 키와 자리는 기본 몸으로 정한다
        views[name] = {"base": base_v, "box": box,
                       **{g: {k: Image.fromarray(rgba[:, x0:x1].copy()) for k, (rgba, _) in edits[g].items()}
                          for g in edits},
                       "changed": {k: keep[:, x0:x1] for k, (_, keep) in edits["heads"].items()}}

    # 캐릭터 머리: 몸틀 머리 조각이 있어야 만든다
    tpl = args.template
    head_parts = {}
    for name in views:
        path = tpl / "parts" / name / "head.png" if tpl else None
        if path and path.exists():
            head_parts[name] = Image.open(path).convert("RGBA")
    if edits["heads"] and len(head_parts) < len(views):
        print("  머리 층은 몸틀 머리 조각(parts/<방향>/head.png)이 있는 방향만 만듭니다."
              " 몸틀을 처음 만들 때는 animate rig·parts 뒤에 다시 돌리세요")
    for name, v in views.items():
        v["head_layers"] = {}
        if name not in head_parts:
            continue
        if head_parts[name].size != v["base"].size:
            sys.exit(f"{name}: 몸틀 머리 조각과 방향 그림의 크기가 다릅니다. base.png 가 몸틀을 만들 때와 같은지 보세요")
        for k, edit in v["heads"].items():
            v["head_layers"][k], dropped = head_layer(head_parts[name], edit, v["changed"][k])
            if dropped > 0.05:
                print(f"  머리 {k} {name}: 머리에 닿지 않은 바뀐 곳 {dropped * 100:.0f}% 를 버렸습니다 (몸이 바뀐 편집본인지 보세요)")

    # 팔레트: 기본 몸과 모든 층에서 하나
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if cfg["palette"]:
        palette = px.read_hex(cfg["palette"])
    else:
        palette = merged_palette(views, cfg)
    outline_rgb = px.hex_to_rgb(cfg["outline_color"])

    polish = None
    if cfg["shade"] or cfg["face"]:
        import polish
    faces, tones = {}, None
    if cfg["face"]:
        print("얼굴 찾기 (자세 인식 모델)")
        for name, v in views.items():
            if polish.view_kind(name) != "back":
                crop = v["base"].crop(v["box"])
                faces[name] = polish.find_face(crop)
        colors = [c for name, f in faces.items() for e in f["eyes"]
                  if (c := polish.iris_color(views[name]["base"].crop(views[name]["box"]), e["pts"])) is not None]
        if colors:
            tones = polish.iris_tones(max(colors, key=lambda c: np.hypot(c[1], c[2])))

    def pixel(img, box):
        pad, _ = to_canvas(img, box, cfg)
        small = px.downscale(pad, (cfg["canvas"], cfg["canvas"]), palette, thr)
        if cfg["shade"]:
            fixed = palette if cfg["palette"] else None
            small = Image.fromarray(polish.shade(small, list(palette) + [outline_rgb], np.array(outline_rgb), fixed))
        return small

    scale = cfg["preview_scale"]
    c = cfg["canvas"]
    (out / "preview").mkdir(exist_ok=True)
    used = set()
    for name, v in views.items():
        base_px = pixel(v["base"], v["box"])
        notes = []
        if name in faces and tones is not None:
            _, (sx, sy, ox, oy) = to_canvas(v["base"], v["box"], cfg)
            a = np.asarray(base_px).copy()
            notes = polish.draw_face(a, faces[name], lambda p: np.array([ox + p[0] / sx, oy + p[1] / sy]),
                                     tones, polish.view_kind(name))
            base_px = Image.fromarray(a)
        item_px = {k: pixel(img, v["box"]) for k, img in v["items"].items()}
        if cfg["outline"]:
            item_px = {k: layer_outline(img, base_px, outline_rgb) for k, img in item_px.items()}
        head_px = {k: pixel(img, v["box"]) for k, img in v["head_layers"].items()}

        layers = [("base", base_px)] + [(f"items/{k}", i) for k, i in item_px.items()] + \
                 [(f"heads/{k}", i) for k, i in head_px.items()]
        for key, img in layers:
            (out / key).mkdir(parents=True, exist_ok=True)
            img.save(out / key / f"{name}.png")
            a = np.asarray(img)
            used |= {tuple(int(t) for t in p) for p in a[a[..., 3] > 0][:, :3]}

        # 미리 보기: 기본 몸 / 기본 몸 + 장비 하나 / 머리 바꾸기 (몸틀 머리 자리를 비우고 새 머리를 얹는다)
        combos = [("base", base_px.copy())]
        for k, top in item_px.items():
            combos.append((k, Image.alpha_composite(base_px, top)))
        if head_px:
            hole = np.asarray(pixel(head_parts[name], v["box"]))[..., 3] > 0
            body = np.asarray(base_px).copy()
            body[hole] = 0
            for k, top in head_px.items():
                combos.append((k, Image.alpha_composite(Image.fromarray(body), top)))
        for combo, comp in combos:
            if cfg["outline"]:
                comp = px.add_outline(comp, outline_rgb)
            comp.resize((c * scale, c * scale), Image.NEAREST).save(out / "preview" / f"{name}_{combo}_x{scale}.png")
        print(f"  {name}" + (f" ({', '.join(notes)})" if notes else ""))

        if tpl:
            body = tpl / f"{name}.png"
            if not body.exists():
                tpl.mkdir(parents=True, exist_ok=True)
                v["base"].save(body)
            elif Image.open(body).size != v["base"].size:
                sys.exit(f"{body} 와 크기가 다릅니다. base.png 가 몸틀을 만들 때와 같은지 보세요")
            for k, img in v["items"].items():
                save_view(img, tpl / "items" / k, name)
            for k, img in v["head_layers"].items():
                save_view(img, tpl / "heads" / k, name)

    if not cfg["palette"]:
        old = [tuple(int(t) for t in p) for p in palette]
        new = sorted(used - set(old) - {tuple(outline_rgb)}, key=lambda p: -sum(p))
        px.write_hex(out / "palette.hex", [p for p in old if p in used] + new + [tuple(outline_rgb)])
    print(f"팔레트: {out / 'palette.hex'}  장비: {', '.join(edits['items']) or '없음'}"
          f"  머리: {', '.join(edits['heads']) or '없음'}" + (f"  몸틀: {tpl}" if tpl else ""))


if __name__ == "__main__":
    main()
