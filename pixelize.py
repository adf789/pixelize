"""AI 일러스트를 정해진 규칙대로 64칸 픽셀 그림으로 바꾸는 도구.

사용법:
    python pixelize.py 입력폴더 -o 출력폴더 [-c config.json]

한 번에 넣은 그림들은 같은 색 묶음(팔레트)을 함께 쓴다.
그래서 같은 캐릭터의 앞/옆/뒤 그림은 한 폴더에 넣고 같이 돌린다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DEFAULTS = {
    "canvas": 64,
    "char_height": 48,
    "foot_margin": 4,
    "colors": 24,
    "palette": None,
    "alpha_threshold": 128,
    "bg_tolerance": 40,
    "min_part_ratio": 0.02,
    "outline": True,
    "outline_color": "#1b1325",
    "preview_scale": 8,
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def load_config(path):
    cfg = dict(DEFAULTS)
    if path:
        cfg.update(json.loads(Path(path).read_text(encoding="utf-8")))
        if cfg["palette"]:
            cfg["palette"] = str(Path(path).parent / cfg["palette"])
    return cfg


def hex_to_rgb(text):
    text = text.strip().lstrip("#")
    return [int(text[i:i + 2], 16) for i in (0, 2, 4)]


def read_hex(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return np.array([hex_to_rgb(line) for line in lines if line.strip()], dtype=np.int32)


def write_hex(path, palette):
    text = "\n".join("%02x%02x%02x" % tuple(c) for c in palette)
    Path(path).write_text(text + "\n", encoding="utf-8")


def remove_background(img, tolerance):
    """테두리와 이어진 배경색 영역을 투명하게 만든다. 이미 투명 배경이면 그대로 둔다."""
    rgba = np.asarray(img.convert("RGBA")).copy()
    if rgba[..., 3].min() < 250:
        return Image.fromarray(rgba)

    rgb = rgba[..., :3].astype(np.int16)
    corners = np.array([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]])
    bg = np.median(corners, axis=0)
    near_bg = np.abs(rgb - bg).max(axis=2) <= tolerance

    # 배경색과 비슷하더라도 캐릭터 안쪽(눈 흰자 등)은 남기려고, 테두리에서 이어진 곳만 지운다.
    # fromarray 결과는 읽기 전용이라 floodfill이 칠하지 못하므로 복사본을 쓴다.
    mask = Image.fromarray(np.where(near_bg, 255, 0).astype(np.uint8), "L").copy()
    w, h = mask.size
    border = ([(x, 0) for x in range(w)] + [(x, h - 1) for x in range(w)]
              + [(0, y) for y in range(h)] + [(w - 1, y) for y in range(h)])
    for xy in border:
        if mask.getpixel(xy) == 255:
            ImageDraw.floodfill(mask, xy, 128)
    rgba[np.asarray(mask) == 128, 3] = 0
    return Image.fromarray(rgba)


def remove_small_parts(img, threshold, min_ratio):
    """본체와 떨어진 작은 조각(그림 아래 'Front' 같은 글자, 티끌)을 지운다.
    가장 큰 덩어리의 min_ratio배보다 작은 덩어리를 지운다."""
    rgba = np.asarray(img).copy()
    solid = rgba[..., 3] >= threshold
    mask = Image.fromarray(np.where(solid, 255, 0).astype(np.uint8), "L").copy()

    parts = []  # (시작점, 넓이)
    while True:
        rest = np.flatnonzero(np.asarray(mask) == 255)
        if rest.size == 0:
            break
        y, x = divmod(int(rest[0]), mask.width)
        ImageDraw.floodfill(mask, (x, y), 128)
        parts.append(((x, y), int((np.asarray(mask) == 128).sum())))
        ImageDraw.floodfill(mask, (x, y), 0)

    biggest = max(size for _, size in parts)
    keep = Image.fromarray(np.where(solid, 255, 0).astype(np.uint8), "L").copy()
    for xy, size in parts:
        if size < biggest * min_ratio:
            ImageDraw.floodfill(keep, xy, 0)
    rgba[solid & (np.asarray(keep) == 0), 3] = 0
    return Image.fromarray(rgba)


def crop_to_content(img, threshold, name):
    alpha = np.asarray(img)[..., 3]
    ys, xs = np.nonzero(alpha >= threshold)
    if len(xs) == 0:
        raise ValueError(f"{name}: 캐릭터를 찾지 못했습니다 (배경 제거 결과가 비어 있음)")
    return img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def build_palette(images, n, threshold):
    """여러 원본 그림의 색을 모아 하나의 팔레트를 만든다."""
    pixels = np.concatenate([np.asarray(im)[..., :3][np.asarray(im)[..., 3] >= threshold] for im in images])
    pixels = pixels[::max(1, len(pixels) // 200_000)]  # 너무 많으면 골고루 덜어 낸다
    strip = Image.fromarray(pixels.reshape(1, -1, 3).astype(np.uint8), "RGB")
    q = strip.quantize(colors=n, method=Image.Quantize.MEDIANCUT, kmeans=n, dither=Image.Dither.NONE)
    used = np.unique(np.asarray(q))
    full = np.array(q.getpalette()[:256 * 3]).reshape(-1, 3)
    return full[used].astype(np.int32)


def nearest_index(rgb, palette):
    """색마다 팔레트에서 가장 가까운 색의 번호를 찾는다."""
    rgb = rgb.reshape(-1, 3).astype(np.int32)
    out = np.empty(len(rgb), np.int64)
    step = 100_000
    for s in range(0, len(rgb), step):
        dist = ((rgb[s:s + step, None, :] - palette[None, :, :]) ** 2 * [2, 4, 3]).sum(-1)
        out[s:s + step] = dist.argmin(1)
    return out


def downscale(img, size, palette, threshold):
    """원본 색을 먼저 팔레트 색으로 바꾼 뒤, 칸마다 가장 많이 나온 색 하나를 고른다.
    평균을 내지 않으므로 색이 섞여 탁해지지 않는다. 칸의 절반 이상이 비어 있으면 투명으로 둔다."""
    a = np.asarray(img)
    h, w = a.shape[:2]
    solid = a[..., 3] >= threshold
    idx = nearest_index(a[..., :3], palette).reshape(h, w)

    out_w, out_h = size
    xs = np.linspace(0, w, out_w + 1).round().astype(int)
    ys = np.linspace(0, h, out_h + 1).round().astype(int)
    out = np.zeros((out_h, out_w, 4), np.uint8)
    for j in range(out_h):
        for i in range(out_w):
            block = solid[ys[j]:ys[j + 1], xs[i]:xs[i + 1]]
            if block.mean() < 0.5:
                continue
            counts = np.bincount(idx[ys[j]:ys[j + 1], xs[i]:xs[i + 1]][block])
            out[j, i] = [*palette[counts.argmax()], 255]
    return Image.fromarray(out)


def place_on_canvas(img, palette, cfg, name):
    """캐릭터 키를 char_height에 맞추고, 발끝을 바닥 기준선에 붙여 가운데 놓는다."""
    canvas = cfg["canvas"]
    scale = cfg["char_height"] / img.height
    max_w = canvas - 2  # 양옆 외곽선 자리
    if img.width * scale > max_w:
        scale = max_w / img.width
        print(f"  주의 {name}: 너무 넓어서 키를 {round(img.height * scale)}칸으로 줄였습니다", file=sys.stderr)

    w = max(1, round(img.width * scale))
    h = max(1, round(img.height * scale))
    small = downscale(img, (w, h), palette, cfg["alpha_threshold"])
    out = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    out.paste(small, ((canvas - w) // 2, canvas - cfg["foot_margin"] - h))
    return out


def add_outline(img, color):
    """캐릭터 바깥쪽에 1칸 외곽선을 두른다 (위아래좌우 이웃 기준)."""
    a = np.asarray(img).copy()
    solid = a[..., 3] == 255
    grown = solid.copy()
    grown[1:] |= solid[:-1]
    grown[:-1] |= solid[1:]
    grown[:, 1:] |= solid[:, :-1]
    grown[:, :-1] |= solid[:, 1:]
    a[grown & ~solid] = [*color, 255]
    return Image.fromarray(a)


def main():
    parser = argparse.ArgumentParser(description="AI 일러스트를 64칸 픽셀 그림으로 바꿉니다.")
    parser.add_argument("input", type=Path, help="원본 그림이 든 폴더")
    parser.add_argument("-o", "--output", type=Path, required=True, help="결과를 넣을 폴더")
    parser.add_argument("-c", "--config", type=Path, help="설정 파일 (없으면 기본값)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    files = sorted(f for f in args.input.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not files:
        sys.exit(f"{args.input} 에 그림 파일이 없습니다")

    threshold = cfg["alpha_threshold"]
    crops = {}
    for f in files:
        img = remove_background(Image.open(f), cfg["bg_tolerance"])
        img = remove_small_parts(img, threshold, cfg["min_part_ratio"])
        crops[f.stem] = crop_to_content(img, threshold, f.name)

    args.output.mkdir(parents=True, exist_ok=True)
    if cfg["palette"]:
        palette = read_hex(cfg["palette"])
        print(f"팔레트 사용: {cfg['palette']} ({len(palette)}색)")
    else:
        palette = build_palette(crops.values(), cfg["colors"], threshold)
        saved = list(palette) + ([hex_to_rgb(cfg["outline_color"])] if cfg["outline"] else [])
        write_hex(args.output / "palette.hex", saved)
        print(f"팔레트 새로 만듦: {args.output / 'palette.hex'} ({len(saved)}색)")

    preview_dir = args.output / "preview"
    preview_dir.mkdir(exist_ok=True)
    canvas, scale = cfg["canvas"], cfg["preview_scale"]
    for name, img in crops.items():
        img = place_on_canvas(img, palette, cfg, name)
        if cfg["outline"]:
            img = add_outline(img, hex_to_rgb(cfg["outline_color"]))
        img.save(args.output / f"{name}.png")
        img.resize((canvas * scale, canvas * scale), Image.NEAREST).save(preview_dir / f"{name}_x{scale}.png")
        print(f"  {name}.png")


if __name__ == "__main__":
    main()
