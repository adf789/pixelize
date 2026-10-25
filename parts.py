"""조각 방식 1단계: 뼈대를 따라 그림을 부위별 조각으로 오리고, 가려진 곳과 관절 끝을 자동으로 채운다.

사용법:
    python animate.py parts 캐릭터폴더 [--force]

결과 (방향마다):
    characters/<이름>/parts/<방향>/<부위>.png   원본과 같은 크기의 투명 PNG. 겹치면 원본 자리에 딱 맞는다.
    characters/<이름>/parts/<방향>/parts.json  부위 목록, 앞뒤 순서(depth), 채운 픽셀 수
    characters/<이름>/check/parts_<방향>.png    조각 확인 그림 (분홍 = 가려져서 채운 곳, 하늘색 = 관절 덮개)

조각은 손으로 고칠 수 있으므로, 이미 있으면 --force 없이는 덮어쓰지 않는다.
"""
import json
import math
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

import animate as an
import pixelize as px

# 골반 조각: 몸통을 숙여도 허리띠·앞치마가 끌려가지 않도록 몸통에서 떼어 낸다.
HIPS_DEPTH = {"front": 1.5, "back": 1.5, "side": 2.5}
FILL_COLOR = (255, 60, 200)
CAP_COLOR = (60, 200, 255)
LIMBS = [f"{p}_{s}" for s in ("l", "r") for p in ("upper_arm", "forearm", "hand", "thigh", "shin", "foot")]
LIMB_REACH = 1.5      # 팔다리 조각은 뼈에서 (굵기의 반 × 이만큼) 안쪽만 가진다 (어깨 옆 망토가 팔에 붙지 않게)
HULL_PARTS = ("torso", "hips")  # 몸통처럼 통통한 조각은 볼록 껍질 안쪽의 가려진 곳까지 채운다
SIDE_NEAR_ARM = ["upper_arm_l", "forearm_l", "hand_l"]  # 옆 그림에서 몸통 위에 겹쳐 그려진 팔
CRUMB_RATIO = 0.2     # 한 조각에서 가장 큰 덩어리의 이 비율보다 작은 떨어진 부스러기는 이웃 조각에 붙인다


def body_rig(rig):
    """장비 층·붙인 물건을 뺀 몸 rig 에 골반 뼈를 더한다. 골반 뼈는 골반에서 아래로 몸통 길이의 30%."""
    j = rig["joints"]
    pelvis, neck = np.array(j["pelvis"], float), np.array(j["neck"], float)
    hips_end = pelvis + 0.3 * (pelvis - neck)
    body = {k: v for k, v in rig.items() if k not in ("layers", "attachments")}
    body["extra_bones"] = [{"name": "hips", "parent": None, "start": pelvis.tolist(), "end": hips_end.tolist(),
                            "depth": HIPS_DEPTH[rig["view"]]}] + list(rig.get("extra_bones", []))
    return body


def half_widths(char_dir, rig, cfg):
    """팔다리 뼈마다 굵기의 반(원본 픽셀). 팔다리가 몸에서 떨어진 T자 그림(앞, 없으면 뒤)에서 재고,
    다른 방향에는 키 비율로 맞춘다. 뼈 가운데 60% 구간에서 그림 가장자리까지 거리의 가운데 값."""
    thr = cfg["alpha_threshold"]
    own_height = (lambda b: b[3] - b[1])(an.content_box(an.load_clean(char_dir / rig["image"], cfg), thr))
    for view in ("front", "back"):
        path = char_dir / f"rig_{view}.json"
        if path.exists():
            ref = json.loads(path.read_text(encoding="utf-8"))
            break
    else:
        return {b: 0.03 * own_height for b in LIMBS}
    img = an.load_clean(char_dir / ref["image"], cfg)
    solid = (np.asarray(img)[..., 3] >= thr).astype(np.uint8)
    dt = cv2.distanceTransform(solid, cv2.DIST_L2, 5)
    _, joints, _ = an.rig_bones(ref)
    scale = own_height / (lambda b: b[3] - b[1])(an.content_box(img, thr))
    out = {}
    for name, _, j0, j1 in an.BONES:
        if name not in LIMBS:
            continue
        a, b = np.array(joints[j0]), np.array(joints[j1])
        pts = a + np.linspace(0.2, 0.8, 15)[:, None] * (b - a)
        xs = np.clip(pts[:, 0].astype(int), 0, dt.shape[1] - 1)
        ys = np.clip(pts[:, 1].astype(int), 0, dt.shape[0] - 1)
        out[name] = float(np.median(dt[ys, xs])) * scale
    for s in ("l", "r"):
        # 허벅지 옆에 망토·앞치마가 붙어 있으면 굵기가 부풀려지므로 정강이의 1.4배로 막는다
        out[f"thigh_{s}"] = min(out[f"thigh_{s}"], out[f"shin_{s}"] * 1.4)
    return out


def canvas_labels(model, step, thr, depth, reach=None, override=None):
    """그리기 크기(캔버스)의 픽셀마다 가장 많이 따르는 뼈 번호. 그림 밖은 가장 가까운 그림 픽셀의 번호.
    reach  {뼈 번호: 거리}: 그 뼈 선분에서 이보다 먼 픽셀은 그 뼈가 갖지 않는다.
    override {뼈 번호: 거리}: 그 뼈 선분에서 이보다 가까우면, 그 뼈가 더 앞에 있을 때 그 뼈가 갖는다."""
    size = model.size
    grid = np.zeros((size // step + 2, size // step + 2, len(model.bones)), np.float32)
    g = (model.verts / step).round().astype(int)
    grid[g[:, 1], g[:, 0]] = model.weights

    labels = np.full((size, size), -1, int)
    ys, xs = np.nonzero(np.asarray(model.rest)[..., 3] >= thr)
    fx, fy = (xs + 0.5) / step, (ys + 0.5) / step
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    tx, ty = (fx - x0)[:, None], (fy - y0)[:, None]
    w = (grid[y0, x0] * (1 - tx) * (1 - ty) + grid[y0, x0 + 1] * tx * (1 - ty)
         + grid[y0 + 1, x0] * (1 - tx) * ty + grid[y0 + 1, x0 + 1] * tx * ty)
    best = w.argmax(1)
    pts = np.c_[xs + 0.5, ys + 0.5]

    def dist(b):
        _, _, j0, j1 = model.bones[b]
        return an.segment_distance(pts, model.joints[j0], model.joints[j1])

    for b, r in (reach or {}).items():
        w[dist(b) > r, b] = 0
    label = np.where(w.max(1) > 0, w.argmax(1), best)
    for b, r in sorted((override or {}).items(), key=lambda br: depth[br[0]]):
        label = np.where((dist(b) <= r) & (depth[b] > depth[label]), b, label)
    labels[ys, xs] = label

    # 빈 칸은 가장 가까운 칠해진 칸의 번호로 (원본 크기로 옮길 때 가장자리가 비지 않게)
    unknown = (labels < 0).astype(np.uint8)
    _, nearest = cv2.distanceTransformWithLabels(unknown, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    known = labels[labels >= 0]  # 행 순서 = DIST_LABEL_PIXEL 의 번호 순서
    return known[nearest - 1]


def source_labels(model, dense, img, thr):
    """원본 그림의 픽셀마다 뼈 번호 (그림 밖은 -1)."""
    a = np.asarray(img)
    ys, xs = np.nonzero(a[..., 3] >= thr)
    c = model.to_canvas(np.c_[xs + 0.5, ys + 0.5])
    cx = np.clip(c[:, 0].astype(int), 0, model.size - 1)
    cy = np.clip(c[:, 1].astype(int), 0, model.size - 1)
    labels = np.full(a.shape[:2], -1, int)
    labels[ys, xs] = dense[cy, cx]
    return labels


def merge_crumbs(labels):
    """조각마다 가장 큰 덩어리의 CRUMB_RATIO 보다 작은 떨어진 부스러기를, 둘레에 가장 많이 닿은 조각에 붙인다."""
    ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for b in np.unique(labels[labels >= 0]):
        count, comp, stats, _ = cv2.connectedComponentsWithStats((labels == b).astype(np.uint8), connectivity=8)
        if count <= 2:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        for c in np.nonzero(areas < areas.max() * CRUMB_RATIO)[0] + 1:
            piece = comp == c
            around = labels[(cv2.dilate(piece.astype(np.uint8), ring) > 0) & ~piece]
            around = around[(around >= 0) & (around != b)]
            if len(around):
                labels[piece] = np.bincount(around).argmax()
    return labels


def enclosed_holes(mask):
    """mask 안에 갇힌 빈 곳 (바깥 테두리와 이어지지 않은 곳)."""
    h, w = mask.shape
    outside = np.pad((~mask).astype(np.uint8), 1, constant_values=1)
    flood = outside.copy()
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 2)
    return (flood[1:-1, 1:-1] == 1)


def make_parts(char_dir, rig, cfg):
    thr = cfg["alpha_threshold"]
    body = body_rig(rig)
    model = an.Model(char_dir, body, cfg)
    img = an.load_clean(char_dir / rig["image"], cfg)
    src = np.asarray(img)
    silhouette = src[..., 3] >= thr
    names = [b[0] for b in model.bones]
    _, joints, depth_of = an.rig_bones(body)

    hw = half_widths(char_dir, rig, cfg)
    f = float(np.linalg.norm(model.to_canvas([1, 0]) - model.to_canvas([0, 0])))  # 원본 → 캔버스 배율
    depth = np.array([depth_of[n] for n in names], float)
    reach = {names.index(b): hw[b] * LIMB_REACH * f for b in LIMBS if b in names}
    override = {names.index(b): hw[b] * f for b in SIDE_NEAR_ARM} if rig["view"] == "side" else {}
    dense = canvas_labels(model, cfg["mesh_step"], thr, depth, reach, override)
    labels = merge_crumbs(source_labels(model, dense, img, thr))
    present = [i for i in range(len(names)) if (labels == i).any()]

    x0, y0, x1, y1 = an.content_box(img, thr)
    height = y1 - y0
    kernel = int(round(0.08 * height)) | 1  # 가려진 곳을 찾는 닫기 크기 (키의 8%)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    bgr = np.ascontiguousarray(src[..., 2::-1])

    parts = []
    for i in present:
        name = names[i]
        mask = labels == i
        depth = depth_of[name]
        in_front = np.isin(labels, [j for j in present if depth_of[names[j]] > depth])

        # 가려진 곳: 조각 모양을 닫거나 속을 채웠을 때 새로 생기는 곳 중, 앞쪽 조각이 덮고 있는 곳
        closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, close_k) > 0
        closed |= enclosed_holes(closed)
        if name in HULL_PARTS:
            hull = cv2.convexHull(cv2.findNonZero(mask.astype(np.uint8)))
            hull_mask = np.zeros(mask.shape, np.uint8)
            cv2.fillConvexPoly(hull_mask, hull, 1)
            closed |= hull_mask > 0
        fill = closed & ~mask & in_front

        # 관절 덮개: 뼈 시작 관절 둘레의 이웃 픽셀을 원래 색 그대로 붙인다.
        # 크기는 이 조각 자신의 굵기 (팔다리는 잰 굵기, 나머지는 관절 근처 조각 안쪽 깊이).
        jx, jy = joints[[b for b in model.bones if b[0] == name][0][2]]
        jx, jy = int(np.clip(jx, 0, src.shape[1] - 1)), int(np.clip(jy, 0, src.shape[0] - 1))
        yy, xx = np.ogrid[:src.shape[0], :src.shape[1]]
        if name in hw:
            r = hw[name] * 1.25
        else:
            own = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
            near = (xx - jx) ** 2 + (yy - jy) ** 2 <= (0.05 * height) ** 2
            r = float(own[near].max()) if near.any() else 0
        r = float(np.clip(r, 3, 0.04 * height))
        cap = ((xx - jx) ** 2 + (yy - jy) ** 2 <= r * r) & silhouette & ~mask & ~fill

        out = np.zeros_like(src)
        keep = mask | cap
        out[keep] = src[keep]
        if fill.any():
            ys, xs = np.nonzero(fill | keep)
            bx0, by0 = max(xs.min() - 4, 0), max(ys.min() - 4, 0)
            bx1, by1 = min(xs.max() + 5, src.shape[1]), min(ys.max() + 5, src.shape[0])
            unknown = (~keep[by0:by1, bx0:bx1]).astype(np.uint8) * 255
            painted = cv2.inpaint(bgr[by0:by1, bx0:bx1], unknown, 3, cv2.INPAINT_TELEA)
            region = fill[by0:by1, bx0:bx1]
            out[by0:by1, bx0:bx1][region, :3] = painted[region][:, ::-1]
            out[fill, 3] = 255
        parts.append({"name": name, "depth": depth, "image": Image.fromarray(out),
                      "pixels": int(mask.sum()), "filled": int(fill.sum()), "cap": int(cap.sum()),
                      "fill_mask": fill, "cap_mask": cap})
    return parts, img


def _move(x, y):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], float)


def _premultiplied(img, tint=False):
    a = np.asarray(img).astype(np.float32) / 255
    a[..., :3] *= a[..., 3:4]
    if tint:
        a[..., :3] *= 0.65  # 옆 그림의 뒤쪽 팔다리 복사본은 어둡게
    return a


def _to_image(acc):
    alpha = acc[..., 3:4]
    rgb = np.divide(acc[..., :3], alpha, out=np.zeros_like(acc[..., :3]), where=alpha > 1e-6)
    return Image.fromarray((np.dstack([rgb, alpha]).clip(0, 1) * 255).round().astype(np.uint8), "RGBA")


class PartModel:
    """조각 방식 그리기. 조각마다 자기 뼈의 행렬로 딱딱하게 돌리고 옮겨, 앞뒤 순서(depth)대로 쌓는다.
    늘어나거나 눌리지 않으므로 크게 웅크리거나 숙여도 그림이 뭉개지지 않는다.
    animate.Model 과 같은 이름의 값·함수를 가져서 render 명령이 둘 중 아무거나 쓸 수 있다.

    층: 0번은 몸 조각들, 그 뒤로 rig 의 layers(덮개 그림을 조각 모양대로 나눠 같은 뼈에 붙인다),
    그 뒤로 attachments(칼처럼 휘지 않는 물건)."""

    def __init__(self, char_dir, rig, cfg):
        thr = cfg["alpha_threshold"]
        img = an.load_clean(char_dir / rig["image"], cfg)
        x0, y0, x1, y1, f, size, w, h, ox, oy = an.canvas_mapping(img, cfg, rig["image"])
        self.k, self.size, self.thr, self.view = cfg["render_scale"], size, thr, rig["view"]
        self.unit = an.length_unit(cfg)
        self.to_canvas = lambda p: (np.asarray(p, float) - [x0, y0]) * f + [ox, oy]

        def place(im):
            canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            canvas.paste(im.crop((x0, y0, x1, y1)).resize((w, h), Image.LANCZOS), (ox, oy))
            return canvas

        self.rest = place(img)
        body = body_rig(rig)
        bones, joints, depth_of = an.rig_bones(body)
        self.joints = {n: self.to_canvas(p) for n, p in joints.items()}
        attachments = rig.get("attachments", [])
        placed = an.place_attachments(attachments, bones, self.joints, depth_of, self.unit)
        self.weapon_bones = list(dict.fromkeys(a.get("slot", a["name"]) for a in attachments))
        # 틈 검사용 관절 원: 팔다리 뼈의 시작 관절, 반지름은 그 팔다리 굵기의 반 × 0.6
        hw = half_widths(char_dir, rig, cfg)
        self.gap_joints = [(name, j0, 0.6 * hw[name] * f) for name, _, j0, _ in bones if name in LIMBS]
        self.bones = bones
        self.base_pose = rig.get("base_pose", {})

        part_dir = char_dir / "parts" / self.view
        meta = json.loads((part_dir / "parts.json").read_text(encoding="utf-8"))
        part_imgs = {p["name"]: Image.open(part_dir / p["file"]).convert("RGBA") for p in meta["parts"]}
        part_depth = {p["name"]: p["depth"] for p in meta["parts"]}
        self.pieces = []

        def add(src_img, bone, depth, layer, tint=False):
            """원본 크기 조각 → 그림이 있는 영역만 캔버스 배율로 줄여 텍스처로 둔다."""
            a = np.asarray(src_img)
            ys, xs = np.nonzero(a[..., 3] > 0)
            if len(xs) == 0:
                return
            bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            tw, th = max(1, round((bx1 - bx0) * f)), max(1, round((by1 - by0) * f))
            tex = src_img.crop((bx0, by0, bx1, by1)).resize((tw, th), Image.LANCZOS)
            corner = self.to_canvas([bx0, by0])
            rest = _move(*corner) @ np.diag([(bx1 - bx0) * f / tw, (by1 - by0) * f / th, 1])
            self.pieces.append({"bone": bone, "depth": depth, "layer": layer, "rest": rest,
                                "tex": _premultiplied(tex, tint)})

        for name, im in part_imgs.items():
            add(im, name, part_depth[name], 0)
            if self.view == "side" and name in an.FAR_OF:
                add(im, an.FAR_OF[name], depth_of[an.FAR_OF[name]], 0, tint=True)

        # 덮개 층: 원본에서 맨 위에 보이는 조각(없으면 가장 가까운 조각)을 따라 나눈다.
        owner = np.full(img.size[::-1], -1, int)
        names = sorted(part_imgs, key=lambda n: part_depth[n])
        for i, n in enumerate(names):
            owner[np.asarray(part_imgs[n])[..., 3] > 0] = i
        _, nearest = cv2.distanceTransformWithLabels((owner < 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                     labelType=cv2.DIST_LABEL_PIXEL)
        owner = owner[owner >= 0][nearest - 1]
        self.layers = [{"name": "body", "slot": "body"}]
        self.layer_images = []
        for layer in rig.get("layers", []):
            limg = px.remove_background(Image.open(char_dir / layer["image"]), cfg["bg_tolerance"])
            if limg.size != img.size:
                sys.exit(f"{layer['image']}: 크기가 {rig['image']} 와 같아야 합니다 ({limg.size} != {img.size})")
            li = len(self.layers)
            self.layers.append({"name": layer["name"], "slot": layer.get("slot", layer["name"])})
            self.layer_images.append(place(limg))
            la = np.asarray(limg)
            for i, n in enumerate(names):
                piece = la.copy()
                piece[(owner != i) | (la[..., 3] == 0)] = 0
                add(Image.fromarray(piece), n, part_depth[n], li)
                if self.view == "side" and n in an.FAR_OF:
                    add(Image.fromarray(piece), an.FAR_OF[n], depth_of[an.FAR_OF[n]], li, tint=True)

        # 붙인 물건: 그림 속 grip→tip 방향을 손잡이→끝 방향에 맞춘다.
        self.attach_images = []
        for a, (grip, end) in zip(attachments, placed):
            li = len(self.layers)
            slot = a.get("slot", a["name"])
            self.layers.append({"name": a["name"], "slot": slot})
            im = Image.open(char_dir / a["image"]).convert("RGBA")
            g, t = np.array(a["grip"], float), np.array(a["tip"], float)
            scale = np.linalg.norm(end - grip) / np.linalg.norm(t - g)
            tw, th = max(1, round(im.width * scale)), max(1, round(im.height * scale))
            tex = im.resize((tw, th), Image.LANCZOS)
            ang = math.degrees(math.atan2(*(end - grip)[::-1]) - math.atan2(*(t - g)[::-1]))
            rest = _move(*grip) @ an._rot(ang) @ _move(*(-g * scale))
            self.pieces.append({"bone": slot, "depth": a.get("depth", 5), "layer": li, "rest": rest,
                                "tex": _premultiplied(tex)})
            self.attach_images.append(tex)

        # 같은 depth 안에서는 몸 → 덮개 → 물건, 그다음 넣은 순서대로 위에 쌓는다.
        self.pieces = [p for _, p in sorted(enumerate(self.pieces), key=lambda ip: (ip[1]["depth"], ip[1]["layer"], ip[0]))]
        first = {}
        for i, layer in enumerate(self.layers):
            first.setdefault(layer["slot"], i)
        self.slots = list(first)
        self.default_layers = sorted(first.values())

    def depth_levels(self):
        return sorted({p["depth"] for p in self.pieces})

    def render(self, pose, parts=False):
        """slot 마다 첫 번째 층을 합친 그림. parts=True 면 (층 번호, depth) 마다 따로 그린 그림도 함께 돌려준다."""
        mats = an.bone_matrices(self.bones, self.joints, pose, self.unit)
        size = self.size
        combined = np.zeros((size, size, 4), np.float32)
        groups = {}
        for p in self.pieces:
            m = (mats[p["bone"]] @ p["rest"])[:2]
            warped = cv2.warpAffine(p["tex"], m, (size, size), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if p["layer"] in self.default_layers:
                combined = warped + combined * (1 - warped[..., 3:4])
            if parts:
                key = (p["layer"], p["depth"])
                acc = groups.get(key)
                groups[key] = warped if acc is None else warped + acc * (1 - warped[..., 3:4])
        if not parts:
            return _to_image(combined)
        return _to_image(combined), {key: _to_image(acc) for key, acc in groups.items()}


def recompose(parts, size):
    canvas = Image.new("RGBA", size)
    for p in sorted(parts, key=lambda p: p["depth"]):
        canvas.alpha_composite(p["image"])
    return canvas


def mismatch(original, rebuilt, thr):
    """다시 쌓은 그림이 원본과 다른 픽셀 비율 (원본 그림 넓이 기준)."""
    a, b = np.asarray(original).astype(int), np.asarray(rebuilt).astype(int)
    sa, sb = a[..., 3] >= thr, b[..., 3] >= thr
    diff = (sa != sb) | (sa & sb & (np.abs(a[..., :3] - b[..., :3]).max(-1) > 8))
    return diff.sum() / max(sa.sum(), 1)


def draw_parts_check(parts, original, rebuilt):
    cell, cols = 200, 6
    rows = math.ceil(len(parts) / cols)
    sheet = Image.new("RGBA", (cell * cols, cell * rows + 420), (255, 255, 255, 255))
    d = ImageDraw.Draw(sheet)
    checker = Image.new("RGBA", (cell, cell - 20))
    cd = ImageDraw.Draw(checker)
    for y in range(0, cell, 10):
        for x in range(0, cell, 10):
            cd.rectangle([x, y, x + 9, y + 9], fill=(215, 215, 215, 255) if (x + y) // 10 % 2 else (240, 240, 240, 255))
    for n, p in enumerate(sorted(parts, key=lambda p: p["depth"])):
        a = np.asarray(p["image"]).copy()
        for m, color in ((p["fill_mask"], FILL_COLOR), (p["cap_mask"], CAP_COLOR)):
            a[m, :3] = (a[m, :3] * 0.4 + np.array(color) * 0.6).astype(np.uint8)
        ys, xs = np.nonzero(a[..., 3] > 0)
        crop = Image.fromarray(a).crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        crop.thumbnail((cell - 10, cell - 30))
        tile = checker.copy()
        tile.alpha_composite(crop, ((cell - crop.width) // 2, (cell - 20 - crop.height) // 2))
        x, y = (n % cols) * cell, (n // cols) * cell
        sheet.paste(tile, (x, y + 20))
        d.text((x + 4, y + 4), f"{p['name']}  d{p['depth']:g}  +{p['filled']}", fill=(0, 0, 0, 255))
    y = rows * cell + 10
    for k, (title, im) in enumerate((("original", original), ("parts rebuilt", rebuilt))):
        im = im.copy()
        im.thumbnail((400, 400))
        sheet.alpha_composite(im, (10 + k * 420, y + 10))
        d.text((10 + k * 420, y - 2), title, fill=(0, 0, 0, 255))
    return sheet


def cmd_parts(args):
    cfg = an.load_config(args.config)
    for view in an.VIEWS:
        rig_path = args.character / f"rig_{view}.json"
        if not rig_path.exists():
            continue
        out_dir = args.character / "parts" / view
        if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
            print(f"  {out_dir} 가 이미 있어 건너뜁니다 (손으로 고친 조각을 지키기 위해서. 다시 만들려면 --force)")
            continue
        rig = json.loads(rig_path.read_text(encoding="utf-8"))
        parts, original = make_parts(args.character, rig, cfg)

        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.png"):
            old.unlink()
        for p in parts:
            p["image"].save(out_dir / f"{p['name']}.png")
        meta = {"view": view, "image": rig["image"],
                "parts": [{"name": p["name"], "file": f"{p['name']}.png", "depth": p["depth"],
                           "pixels": p["pixels"], "filled": p["filled"], "cap": p["cap"]}
                          for p in sorted(parts, key=lambda p: p["depth"])]}
        (out_dir / "parts.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        rebuilt = recompose(parts, original.size)
        ratio = mismatch(original, rebuilt, cfg["alpha_threshold"])
        check_dir = args.character / "check"
        check_dir.mkdir(exist_ok=True)
        draw_parts_check(parts, original, rebuilt).save(check_dir / f"parts_{view}.png")
        filled = sum(p["filled"] for p in parts)
        print(f"  {view}: 조각 {len(parts)}개, 채운 픽셀 {filled}, 다시 쌓기 차이 {ratio:.2%}"
              f"  (확인 그림: {check_dir / f'parts_{view}.png'})")
    sys.stdout.flush()
