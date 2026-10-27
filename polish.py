"""픽셀로 줄인 그림의 마무리 손질. 어느 캐릭터에도 같은 규칙을 쓴다.

shade (config "shade"): 재질별 단계 색 + 빛 방향 음영
  1. 팔레트 색을 색상·채도로 묶어 재질을 나눈다. 같은 묶음 안에서 밝기가 크게 벌어지면 다른 재질로 쪼갠다.
  2. 재질마다 밝기 순으로 줄 세우고, 양 끝에 한 단계 밝은 색·어두운 색을 새로 만들어 붙인다
     (밝게는 노란 쪽, 어둡게는 푸른 쪽으로 색상을 살짝 튼다. 피부 그림자는 붉은 쪽).
  3. 빛은 왼쪽 위. 덩어리의 오른쪽·아래 가장자리와 큰 덩어리에 덮인 곳(앞머리 밑, 허리띠 밑)은 한 단계 어둡게,
     바깥·선에 닿은 왼쪽·위 가장자리는 한 단계 밝게. 작은 덩어리와 선은 그대로. 피부는 덮인 곳의 그림자만.

face (config "face"): 눈·입 다시 그리기
  자세 인식 모델(RTMW)의 얼굴 68점으로 원본에서 눈·입 자리를 찾고, 픽셀 그림의 그 자리에 정해 둔 틀을 찍는다.
  눈동자 색은 캐릭터 폴더 전체에서 가장 선명하게 뽑힌 색 하나를 3단으로 쓴다. 뒷모습은 그리지 않고, 옆모습은 눈 하나.
"""
import cv2
import numpy as np

import oklab

LINE_L = 0.22      # 이보다 어둡고 채도 낮은 색은 선
NEUTRAL_C = 0.04   # 무채색 판단 (아래 neutral_mask)
HUE_GAP = 12       # 색상(도)이 이만큼 가까우면 같은 묶음 후보
SAT_GAP = 0.08     # 채도/밝기 비율 차이가 이만큼 안쪽이면 같은 묶음 (어두울수록 채도가 낮아지는 것을 감안)
L_GAP = 0.14       # 묶음 안에서 밝기가 이만큼 벌어지면 다른 재질
STEP_L = 0.08      # 새로 만드는 단계 색의 밝기 차
MIN_AREA = 16      # 이보다 작은 덩어리는 음영을 넣지 않고, 그림자도 드리우지 않는다


def hex_rgb(text):
    text = text.strip().lstrip("#")
    return np.array([int(text[i:i + 2], 16) for i in (0, 2, 4)])


def is_skin(lab):
    """밝고 따뜻한 옅은 색 = 피부."""
    L, C, H = oklab.lch(lab)
    return bool(L > 0.75 and 0.025 < C < 0.1 and 30 < H < 80)


def shift(lab, dl):
    """밝기를 dl 만큼 옮긴 단계 색 (OKLab)."""
    L, C, H = oklab.lch(lab)
    skin = is_skin(lab) and dl < 0
    target = 110 if dl > 0 else (25 if skin else 265)
    d = (target - H + 180) % 360 - 180
    H2 = H + np.clip(d, -5, 5) * min(1, C / 0.05)   # 거의 무채색이면 거의 틀지 않는다
    C2 = C * (0.9 if dl > 0 else (1.4 if skin else 0.95))
    return np.array([np.clip(L + dl, 0.05, 0.99), C2 * np.cos(np.radians(H2)), C2 * np.sin(np.radians(H2))])


def is_line_lab(lab):
    L, C, _ = oklab.lch(lab)
    return (L < LINE_L) & (C < 0.05)


# ---------------------------------------------------------------- 재질 묶기와 음영

def materials(pal_rgb, outline_rgb):
    """팔레트 색마다 재질 번호. 선은 -1."""
    lab = oklab.srgb_to_oklab(pal_rgb)
    L, C, H = oklab.lch(lab)
    sat = C / np.maximum(L, 0.05)
    is_line = is_line_lab(lab) | (pal_rgb == outline_rgb).all(1)
    # 무채색: 채도가 아주 낮거나, 낮으면서 차가운(푸른·보라) 회색. 크림·베이지 같은 따뜻한 옅은 색은 유채색으로 남긴다
    cool = (H > 180) & (H < 330)
    is_neutral = (C < 0.02) | ((C < NEUTRAL_C) & (sat < 0.07) & cool)
    idx = [i for i in range(len(pal_rgb)) if not is_line[i]]
    groups = [[i for i in idx if is_neutral[i]]]
    chroma = [i for i in idx if not is_neutral[i]]

    parent = {i: i for i in chroma}

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i in chroma:   # 색상과 채도 비율이 둘 다 가까운 색끼리 사슬처럼 이어 한 묶음
        for j in chroma:
            dh = abs((H[i] - H[j] + 180) % 360 - 180)
            if i < j and dh <= HUE_GAP and abs(sat[i] - sat[j]) <= SAT_GAP:
                parent[find(i)] = find(j)
    roots = {}
    for i in chroma:
        roots.setdefault(find(i), []).append(i)
    groups += list(roots.values())

    def split(g):   # 크게 벌어진 밝기는 늘 나누고, 흰색~검은색처럼 폭이 아주 넓으면 가장 크게 벌어진 곳에서도 나눈다
        g = sorted(g, key=lambda i: L[i])
        if len(g) < 2:
            return [g]
        gaps = np.diff(L[g])
        k = int(gaps.argmax())
        if gaps[k] > L_GAP or (L[g[-1]] - L[g[0]] > 0.45 and gaps[k] > 0.06):
            return split(g[:k + 1]) + split(g[k + 1:])
        return [g]

    mat = np.full(len(pal_rgb), -1)
    n = 0
    for g in groups:
        for part in split(g) if g else []:
            mat[part] = n
            n += 1
    return mat, lab


def shade(img, palette, outline_rgb, fixed=None):
    """img(RGBA, 팔레트 색만 씀)에 재질별 빛 방향 음영을 넣은 새 그림.
    fixed(정해 둔 팔레트)를 주면 새 단계 색을 그 팔레트에서 가장 가까운 색으로 바꾼다."""
    pal_rgb = np.asarray(palette)
    mat, lab = materials(pal_rgb, outline_rgb)
    ramps = {}
    for m in set(mat) - {-1}:
        idx = sorted(np.flatnonzero(mat == m), key=lambda i: lab[i, 0])
        tones = [shift(lab[idx[0]], -STEP_L)] + [lab[i] for i in idx] + [shift(lab[idx[-1]], STEP_L)]
        rgb = oklab.oklab_to_srgb(np.array(tones))
        if fixed is not None:
            rgb = nearest_colors(rgb, fixed)
        ramps[m] = (idx, rgb, is_skin(lab[idx[-1]]))

    a = np.asarray(img).copy()
    h, w = a.shape[:2]
    solid = a[..., 3] > 0
    key = {tuple(c): k for k, c in enumerate(pal_rgb)}
    pidx = np.full((h, w), -1)
    for y, x in zip(*np.nonzero(solid)):
        pidx[y, x] = key.get(tuple(a[y, x, :3]), -1)
    cell_mat = np.where(pidx >= 0, mat[np.maximum(pidx, 0)], -1)
    tone = np.zeros((h, w), int)
    for m, (idx, _, _) in ramps.items():
        for t, i in enumerate(idx, start=1):
            tone[pidx == i] = t

    open_pad = np.pad(~solid | (cell_mat == -1), 1, constant_values=True)   # 바깥·선
    big = np.zeros_like(solid)   # 큰 덩어리 칸: 그림자를 드리울 수 있는 쪽
    comps = {}
    for m in ramps:
        nlab, lab_map, stats, _ = cv2.connectedComponentsWithStats((cell_mat == m).astype(np.uint8), connectivity=4)
        comps[m] = (nlab, lab_map, stats)
        for r in range(1, nlab):
            if stats[r, cv2.CC_STAT_AREA] >= MIN_AREA:
                big |= lab_map == r
    big_up = np.pad(big, 1)[:-2, 1:-1]

    new_tone = tone.copy()
    for m, (nlab, lab_map, stats) in comps.items():
        skin = ramps[m][2]
        for r in range(1, nlab):
            if stats[r, cv2.CC_STAT_AREA] < MIN_AREA:
                continue
            reg = lab_map == r
            # 안쪽 구멍(눈·입 등)은 덩어리의 일부로 보고 가장자리를 바깥 둘레에서만 잰다
            inv = np.pad(~reg, 1, constant_values=True).astype(np.uint8)
            cv2.floodFill(inv, None, (0, 0), 2)
            filled = reg | (inv[1:-1, 1:-1] == 1)
            pad = np.pad(filled, 1)
            out_r, out_d = ~pad[1:-1, 2:], ~pad[2:, 1:-1]
            out_l, out_u = ~pad[1:-1, :-2], ~pad[:-2, 1:-1]
            open_l, open_u = open_pad[1:-1, :-2], open_pad[:-2, 1:-1]
            cast = reg & out_u & big_up
            if skin:   # 피부는 덮인 곳의 그림자만. 얼굴 둘레를 어둡게 하면 지저분해진다
                dark, light = cast, np.zeros_like(reg)
            else:
                dark = reg & (out_r | out_d | cast) & ~(out_l & out_r)
                light = reg & ((out_l & open_l) | (out_u & open_u)) & ~dark & ~(out_u & out_d)
            new_tone[dark] -= 1
            new_tone[light] += 1

    for m, (idx, rgb, _) in ramps.items():
        sel = cell_mat == m
        a[sel, :3] = rgb[np.clip(new_tone[sel], 0, len(rgb) - 1)]
    return a


def nearest_colors(rgb, fixed):
    lab, flab = oklab.srgb_to_oklab(rgb), oklab.srgb_to_oklab(fixed)
    return np.asarray(fixed)[((lab[:, None] - flab[None]) ** 2).sum(-1).argmin(1)]


# ---------------------------------------------------------------- 눈·입

EYE_A, EYE_B = slice(59, 65), slice(65, 71)   # COCO 전신 133점 중 얼굴 68점의 두 눈 윤곽 (각 6점, 0번·3번이 눈꼬리)
MOUTH = slice(71, 83)                          # 입술 바깥 윤곽 12점
LASH, WHITE, SCLERA = (4, 3, 3), (251, 251, 250), (237, 233, 231)


def view_kind(name):
    """파일 이름으로 방향 종류: back(눈 없음) / side(눈 하나) / front."""
    parts = name.lower().split("_")
    if "back" in parts:
        return "back"
    if "side" in parts or (parts[-1] in ("left", "right") and "front" not in parts):
        return "side"
    return "front"


def find_face(img, threshold=128):
    """원본(배경 지운 RGBA)에서 눈 둘(윤곽 6점씩)과 입(12점)."""
    import animate as an   # 무거운 자세 인식 모델은 필요할 때만 불러온다
    model = an.pose_model()
    bg = _on_gray(img)
    x0, y0, x1, y1 = an.content_box(img, threshold)
    m = 0.05 * max(x1 - x0, y1 - y0)
    kps, scores = model(bg, bboxes=[[x0 - m, y0 - m, x1 + m, y1 + m]])
    kp = np.asarray(kps[0], float)
    eyes = [{"pts": kp[sl], "outer": kp[sl][o], "score": float(scores[0][sl].mean())} for sl, o in ((EYE_A, 0), (EYE_B, 3))]
    return {"eyes": eyes, "mouth": kp[MOUTH]}


def _on_gray(img):
    from PIL import Image
    bg = Image.new("RGBA", img.size, (230, 230, 230, 255))
    bg.alpha_composite(img)
    return np.asarray(bg.convert("RGB"))[..., ::-1].copy()


def iris_color(img, pts):
    """눈 윤곽 안의 눈동자 색 (OKLab): 선·동공·흰자를 빼고, 채도 높은 쪽 중 밝은 쪽의 가운데 값
    (어두운 쪽까지 섞으면 원본보다 칙칙한 색이 나온다)."""
    a = np.asarray(img).astype(int)
    x0, y0 = np.floor(pts.min(0)).astype(int)
    x1, y1 = np.ceil(pts.max(0)).astype(int)
    patch = a[max(0, y0):y1, max(0, x0):x1]
    rgb = patch[..., :3][patch[..., 3] > 0]
    if len(rgb) == 0:
        return None
    lab = oklab.srgb_to_oklab(rgb)
    L, C, _ = oklab.lch(lab)
    keep = (L > 0.3) & (L < 0.85)
    if keep.sum() < 5:
        return None
    lab, L, C = lab[keep], L[keep], C[keep]
    vivid = C >= np.median(C)
    lab, L = lab[vivid], L[vivid]
    return np.median(lab[L >= np.median(L)], axis=0)


def iris_tones(color_lab):
    return oklab.oklab_to_srgb(np.array([shift(color_lab, -0.18), color_lab, shift(color_lab, 0.12)]))


def _is_line(px):
    return is_line_lab(oklab.srgb_to_oklab(np.asarray(px)[..., :3])) & (np.asarray(px)[..., 3] > 0)


def _is_white(px):
    L, C, _ = oklab.lch(oklab.srgb_to_oklab(np.asarray(px)[..., :3]))
    return (L > 0.9) & (C < 0.02)


def _stamp_eye(out, bx, top, w, outer_left, tones):
    """눈 틀 (바깥쪽 → 안쪽): 맨 위 속눈썹, 그 아래 3줄 눈동자.
    눈동자 첫 줄 가운데 흰 점, 둘째 줄 가운데 동공, 셋째 줄 가운데는 밝게. 폭 4칸 이상이면 바깥 1열은 속눈썹 끝·흰자."""
    dark, mid, light = tones
    cols = list(range(w)) if outer_left else list(range(w))[::-1]
    iris = cols[1:] if w >= 4 else cols
    hi = iris[(len(iris) - 1) // 2] if outer_left else iris[len(iris) // 2]
    for c in cols:
        out[top, bx + c] = [*LASH, 255]
    for r, tone in enumerate((dark, mid, mid)):
        for c in iris:
            out[top + 1 + r, bx + c] = [*tone, 255]
    out[top + 1, bx + hi] = [*WHITE, 255]
    out[top + 2, bx + hi] = [*dark, 255]
    out[top + 3, bx + hi] = [*light, 255]
    if w >= 4:
        out[top + 1, bx + cols[0]] = [*LASH, 255]
        out[top + 2, bx + cols[0]] = [*SCLERA, 255]


def _common(out, cells):
    vals, counts = np.unique(np.array([out[y, x, :3] for x, y in cells]), axis=0, return_counts=True)
    return vals[counts.argmax()]


def draw_face(out, face, to_cell, tones, kind):
    """out(RGBA 배열)에 눈·입을 찍는다. to_cell: 원본 좌표 → 픽셀 칸 좌표. 찍은 내용을 글로 돌려준다."""
    if kind == "back":
        return []
    h, w = out.shape[:2]
    boxes = []
    for e in face["eyes"]:
        cells = np.array([to_cell(p) for p in e["pts"]])
        ew = int(np.clip(round(np.ptp(cells[:, 0]) + 0.5), 2, 5))
        cx = cells[:, 0].mean()
        bx = int(round(cx - ew / 2))
        top = int(round(cells[[1, 2, 4, 5], 1].min() - 0.5))   # 윗눈꺼풀 = 속눈썹 줄
        # 이미 그려진 속눈썹 줄(선 색 칸이 가장 많이 이어진 줄)에 맞춘다
        best = None
        for r in range(top - 2, top + 2):
            n = int(_is_line(out[r, max(0, bx - 1):bx + ew + 1]).sum())
            if n >= 2 and (best is None or n > best[0]):
                best = (n, r)
        top = best[1] if best else top
        lashes = int(_is_line(out[top, bx:bx + ew]).sum())
        boxes.append((lashes, bx, top, ew, to_cell(e["outer"])[0] < cx))
    if kind == "side" or (len(boxes) == 2 and abs(boxes[0][1] - boxes[1][1]) < max(boxes[0][3], boxes[1][3])):
        boxes = [max(boxes, key=lambda b: b[0])]   # 옆모습·겹친 눈: 속눈썹이 실제로 그려진 눈 하나만
    notes = []
    for lashes, bx, top, ew, outer_left in boxes:
        if not (0 < bx and bx + ew < w - 1 and 0 < top and top + 5 < h):
            continue
        skin = _common(out, [(x, top + 5) for x in range(bx, bx + ew)])
        _stamp_eye(out, bx, top, ew, outer_left, tones)
        # 둘레에 남은 예전 흰자·검은 칸을 피부색으로
        for r in range(top + 1, top + 5):
            for x in (bx - 1, bx + ew) if r <= top + 3 else range(bx - 1, bx + ew + 1):
                if out[r, x, 3] == 0:
                    continue
                near_skin = sum((out[r + dy, x + dx, :3] == skin).all() for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
                if _is_white(out[r, x]) or (r > top + 1 and _is_line(out[r, x]) and near_skin >= 2):
                    out[r, x, :3] = skin
        notes.append(f"눈({bx},{top}) 폭{ew}")

    # 입: 입술 윤곽 가운데, 피부 위에만. 앞은 2칸, 옆은 1칸, 피부 그림자색으로
    mc = np.mean([to_cell(p) for p in face["mouth"]], axis=0)
    mw = 1 if kind == "side" else 2
    mx, my = int(round(mc[0] - mw / 2)), int(round(mc[1] - 0.5))
    cells = [(mx + i, my) for i in range(mw)]
    if all(0 <= x < w and 0 <= my < h and out[my, x, 3] > 0 for x, _ in cells):
        skin = _common(out, [(x, my + dy) for x, _ in cells for dy in (-1, 1)])
        skin_lab = oklab.srgb_to_oklab(skin)
        if is_skin(skin_lab):
            mouth = oklab.oklab_to_srgb(shift(skin_lab, -0.2))
            for x, y in cells:
                out[y, x, :3] = mouth
            notes.append(f"입({mx},{my})")
    return notes
