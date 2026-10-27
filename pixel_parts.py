"""픽셀 조각 방식: 조각마다 픽셀 그림을 한 번만 만들어 두고, 움직일 때는 그 픽셀 조각을 RotSprite 로 돌린다.

예전 방식은 프레임마다 큰 그림을 새로 그려 줄였다. 그러면 같은 옷자락도 프레임마다 다른 칸에 걸쳐
색이 바뀌어 픽셀이 깜빡인다. 이 방식은 조각의 픽셀이 정해져 있어서, 움직여도 같은 픽셀 덩어리가 따라간다.

1. 쉼 자세에서 조각 하나씩을 최종 칸(캔버스)에 맞춰 줄인다 (칸마다 가장 많은 팔레트 색).
   한 칸을 여러 조각이 나눠 덮어 어느 조각도 절반이 안 되면, 합친 그림이 그 칸을 칠할 때 가장 많이 덮은 조각이 맡는다
   (조각 사이에 빈칸이 생기지 않게).
2. 움직일 때 (RotSprite): 픽셀 조각을 Scale2x 로 render_scale 배(8배) 키운 뒤, 뼈 행렬대로 가장 가까운 칸을 그대로
   가져오며 돌리고(색을 섞지 않는다), 칸마다 가장 많은 색으로 다시 줄인다 (칸의 3/8 이상 덮이면 칠한다).
   원래 RotSprite 는 칸 가운데 점 하나를 고르는데, 보는 사람 쪽으로 기울어 짧아진 조각(칼 등)은 그러면 줄이 점선처럼 끊긴다.
3. 앞뒤 순서(depth + z)대로 쌓는다. 붙인 물건의 각도별 그림(variants)도 같은 규칙으로 고른다.
"""
import cv2
import numpy as np
from PIL import Image

import animate as an

EMPTY = 255  # 색 번호 그림에서 빈칸
COVER = 3 / 8  # 줄일 때 칸을 칠하는 덮인 비율
COVER_THIN = 1 / 16  # 칼처럼 가는 물건: 날을 옆에서 보아 아주 얇아져도 1칸 선으로 남게


def scale2x(idx):
    """Scale2x(EPX): 픽셀 그림을 2배로 키우되, 대각선 계단을 매끄럽게 잇는다. 색 번호 그림에 쓴다."""
    p = np.pad(idx, 1, mode="edge")
    e = p[1:-1, 1:-1]
    b, h = p[:-2, 1:-1], p[2:, 1:-1]   # 위, 아래
    d, f = p[1:-1, :-2], p[1:-1, 2:]   # 왼쪽, 오른쪽
    out = np.empty((idx.shape[0] * 2, idx.shape[1] * 2), idx.dtype)
    out[0::2, 0::2] = np.where((d == b) & (b != f) & (d != h), d, e)
    out[0::2, 1::2] = np.where((b == f) & (b != d) & (f != h), f, e)
    out[1::2, 0::2] = np.where((d == h) & (d != b) & (h != f), d, e)
    out[1::2, 1::2] = np.where((h == f) & (d != h) & (b != f), f, e)
    return out


def upscale(idx, k):
    """k 배로 키운다. k 가 2의 거듭제곱이면 Scale2x 를 거듭, 아니면 그냥 늘린다."""
    out = idx
    while k > 1 and k % 2 == 0:
        out = scale2x(out)
        k //= 2
    if k > 1:
        out = np.repeat(np.repeat(out, k, axis=0), k, axis=1)
    return out


class PixelParts:
    """PartModel 의 조각들을 픽셀 조각으로 바꿔 두고, 자세마다 RotSprite 로 그린다."""

    def __init__(self, model, palette, cfg):
        self.model = model
        self.palette = np.asarray(palette, np.int32)
        k, size = model.k, model.size
        n = size // k
        self.k, self.n = k, n
        thr = cfg["alpha_threshold"] / 255
        ncol = len(self.palette)

        # 1) 조각마다 쉼 자리에 그려 칸별로 덮은 비율과 가장 많은 색을 잰다
        covs, modes, layer_solid = [], [], {}
        for p in model.pieces:
            warped = cv2.warpAffine(p["tex"], p["rest"][:2], (size, size), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            alpha = warped[..., 3]
            solid = alpha >= thr
            covs.append(solid.reshape(n, k, n, k).mean(axis=(1, 3)))
            if (p.get("show") or [-1])[0] < 0:  # 각도별로 바꿔 끼우는 그림은 빼고
                layer_solid[p["layer"]] = layer_solid.get(p["layer"], False) | solid
            idx = np.zeros((size, size), np.int64)
            if solid.any():
                rgb = warped[..., :3][solid] / alpha[solid][:, None] * 255  # 알파를 곱한 색 → 원래 색
                idx[solid] = an.px.nearest_index(rgb.clip(0, 255), self.palette)
            block = (np.arange(size)[:, None] // k) * n + (np.arange(size)[None, :] // k)
            counts = np.bincount((block[solid] * ncol + idx[solid]), minlength=n * n * ncol).reshape(n * n, ncol)
            modes.append(counts.argmax(axis=1).reshape(n, n))
        covs = np.array(covs)

        # 층의 조각을 합치면 절반 넘게 덮는데 어느 조각도 절반을 못 덮은 칸은, 그 층에서 가장 많이 덮은 조각이 맡는다.
        # 층마다 따로 정해서, 게임에서 층을 골라 쌓아도 조각 사이에 빈칸이 생기지 않는다
        take = covs >= 0.5
        own = np.array([p["layer"] for p in model.pieces])
        base = np.array([(p.get("show") or [-1])[0] < 0 for p in model.pieces])
        for layer, solid in layer_solid.items():
            sel = (own == layer) & base
            union = solid.reshape(n, k, n, k).mean(axis=(1, 3)) >= 0.5
            owner = np.where(sel[:, None, None], covs, -1).argmax(axis=0)
            orphan = union & ~(take & sel[:, None, None]).any(axis=0)
            for i in np.nonzero(sel)[0]:
                take[i] |= orphan & (owner == i)

        # 장비 둘레 외곽선 색 (팔레트 끝에 더한다. 조각 색은 위에서 원래 팔레트로만 골랐다)
        self.outline = None
        if cfg["outline"]:
            self.outline = len(self.palette)
            self.palette = np.vstack([self.palette, an.px.hex_to_rgb(cfg["outline_color"])])

        self.pieces = []
        for i, p in enumerate(model.pieces):
            small = np.full((n, n), EMPTY, np.uint8)
            small[take[i]] = modes[i][take[i]]
            self.pieces.append({**{key: p[key] for key in ("bone", "depth", "layer")}, "show": p.get("show"),
                                "small": small, "big": upscale(small, k),
                                "cover": COVER_THIN if p["bone"] in model.weapon_bones else COVER})

    def reduce(self, big, cover=COVER):
        """k 배 색 번호 그림 → 최종 칸: 칸마다 가장 많은 색, 덮인 비율이 cover 보다 작으면 빈칸."""
        k, n = self.k, self.n
        blocks = big.reshape(n, k, n, k).transpose(0, 2, 1, 3).reshape(n * n, k * k)
        filled = blocks != EMPTY
        out = np.full(n * n, EMPTY, np.uint8)
        rows = np.nonzero(filled.mean(axis=1) >= cover)[0]
        if len(rows):
            sub = blocks[rows]
            counts = np.zeros((len(rows), 256), np.int32)
            np.add.at(counts, (np.repeat(np.arange(len(rows)), k * k), sub.ravel()), 1)
            counts[:, EMPTY] = 0
            out[rows] = counts.argmax(axis=1)
        return out.reshape(n, n)

    def to_rgba(self, idx):
        out = np.zeros((*idx.shape, 4), np.uint8)
        m = idx != EMPTY
        out[m, :3] = self.palette[idx[m]]
        out[m, 3] = 255
        return Image.fromarray(out, "RGBA")

    def render(self, pose, parts=False):
        """합친 그림(최종 칸, 바깥 외곽선 없음). parts=True 면 (층 번호, 뼈) 묶음마다 그린 그림도
        뒤에서 앞 순서의 dict 로 함께 돌려준다. 묶음을 이 순서로 쌓으면 합친 그림과 똑같다.
        장비 층(slot 이 body·head 가 아닌 층)은 둘레 중 몸 위에 놓이는 칸에 1칸 외곽선을 넣는다
        (장비가 몸에 묻히지 않게. 몸은 모든 캐릭터가 같으므로 미리 넣어 둘 수 있다)."""
        m = self.model
        n, size = self.n, m.size
        mats, chain = an.bone_matrices(m.bones, m.joints, pose, m.unit, with_chain=True)
        groups = {}
        for i in m.order(self.pieces, pose):
            p = self.pieces[i]
            bone_m = chain[p["bone"]] if (p["show"] or [-1])[0] >= 0 else mats[p["bone"]]
            warped = cv2.warpAffine(p["big"], bone_m[:2], (size, size), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=EMPTY)
            small = self.reduce(warped, p["cover"])
            hit = small != EMPTY
            g = groups.setdefault((p["layer"], p["bone"]), np.full((n, n), EMPTY, np.uint8))
            g[hit] = small[hit]

        if self.outline is not None:
            body = np.zeros((n, n), bool)
            for (li, _), g in groups.items():
                if m.layers[li]["slot"] == "body":
                    body |= g != EMPTY
            for li, layer in enumerate(m.layers):
                if layer["slot"] in ("body", "head"):
                    continue
                keys = [key for key in groups if key[0] == li]
                mine = np.zeros((n, n), bool)
                for key in keys:
                    mine |= groups[key] != EMPTY
                ring = near4(mine) & ~mine & body
                for key in keys:  # 외곽선 칸은 닿은 묶음이 맡는다 (그 묶음의 앞뒤 순서로 그려진다)
                    spot = near4(groups[key] != EMPTY) & ring
                    groups[key][spot] = self.outline
                    ring &= ~spot

        combined = np.full((n, n), EMPTY, np.uint8)
        for (li, _), g in groups.items():
            if li in m.default_layers:
                combined[g != EMPTY] = g[g != EMPTY]
        img = self.to_rgba(combined)
        if not parts:
            return img
        return img, {key: self.to_rgba(g) for key, g in groups.items()}


def near4(mask):
    """위아래좌우로 mask 에 닿은 칸 (mask 자신 포함)."""
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out
