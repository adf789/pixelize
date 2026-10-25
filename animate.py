"""T자 일러스트에 뼈대를 심어 동작을 만들고, 프레임마다 64칸 픽셀 그림으로 찍는 도구.

사용법:
    python animate.py rig 캐릭터폴더 [-c config.json]
        front.png / back.png / side.png 에서 관절을 찾아 rig_<방향>.json 과 확인 그림 check/rig_<방향>.png 를 만든다.
    python animate.py check 캐릭터폴더
        rig_<방향>.json 을 손으로 고친 뒤 확인 그림만 다시 그린다.
    python animate.py render 캐릭터폴더 -o 출력폴더 [-c config.json] [-a anims] [--only front/walk]
        rig_<방향>.json 과 anims/<방향>/*.json 으로 동작을 만들어 픽셀 스프라이트 시트로 내보낸다.

그림은 자르지 않는다. 그림 전체에 촘촘한 그물(메시)을 씌우고, 그물의 점마다 가까운 뼈를 따라가게 해서
고무판처럼 휘게 한다. 옆 그림은 다리가 겹쳐 있으므로 보이는 팔다리(_l)를 복사해 뒤쪽 팔다리(_r)로 쓴다.
"""
import argparse
import heapq
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # 내장 Python은 스크립트 폴더를 자동으로 찾지 않는다
import pixelize as px  # noqa: E402
VIEWS = ("front", "back", "side")
ANIM_DEFAULTS = {"render_scale": 8, "mesh_step": 4}
MODEL_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
             "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip")

# (뼈, 부모, 시작 관절, 끝 관절). 부모가 항상 먼저 나온다.
BONES = [
    ("torso", None, "pelvis", "neck"),
    ("head", "torso", "neck", "head_center"),
    ("upper_arm_l", "torso", "shoulder_l", "elbow_l"),
    ("forearm_l", "upper_arm_l", "elbow_l", "wrist_l"),
    ("hand_l", "forearm_l", "wrist_l", "hand_tip_l"),
    ("upper_arm_r", "torso", "shoulder_r", "elbow_r"),
    ("forearm_r", "upper_arm_r", "elbow_r", "wrist_r"),
    ("hand_r", "forearm_r", "wrist_r", "hand_tip_r"),
    ("thigh_l", None, "hip_l", "knee_l"),
    ("shin_l", "thigh_l", "knee_l", "ankle_l"),
    ("foot_l", "shin_l", "ankle_l", "toe_l"),
    ("thigh_r", None, "hip_r", "knee_r"),
    ("shin_r", "thigh_r", "knee_r", "ankle_r"),
    ("foot_r", "shin_r", "ankle_r", "toe_r"),
]
BONE_NAMES = [b[0] for b in BONES]
LIMBS_L = ["upper_arm_l", "forearm_l", "hand_l", "thigh_l", "shin_l", "foot_l"]
FAR_OF = {name: name[:-2] + "_r" for name in LIMBS_L}  # 옆 그림: 보이는 팔다리 → 뒤쪽 복사본

# 그리는 순서 (큰 수가 앞). rig 파일에서 고칠 수 있다.
DEFAULT_DEPTH = {
    "front": {"thigh": 1, "shin": 1, "foot": 1, "torso": 2, "head": 3, "upper_arm": 4, "forearm": 4, "hand": 4},
    "back": {"thigh": 1, "shin": 1, "foot": 1, "torso": 2, "head": 3, "upper_arm": 4, "forearm": 4, "hand": 4},
    "side": {"torso": 3, "head": 4},
}
SIDE_LIMB_DEPTH = {"upper_arm": (0, 5), "forearm": (0, 5), "hand": (0, 5),
                   "thigh": (1, 2), "shin": (1, 2), "foot": (1, 2)}  # (뒤쪽 _r, 앞쪽 _l)

# 자세 인식 모델(COCO 전신 133점) 번호
KP = {"nose": 0, "eye_l": 1, "eye_r": 2, "ear_l": 3, "ear_r": 4,
      "shoulder_l": 5, "shoulder_r": 6, "elbow_l": 7, "elbow_r": 8, "wrist_l": 9, "wrist_r": 10,
      "hip_l": 11, "hip_r": 12, "knee_l": 13, "knee_r": 14, "ankle_l": 15, "ankle_r": 16,
      "bigtoe_l": 17, "smalltoe_l": 18, "bigtoe_r": 20, "smalltoe_r": 21}

BONE_COLORS = [(230, 60, 60), (240, 200, 40), (60, 140, 230), (40, 200, 220), (150, 230, 255),
               (60, 200, 90), (150, 230, 120), (210, 255, 190), (200, 80, 220), (240, 140, 240),
               (255, 210, 250), (240, 130, 40), (250, 180, 110), (255, 230, 200),
               (120, 90, 255), (255, 255, 90), (140, 70, 30), (0, 150, 130)]


# ---------------------------------------------------------------- 공통

def load_config(path):
    cfg = px.load_config(path)
    for key, value in ANIM_DEFAULTS.items():
        cfg.setdefault(key, value)
    return cfg


def load_clean(path, cfg):
    """배경과 떨어진 조각을 지운 RGBA 그림."""
    img = px.remove_background(Image.open(path), cfg["bg_tolerance"])
    return px.remove_small_parts(img, cfg["alpha_threshold"], cfg["min_part_ratio"])


def content_box(img, threshold):
    ys, xs = np.nonzero(np.asarray(img)[..., 3] >= threshold)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def view_images(char_dir):
    return {v: char_dir / f"{v}.png" for v in VIEWS if (char_dir / f"{v}.png").exists()}


# ---------------------------------------------------------------- rig: 관절 찾기

_pose_model = None


def pose_model():
    global _pose_model
    if _pose_model is None:
        os.environ.setdefault("TORCH_HOME", str(HERE / "models"))
        import onnxruntime  # 무거워서 필요할 때만 불러온다
        from rtmlib import RTMPose
        onnxruntime.set_default_logger_severity(3)
        _pose_model = RTMPose(MODEL_URL, model_input_size=(288, 384))
    return _pose_model


def detect_joints(img, view, threshold):
    model = pose_model()
    bg = Image.new("RGBA", img.size, (230, 230, 230, 255))
    bg.alpha_composite(img)
    bgr = np.asarray(bg.convert("RGB"))[..., ::-1].copy()
    x0, y0, x1, y1 = content_box(img, threshold)
    m = 0.05 * max(x1 - x0, y1 - y0)
    kps, scores = model(bgr, bboxes=[[x0 - m, y0 - m, x1 + m, y1 + m]])
    kp, sc = kps[0], scores[0]

    def pt(name):
        return np.array(kp[KP[name]], dtype=float)

    if view == "side":
        # 겹친 팔다리는 점수가 높은 쪽 하나만 믿고, 왼쪽·오른쪽을 같은 자리로 둔다.
        def pick(part):
            a, b = KP[part + "_l"], KP[part + "_r"]
            return np.array(kp[a] if sc[a] >= sc[b] else kp[b], dtype=float)
        p = {part: pick(part) for part in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")}
        toe_i = max(["l", "r"], key=lambda s: sc[KP["bigtoe_" + s]])
        toe = (pt("bigtoe_" + toe_i) + pt("smalltoe_" + toe_i)) / 2
        side = {part: (p[part], p[part]) for part in p}
        toes = (toe, toe)
    else:
        side = {part: (pt(part + "_l"), pt(part + "_r"))
                for part in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")}
        toes = ((pt("bigtoe_l") + pt("smalltoe_l")) / 2, (pt("bigtoe_r") + pt("smalltoe_r")) / 2)

    j = {}
    for i, s in enumerate(("l", "r")):
        for part, pair in side.items():
            j[f"{part}_{s}"] = pair[i]
        j[f"toe_{s}"] = toes[i]
        j[f"hand_tip_{s}"] = j[f"wrist_{s}"] + 0.4 * (j[f"wrist_{s}"] - j[f"elbow_{s}"])
    j["pelvis"] = (j["hip_l"] + j["hip_r"]) / 2
    j["neck"] = (j["shoulder_l"] + j["shoulder_r"]) / 2
    j["head_center"] = np.mean([pt(n) for n in ("nose", "eye_l", "eye_r", "ear_l", "ear_r")], axis=0)
    return {k: [round(float(v[0]), 1), round(float(v[1]), 1)] for k, v in j.items()}


def default_base_pose(joints, view):
    """T자로 벌린 팔을 아래로 내리는 기본 자세. 팔이 45도보다 많이 벌어져 있을 때만 내린다."""
    pose = {}
    if view == "side":
        return pose
    neck_x = joints["neck"][0]
    for s in ("l", "r"):
        sx, sy = joints[f"shoulder_{s}"]
        ex, ey = joints[f"elbow_{s}"]
        angle = math.degrees(math.atan2(ey - sy, ex - sx))  # 아래 = 90
        if abs(angle - 90) <= 45:
            continue
        target = 90 - 8 if sx > neck_x else 90 + 8  # 몸 바깥쪽으로 살짝 벌린다
        pose[f"upper_arm_{s}"] = {"rot": round((target - angle + 180) % 360 - 180, 1)}
    return pose


def default_depth(view):
    depth = {}
    for name in BONE_NAMES:
        kind = name.rsplit("_", 1)[0] if name.endswith(("_l", "_r")) else name
        if view == "side" and kind in SIDE_LIMB_DEPTH:
            depth[name] = SIDE_LIMB_DEPTH[kind][0 if name.endswith("_r") else 1]
        else:
            depth[name] = DEFAULT_DEPTH[view][kind]
    return depth


def rig_bones(rig):
    """기본 뼈 + rig 파일의 extra_bones (망토, 꼬리, 무기처럼 따로 움직일 부분).
    추가 뼈의 관절은 '<이름>_start', '<이름>_end' 로 붙인다."""
    bones, joints, depth = list(BONES), dict(rig["joints"]), dict(rig["depth"])
    for e in rig.get("extra_bones", []):
        bones.append((e["name"], e.get("parent"), e["name"] + "_start", e["name"] + "_end"))
        joints[e["name"] + "_start"], joints[e["name"] + "_end"] = e["start"], e["end"]
        depth[e["name"]] = e.get("depth", 0)
    return bones, joints, depth


def cmd_rig(args):
    cfg = load_config(args.config)
    for view, path in view_images(args.character).items():
        img = load_clean(path, cfg)
        joints = detect_joints(img, view, cfg["alpha_threshold"])
        rig = {"image": path.name, "view": view, "joints": joints,
               "base_pose": default_base_pose(joints, view), "depth": default_depth(view)}
        rig_path = args.character / f"rig_{view}.json"
        text = json.dumps(rig, indent=2, ensure_ascii=False)
        text = re.sub(r"\[\s+(-?[\d.]+),\s+(-?[\d.]+)\s+\]", r"[\1, \2]", text)  # 좌표는 한 줄로
        rig_path.write_text(text, encoding="utf-8")
        print(f"  {rig_path}  (확인 그림: {save_rig_check(args.character, rig, cfg)})")


def cmd_check(args):
    """rig 파일을 손으로 고친 뒤 확인 그림만 다시 그린다 (모델을 다시 돌리지 않는다)."""
    cfg = load_config(args.config)
    for view in VIEWS:
        rig_path = args.character / f"rig_{view}.json"
        if rig_path.exists():
            rig = json.loads(rig_path.read_text(encoding="utf-8"))
            print(f"  {save_rig_check(args.character, rig, cfg)}")


def save_rig_check(char_dir, rig, cfg):
    check_dir = char_dir / "check"
    check_dir.mkdir(exist_ok=True)
    path = check_dir / f"rig_{rig['view']}.png"
    draw_rig_check(char_dir / rig["image"], rig, cfg).save(path)
    return path


# ---------------------------------------------------------------- 메시와 무게

class Model:
    """한 방향 그림의 쉼 자세 그림, 관절, 메시, 무게를 그리기 크기에 맞춰 준비한다.

    층: 0번은 몸(rig 의 image), 그 뒤로 rig 의 layers(갑옷·투구처럼 몸 그림에 맞춰 그린 덮개),
    그 뒤로 attachments(칼처럼 휘지 않는 물건). 같은 slot 의 층들은 서로 바꿔 끼우는 변형이고,
    합친 그림에는 slot 마다 첫 번째 층만 들어간다."""

    def __init__(self, char_dir, rig, cfg):
        k, thr = cfg["render_scale"], cfg["alpha_threshold"]
        img = load_clean(char_dir / rig["image"], cfg)
        x0, y0, x1, y1 = content_box(img, thr)
        f = cfg["char_height"] * k / (y1 - y0)
        size = cfg["canvas"] * k
        w, h = round((x1 - x0) * f), round((y1 - y0) * f)
        ox, oy = (size - w) // 2, size - cfg["foot_margin"] * k - h
        if w > size:
            sys.exit(f"{rig['image']}: 그림이 너무 넓습니다 ({w} > {size}). char_height 를 줄이세요")

        def place(im):
            canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            canvas.paste(im.crop((x0, y0, x1, y1)).resize((w, h), Image.LANCZOS), (ox, oy))
            return canvas

        self.k, self.size, self.thr, self.view = k, size, thr, rig["view"]
        # 길이·이동값은 "64칸 기준 픽셀"로 적는다. 32칸이면 절반, 128칸이면 두 배로 그려져
        # 같은 rig·동작 파일을 어느 크기에서든 그대로 쓸 수 있다.
        self.unit = size / 64
        self.rest = place(img)
        self.layers = [{"name": "body", "slot": "body"}]
        rests = [self.rest]
        for layer in rig.get("layers", []):
            limg = px.remove_background(Image.open(char_dir / layer["image"]), cfg["bg_tolerance"])
            if limg.size != img.size:
                sys.exit(f"{layer['image']}: 크기가 {rig['image']} 와 같아야 합니다 ({limg.size} != {img.size})")
            rests.append(place(limg))
            self.layers.append({"name": layer["name"], "slot": layer.get("slot", layer["name"])})
        self.layer_images = rests[1:]
        self.texture = np.concatenate([np.asarray(r) for r in rests], axis=1)  # 층 그림을 가로로 이어 붙인다
        self.to_canvas = lambda p: (np.asarray(p, float) - [x0, y0]) * f + [ox, oy]
        self.to_source = lambda p: (np.asarray(p, float) - [ox, oy]) / f + [x0, y0]

        bones, joints, depth_of = rig_bones(rig)
        self.joints = {n: self.to_canvas(p) for n, p in joints.items()}
        attachments = rig.get("attachments", [])
        placed = []
        for a in attachments:
            # 붙일 뼈 위 'at' 비율 자리가 손잡이, 그 뼈 방향에서 'angle'도 돌린 쪽이 끝.
            # 같은 slot 의 물건들은 뼈 하나(이름 = slot)를 함께 쓴다.
            slot = a.get("slot", a["name"])
            j0, j1 = next((b[2], b[3]) for b in bones if b[0] == a["bone"])
            s, e = self.joints[j0], self.joints[j1]
            grip = s + (e - s) * a.get("at", 0.5)
            ang = math.atan2(e[1] - s[1], e[0] - s[0]) + math.radians(a.get("angle", 0))
            end = grip + a["length"] * self.unit * np.array([math.cos(ang), math.sin(ang)])
            placed.append((grip, end))
            if slot not in [b[0] for b in bones]:
                self.joints[slot + "_start"], self.joints[slot + "_end"] = grip, end
                bones.append((slot, a["bone"], slot + "_start", slot + "_end"))
                depth_of[slot] = a.get("depth", 5)
        self.bones = bones
        names = [b[0] for b in bones]
        self.base_pose = rig.get("base_pose", {})

        # 메시는 모든 층을 합친 모양에 씌워서, 어느 층이든 같은 그물로 휜다.
        step = cfg["mesh_step"]
        alpha = self.texture[..., 3].reshape(size, len(rests), size) >= thr
        self.verts, self.tris = build_mesh(alpha.any(1), step)
        far = set(FAR_OF.values()) if self.view == "side" else set()
        skip = far | {a.get("slot", a["name"]) for a in attachments}
        self.weights = skin_weights(self.verts, self.tris, self.joints, step, skip, bones)

        depth = np.array([depth_of[n] for n in names], float)
        dominant = self.weights[self.tris].sum(1).argmax(1)
        self.tri_depth = depth[dominant]
        self.tri_tint = np.zeros(len(self.tris), bool)
        self.tri_src = self.tris  # 텍스처 좌표로 쓸 쉼 자세 점 번호
        self.all_verts, self.all_weights = self.verts, self.weights

        if self.view == "side":
            # 보이는 팔다리 삼각형을 복사해 뒤쪽 팔다리 뼈에 붙인다.
            n = len(self.verts)
            limb = np.isin(dominant, [names.index(b) for b in LIMBS_L])
            w2 = self.weights.copy()
            for near, far_name in FAR_OF.items():
                a, b = names.index(near), names.index(far_name)
                w2[:, b], w2[:, a] = w2[:, a], 0
            copies = self.tris[limb] + n
            self.all_verts = np.vstack([self.verts, self.verts])
            self.all_weights = np.vstack([self.weights, w2])
            self.tris = np.vstack([self.tris, copies])
            self.tri_src = np.vstack([self.tri_src, self.tri_src[limb]])
            self.tri_depth = np.concatenate([self.tri_depth, depth[self.all_weights[copies].sum(1).argmax(1)]])
            self.tri_tint = np.concatenate([self.tri_tint, np.ones(limb.sum(), bool)])

        # 층마다 같은 삼각형을 쓰되 텍스처 자리만 옮긴다. 덮개 층은 그림이 있는 칸의 삼각형만 남긴다.
        n = len(self.verts)
        self.src_pts = np.vstack([self.verts + [i * size, 0] for i in range(len(rests))])
        cell = (self.verts[self.tri_src].min(1) // step).astype(int)
        rows, cols = math.ceil(size / step), math.ceil(size / step)
        tris, src, dep, tint, lay = [self.tris], [self.tri_src], [self.tri_depth], [self.tri_tint], [np.zeros(len(self.tris), int)]
        for i in range(1, len(rests)):
            pad = np.zeros((rows * step, cols * step), bool)
            pad[:size, :size] = alpha[:, i]
            occupied = pad.reshape(rows, step, cols, step).any(axis=(1, 3))
            keep = occupied[cell[:, 1], cell[:, 0]]
            tris.append(self.tris[keep])
            src.append(self.tri_src[keep] + i * n)
            dep.append(self.tri_depth[keep])
            tint.append(self.tri_tint[keep])
            lay.append(np.full(keep.sum(), i))
        self.tris, self.tri_src = np.vstack(tris), np.vstack(src)
        self.tri_depth, self.tri_tint, self.tri_layer = np.concatenate(dep), np.concatenate(tint), np.concatenate(lay)

        self.attach_quads, self.attach_images = [], []
        for a, (grip, end) in zip(attachments, placed):
            self.layers.append({"name": a["name"], "slot": a.get("slot", a["name"])})
            self._attach(char_dir / a["image"], a, grip, end, names.index(a.get("slot", a["name"])),
                         a.get("depth", 5), len(self.layers) - 1)

        slots = []
        for layer in self.layers:
            if layer["slot"] not in slots:
                slots.append(layer["slot"])
        self.slots = slots
        first = {}
        for i, layer in enumerate(self.layers):
            first.setdefault(layer["slot"], i)
        self.default_layers = sorted(first.values())

    def _attach(self, path, a, start, end, bone_i, depth, layer_i):
        """무기처럼 휘지 않는 그림 한 장을 뼈에 붙인다. 네모 하나(삼각형 두 개)로 그린다.
        그림은 텍스처 오른쪽에 이어 붙이고, 그림 속 grip→tip 방향을 start→end 방향에 맞춘다."""
        img = Image.open(path).convert("RGBA")
        grip, tip = np.array(a["grip"], float), np.array(a["tip"], float)
        scale = np.linalg.norm(end - start) / np.linalg.norm(tip - grip)
        tw, th = max(1, round(img.width * scale)), max(1, round(img.height * scale))
        img = img.resize((tw, th), Image.LANCZOS)

        tex = self.texture
        ax = tex.shape[1]
        atlas = np.zeros((max(tex.shape[0], th), ax + tw, 4), np.uint8)
        atlas[:tex.shape[0], :ax] = tex
        atlas[:th, ax:] = np.asarray(img)
        self.texture = atlas

        corners = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], float)
        ang = math.atan2(*(end - start)[::-1]) - math.atan2(*(tip - grip)[::-1])
        c, s = math.cos(ang), math.sin(ang)
        rest = start + (corners - grip * scale) @ np.array([[c, s], [-s, c]])

        w = np.zeros((4, self.all_weights.shape[1]))
        w[:, bone_i] = 1
        quad = np.array([[0, 1, 2], [0, 2, 3]])
        self.tris = np.vstack([self.tris, quad + len(self.all_verts)])
        self.tri_src = np.vstack([self.tri_src, quad + len(self.src_pts)])
        self.all_verts = np.vstack([self.all_verts, rest])
        self.all_weights = np.vstack([self.all_weights, w])
        self.src_pts = np.vstack([self.src_pts, corners + [ax, 0]])
        self.tri_depth = np.concatenate([self.tri_depth, [depth, depth]])
        self.tri_tint = np.concatenate([self.tri_tint, [False, False]])
        self.tri_layer = np.concatenate([self.tri_layer, [layer_i, layer_i]])
        self.attach_quads.append(rest)
        self.attach_images.append(img)

    def _candidates(self, pose):
        mats = bone_matrices(self.bones, self.joints, pose, self.unit)
        stack = np.stack([mats[b[0]] for b in self.bones])  # (B,3,3)
        homo = np.c_[self.all_verts, np.ones(len(self.all_verts))]
        moved = np.einsum("bij,nj->nbi", stack, homo)[..., :2]  # (N,B,2)
        dst = (moved * self.all_weights[..., None]).sum(1)
        pix, tri, rgba = rasterize(dst, self.src_pts, self.tris, self.tri_src, self.tri_tint,
                                   self.texture, self.size, self.thr)
        # 앞뒤: depth 가 먼저, 같으면 뒤에 오는 층(덮개)이 앞, 그다음 삼각형 순서
        prio = self.tri_depth[tri] * 1e8 + self.tri_layer[tri] * 1e6 + tri
        return pix, tri, prio, rgba

    def render(self, pose, parts=False):
        """slot 마다 첫 번째 층을 합친 그림. parts=True 면 (층 번호, depth) 마다 따로 그린 그림도 함께 돌려준다.
        따로 그린 그림을 게임에서 depth 순서대로 쌓으면 합친 그림과 같아진다."""
        pix, tri, prio, rgba = self._candidates(pose)
        layer = self.tri_layer[tri]
        m = np.isin(layer, self.default_layers)
        combined = compose(pix[m], prio[m], rgba[m], self.size)
        if not parts:
            return combined
        depth = self.tri_depth[tri]
        pieces = {}
        for key in sorted(set(zip(layer.tolist(), depth.tolist()))):
            m = (layer == key[0]) & (depth == key[1])
            pieces[key] = compose(pix[m], prio[m], rgba[m], self.size)
        return combined, pieces

    def depth_levels(self):
        return sorted(set(self.tri_depth.tolist()))


def build_mesh(solid, step):
    """그림이 있는 칸만 골라 칸마다 삼각형 두 개를 만든다."""
    h, w = solid.shape
    rows, cols = math.ceil(h / step), math.ceil(w / step)
    pad = np.zeros((rows * step, cols * step), bool)
    pad[:h, :w] = solid
    cells = pad.reshape(rows, step, cols, step).any(axis=(1, 3))
    cy, cx = np.nonzero(cells)
    node_id = -np.ones((rows + 1, cols + 1), int)
    corners = [(0, 0), (0, 1), (1, 1), (1, 0)]
    for dy, dx in corners:
        node_id[cy + dy, cx + dx] = 0
    ny, nx = np.nonzero(node_id == 0)
    node_id[ny, nx] = np.arange(len(ny))
    verts = np.c_[nx, ny].astype(float) * step
    a, b = node_id[cy, cx], node_id[cy, cx + 1]
    c, d = node_id[cy + 1, cx + 1], node_id[cy + 1, cx]
    tris = np.vstack([np.c_[a, b, c], np.c_[a, c, d]])
    return verts, tris


def segment_distance(points, a, b):
    ab = b - a
    t = np.clip(((points - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def skin_weights(verts, tris, joints, step, skip, bones):
    """메시 안쪽으로만 잰 거리(뼈에서 그물을 따라 걸어간 거리)로 점마다 뼈 무게를 정한다.
    그래서 붙어 보여도 틈으로 떨어진 두 다리는 서로를 따라가지 않는다. 가까운 뼈 두 개만 섞는다."""
    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    lengths = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
    nbrs = [[] for _ in verts]
    for (i, j), d in zip(edges.tolist(), lengths.tolist()):
        nbrs[i].append((j, d))
        nbrs[j].append((i, d))

    dist = np.full((len(verts), len(bones)), np.inf)
    for bi, (name, _, j0, j1) in enumerate(bones):
        if name in skip:
            continue
        d0 = segment_distance(verts, joints[j0], joints[j1])
        seeds = np.nonzero(d0 <= step)[0]
        if len(seeds) == 0:
            seeds = [int(d0.argmin())]
        best = np.full(len(verts), np.inf)
        heap = [(d0[s], int(s)) for s in seeds]
        for d, s in heap:
            best[s] = d
        heapq.heapify(heap)
        while heap:
            d, i = heapq.heappop(heap)
            if d > best[i]:
                continue
            for j, w in nbrs[i]:
                if d + w < best[j]:
                    best[j] = d + w
                    heapq.heappush(heap, (d + w, j))
        dist[:, bi] = best

    w = np.where(np.isfinite(dist), 1.0 / (dist + 1.0) ** 4, 0.0)
    top2 = np.argsort(-w, axis=1)[:, :2]
    keep = np.zeros_like(w, bool)
    np.put_along_axis(keep, top2, True, axis=1)
    w = np.where(keep, w, 0.0)
    total = w.sum(1, keepdims=True)
    lonely = total[:, 0] == 0  # 어느 뼈와도 이어지지 않은 조각은 가장 가까운 뼈 하나를 따른다
    if lonely.any():
        near = np.stack([segment_distance(verts[lonely], joints[j0], joints[j1]) for _, _, j0, j1 in bones], 1)
        for s in skip:
            near[:, [b[0] for b in bones].index(s)] = np.inf
        w[np.nonzero(lonely)[0], near.argmin(1)] = 1.0
        total = w.sum(1, keepdims=True)
    return w / total


# ---------------------------------------------------------------- 뼈 움직이기

def _rot(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _move(x, y):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], float)


def bone_matrices(bones, joints, pose, unit):
    """뼈마다 '쉼 자세 좌표 → 움직인 좌표' 행렬. 회전은 화면에서 시계 방향이 양수(도).
    len 은 뼈 방향으로 늘이고 줄이는 비율, x/y 는 64칸 기준 픽셀 단위 이동 (unit = 그리기 픽셀 / 64칸 기준 픽셀)."""
    root = pose.get("root", {})
    root_m = _move(root.get("x", 0) * unit, root.get("y", 0) * unit)
    mats = {}
    for name, parent, j0, j1 in bones:
        p = pose.get(name, {})
        s, e = joints[j0], joints[j1]
        a = math.degrees(math.atan2(e[1] - s[1], e[0] - s[0]))
        stretch = _rot(a) @ np.diag([p.get("len", 1.0), 1, 1]) @ _rot(-a)
        local = (_move(p.get("x", 0) * unit, p.get("y", 0) * unit) @ _move(*s) @ _rot(p.get("rot", 0))
                 @ stretch @ _move(-s[0], -s[1]))
        mats[name] = (mats[parent] if parent else root_m) @ local
    return mats


def rasterize(dst, src, tris, tri_src, tint, texture, size, thr):
    """움직인 삼각형마다 쉼 자세 그림에서 색을 가져온다.
    겹침은 정하지 않고 (픽셀 번호, 삼각형 번호, 색) 후보를 모두 돌려준다 → compose 에서 앞뒤를 고른다."""
    pix_all, tri_all, rgba_all = [], [], []
    # 묶음마다 가장 큰 삼각형 크기만큼 칸을 잡으므로, 크기가 비슷한 것끼리 묶는다 (칼처럼 큰 삼각형 때문에 느려지지 않게)
    span = np.ptp(dst[tris], axis=1).max(1)
    order = np.argsort(span)
    start = 0
    while start < len(order):
        side = int(math.ceil(span[order[min(start + 2047, len(order) - 1)]])) + 2
        count = int(max(1, min(2048, 4_000_000 // (side * side))))
        idx = order[start:start + count]
        start += count
        P = dst[tris[idx]]          # (m,3,2) 움직인 꼭짓점
        S = src[tri_src[idx]]       # (m,3,2) 쉼 자세 꼭짓점 = 텍스처 좌표
        lo = np.floor(P.min(1)).astype(int)
        hi = np.ceil(P.max(1)).astype(int)
        bw, bh = (hi - lo).max(0) + 1
        oy, ox = np.mgrid[0:bh, 0:bw]
        X = lo[:, :1] + ox.ravel()[None]   # (m,K)
        Y = lo[:, 1:] + oy.ravel()[None]
        cx, cy = X + 0.5, Y + 0.5
        ax, ay = P[:, 0, 0:1], P[:, 0, 1:2]
        v0 = P[:, 1] - P[:, 0]
        v1 = P[:, 2] - P[:, 0]
        den = v0[:, 0:1] * v1[:, 1:2] - v1[:, 0:1] * v0[:, 1:2]
        den = np.where(np.abs(den) < 1e-9, np.nan, den)
        l1 = ((cx - ax) * v1[:, 1:2] - v1[:, 0:1] * (cy - ay)) / den
        l2 = (v0[:, 0:1] * (cy - ay) - (cx - ax) * v0[:, 1:2]) / den
        l0 = 1 - l1 - l2
        inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6) & (X >= 0) & (X < size) & (Y >= 0) & (Y < size)
        sx = l0 * S[:, 0, 0:1] + l1 * S[:, 1, 0:1] + l2 * S[:, 2, 0:1]
        sy = l0 * S[:, 0, 1:2] + l1 * S[:, 1, 1:2] + l2 * S[:, 2, 1:2]
        tx = np.clip(np.nan_to_num(np.floor(sx)), 0, texture.shape[1] - 1).astype(int)
        ty = np.clip(np.nan_to_num(np.floor(sy)), 0, texture.shape[0] - 1).astype(int)
        color = texture[ty, tx]
        keep = inside & (color[..., 3] >= thr)
        rows, cols = np.nonzero(keep)
        pix_all.append(Y[rows, cols] * size + X[rows, cols])
        tri_all.append(idx[rows])
        c = color[rows, cols].astype(np.float32)
        c[tint[idx][rows], :3] *= 0.65  # 뒤쪽 복사본은 어둡게
        rgba_all.append(c)
    return np.concatenate(pix_all), np.concatenate(tri_all), np.concatenate(rgba_all)


def compose(pix, prio, rgba, size):
    """같은 픽셀에 여러 후보가 있으면 prio 가 큰 것을 남긴다."""
    o = np.lexsort((prio, pix))
    pix, rgba = pix[o], rgba[o]
    last = np.r_[pix[1:] != pix[:-1], True] if len(pix) else np.zeros(0, bool)
    out = np.zeros((size * size, 4), np.uint8)
    out[pix[last]] = rgba[last].round().astype(np.uint8)
    return Image.fromarray(out.reshape(size, size, 4))


# ---------------------------------------------------------------- 동작

def load_anim(path, anims_dir):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "mirror_of" in data:
        # 좌우를 뒤집은 동작: 회전과 x 이동의 부호를 바꾼다 (예: 앞 그림 동작을 뒤 그림에 쓰기)
        base = load_anim(anims_dir / f"{data['mirror_of']}.json", anims_dir)
        keys = {bone: {ch: [[t, -v if ch in ("rot", "x") else v] for t, v in vals] for ch, vals in chans.items()}
                for bone, chans in base["keys"].items()}
        data = {**base, **{k: v for k, v in data.items() if k != "mirror_of"}, "keys": keys}
    data.setdefault("frames", 8)
    data.setdefault("fps", 10)
    data.setdefault("loop", True)
    return data


def sample(keys, t):
    """열쇠 프레임 [[시간(0~1), 값], ...] 사이를 부드럽게 잇는다."""
    keys = sorted(keys)
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            u = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
            u = u * u * (3 - 2 * u)
            return v0 + (v1 - v0) * u
    return keys[-1][1]


def pose_at(anim, base_pose, t):
    pose = {bone: dict(chans) for bone, chans in base_pose.items()}
    for bone, chans in anim["keys"].items():
        p = pose.setdefault(bone, {})
        for ch, keys in chans.items():
            v = sample(keys, t)
            p[ch] = p.get(ch, 1.0) * v if ch == "len" else p.get(ch, 0.0) + v
    return pose


# ---------------------------------------------------------------- 확인 그림

def draw_rig_check(path, rig, cfg):
    """원본 위에 관절, 뼈, 메시 점(가장 많이 따르는 뼈 색)과 100칸 눈금을 그린다."""
    src = Image.open(path).convert("RGBA")
    model = Model(path.parent, rig, cfg)
    canvas = Image.new("RGBA", src.size)
    d = ImageDraw.Draw(canvas)
    dominant = model.weights.argmax(1)
    color = [BONE_COLORS[i % len(BONE_COLORS)] for i in range(len(model.bones))]
    for (x, y), b in zip(model.to_source(model.verts), dominant):
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(*color[b], 150))
    j = {n: model.to_source(p).tolist() for n, p in model.joints.items()}
    for quad in model.attach_quads:
        d.polygon([tuple(p) for p in model.to_source(quad)], outline=(255, 255, 255, 255), width=3)
    for i, (name, _, j0, j1) in enumerate(model.bones):
        d.line([tuple(j[j0]), tuple(j[j1])], fill=(*color[i], 255), width=5)
    for name, (x, y) in j.items():
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))
        d.text((x + 7, y - 6), name, fill=(0, 0, 0, 255))
    w, h = src.size
    for x in range(0, w, 100):
        d.line([(x, 0), (x, h)], fill=(0, 0, 0, 60))
        d.text((x + 2, 2), str(x), fill=(0, 0, 0, 255))
    for y in range(0, h, 100):
        d.line([(0, y), (w, y)], fill=(0, 0, 0, 60))
        d.text((2, y + 2), str(y), fill=(0, 0, 0, 255))
    src.alpha_composite(canvas)
    return src


def clipped_sides(img, k):
    """칸 가장자리 1픽셀(외곽선 자리) 안에 그림이 닿으면 잘린 것으로 본다."""
    a = np.asarray(img)[..., 3] > 0
    sides = {"위": a[:k], "아래": a[-k:], "왼쪽": a[:, :k], "오른쪽": a[:, -k:]}
    return [name for name, part in sides.items() if part.any()]


def on_gray(img, color=(200, 200, 200, 255)):
    bg = Image.new("RGBA", img.size, color)
    bg.alpha_composite(img)
    return bg


def draw_anim_check(hi_frames, pix_frames, stacked=None):
    """위 줄: 휘게 한 원본(크게), 가운데 줄: 픽셀 결과(4배), (있으면) 아래 줄: 층 시트를 순서대로 쌓은 결과."""
    cell = 256
    rows = [pix_frames] + ([stacked] if stacked else [])
    sheet = Image.new("RGBA", (cell * len(hi_frames), cell * (1 + len(rows))), (255, 255, 255, 255))
    for i, hi in enumerate(hi_frames):
        sheet.paste(on_gray(hi.resize((cell, cell), Image.LANCZOS)), (i * cell, 0))
        for r, frames in enumerate(rows, start=1):
            sheet.paste(on_gray(frames[i].resize((cell, cell), Image.NEAREST)), (i * cell, r * cell))
    return sheet


def write_layer_sheets(out, name, model, part_frames, n, palette, cfg):
    """(층, depth) 마다 시트를 하나씩 쓴다. 파일 이름 앞 두 자리는 depth 순위(작을수록 뒤).
    겹칠 경계에 검은 줄이 생기지 않도록 층 시트에는 외곽선을 두르지 않는다."""
    canvas = cfg["canvas"]
    layer_dir = out / "layers" / name
    if layer_dir.exists():
        shutil.rmtree(layer_dir)
    layer_dir.mkdir(parents=True)
    rank = {d: i for i, d in enumerate(model.depth_levels())}
    thr, k = cfg["alpha_threshold"], model.k

    def solid(img):
        return np.asarray(img)[..., 3] >= thr

    def per_cell(mask):  # 64칸의 칸마다 그림이 차지한 비율
        return mask.reshape(canvas, k, canvas, k).mean(axis=(1, 3))

    # 한 칸이 여러 층에 나뉘어 걸치면 어느 층도 절반이 안 돼 구멍이 난다.
    # 그래서 칠할지는 (합친 그림 + 이 층) 모양으로 정하고, 그 칸은 가장 많이 차지한 층이 맡는다.
    default_keys = [key for key in part_frames if key[0] in model.default_layers]
    masks = {key: {i: solid(img) for i, img in frames.items()} for key, frames in part_frames.items()}
    cells = {key: {i: per_cell(m) for i, m in frames.items()} for key, frames in masks.items()}
    default_union = {}
    for i in range(n):
        u = np.zeros((model.size, model.size), bool)
        for o in default_keys:
            if i in masks[o]:
                u |= masks[o][i]
        default_union[i] = u

    def keep_mask(key, i):
        own = cells[key][i]
        others = [cells[o][i] for o in default_keys if o != key and i in cells[o]]
        best = own >= np.max(others, axis=0) if others else True
        filled = per_cell(default_union[i] | masks[key][i]) >= 0.5
        return (own >= 0.5) | (filled & best & (own > 0))

    parts = []
    stacked = [Image.new("RGBA", (canvas, canvas)) for _ in range(n)]
    for (layer_i, depth) in sorted(part_frames, key=lambda kd: (rank[kd[1]], model.slots.index(model.layers[kd[0]]["slot"]), kd[0])):
        frames = part_frames[(layer_i, depth)]
        small = [px.downscale(frames[i], (canvas, canvas), palette, thr, keep_mask((layer_i, depth), i)) if i in frames
                 else Image.new("RGBA", (canvas, canvas)) for i in range(n)]
        if not any(np.asarray(s)[..., 3].any() for s in small):
            continue
        layer = model.layers[layer_i]
        file = f"{rank[depth]:02d}_{layer['name']}.png"
        sheet = Image.new("RGBA", (canvas * n, canvas))
        for i, s in enumerate(small):
            sheet.paste(s, (i * canvas, 0))
        sheet.save(layer_dir / file)
        parts.append({"file": f"layers/{name}/{file}", "layer": layer["name"], "slot": layer["slot"], "depth": depth,
                      "sorting_order": rank[depth] * 100 + model.slots.index(layer["slot"])})
        if layer_i in model.default_layers:
            for i, s in enumerate(small):
                stacked[i].alpha_composite(s)
    return parts, stacked


# ---------------------------------------------------------------- render

def cmd_render(args):
    cfg = load_config(args.config)
    anims_dir = args.anims
    rigs = {}
    for view in VIEWS:
        rig_path = args.character / f"rig_{view}.json"
        if rig_path.exists():
            rigs[view] = json.loads(rig_path.read_text(encoding="utf-8"))
    if not rigs:
        sys.exit(f"{args.character} 에 rig_*.json 이 없습니다. 먼저 rig 명령을 돌리세요")

    print("메시와 무게 준비 중...")
    models = {view: Model(args.character, rig, cfg) for view, rig in rigs.items()}

    out = args.output
    (out / "check").mkdir(parents=True, exist_ok=True)
    if cfg["palette"]:
        palette = px.read_hex(cfg["palette"])
    else:
        sources = [img for m in models.values() for img in [m.rest] + m.layer_images + m.attach_images]
        palette = px.build_palette(sources, cfg["colors"], cfg["alpha_threshold"])
        saved = list(palette) + ([px.hex_to_rgb(cfg["outline_color"])] if cfg["outline"] else [])
        px.write_hex(out / "palette.hex", saved)

    meta_path = out / "animations.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    layers_path = out / "layers.json"
    layers_meta = json.loads(layers_path.read_text(encoding="utf-8")) if layers_path.exists() else {}
    canvas = cfg["canvas"]
    for view, model in models.items():
        for anim_path in sorted((anims_dir / view).glob("*.json")):
            key = f"{view}/{anim_path.stem}"
            if args.only and key not in args.only:
                continue
            anim = load_anim(anim_path, anims_dir)
            n = anim["frames"]
            hi_frames, pix_frames = [], []
            part_frames = {}  # (층, depth) → {프레임 번호: 그림}
            for i in range(n):
                t = i / n if anim["loop"] else i / max(1, n - 1)
                pose = pose_at(anim, model.base_pose, t)
                if args.layers:
                    hi, pieces = model.render(pose, parts=True)
                    for key, img in pieces.items():
                        part_frames.setdefault(key, {})[i] = img
                else:
                    hi = model.render(pose)
                cut = clipped_sides(hi, model.k)
                if cut:
                    print(f"  주의 {view}_{anim_path.stem} {i}번 프레임: 그림이 칸 밖으로 잘림 ({', '.join(cut)})")
                small = px.downscale(hi, (canvas, canvas), palette, cfg["alpha_threshold"])
                if cfg["outline"]:
                    small = px.add_outline(small, px.hex_to_rgb(cfg["outline_color"]))
                hi_frames.append(hi)
                pix_frames.append(small)

            name = f"{view}_{anim_path.stem}"
            sheet = Image.new("RGBA", (canvas * n, canvas))
            for i, f in enumerate(pix_frames):
                sheet.paste(f, (i * canvas, 0))
            sheet.save(out / f"{name}.png")
            gif = [on_gray(f).resize((canvas * 4, canvas * 4), Image.NEAREST).convert("P") for f in pix_frames]
            gif[0].save(out / "check" / f"{name}.gif", save_all=True, append_images=gif[1:],
                        duration=round(1000 / anim["fps"]), loop=0)
            meta[name] = {"frames": n, "fps": anim["fps"], "loop": anim["loop"],
                          "frame_size": canvas, "pivot": [0.5, cfg["foot_margin"] / canvas]}
            stacked = None
            if args.layers:
                parts, stacked = write_layer_sheets(out, name, model, part_frames, n, palette, cfg)
                layers_meta[name] = {**meta[name], "parts": parts}
            draw_anim_check(hi_frames, pix_frames, stacked).save(out / "check" / f"{name}.png")
            print(f"  {name}.png ({n}프레임)" + (f", 층 시트 {len(parts)}장" if args.layers else ""))
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.layers:
        layers_path.write_text(json.dumps(layers_meta, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="T자 일러스트로 뼈대 동작을 만들어 픽셀 스프라이트로 찍습니다.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_rig = sub.add_parser("rig", help="관절을 찾아 rig_<방향>.json 을 만든다")
    p_rig.add_argument("character", type=Path, help="front.png / back.png / side.png 가 든 캐릭터 폴더")
    p_rig.add_argument("-c", "--config", type=Path, default=HERE / "config.json")
    p_rig.set_defaults(func=cmd_rig)

    p_check = sub.add_parser("check", help="고친 rig 파일로 확인 그림만 다시 그린다")
    p_check.add_argument("character", type=Path)
    p_check.add_argument("-c", "--config", type=Path, default=HERE / "config.json")
    p_check.set_defaults(func=cmd_check)

    p_render = sub.add_parser("render", help="동작을 만들어 스프라이트 시트로 내보낸다")
    p_render.add_argument("character", type=Path)
    p_render.add_argument("-o", "--output", type=Path, required=True)
    p_render.add_argument("-c", "--config", type=Path, default=HERE / "config.json")
    p_render.add_argument("-a", "--anims", type=Path, default=HERE / "anims")
    p_render.add_argument("--only", nargs="*", help="이 동작만 만든다 (예: front/walk side/attack)")
    p_render.add_argument("--layers", action="store_true",
                          help="층(몸·덮개·물건)과 앞뒤 순서별로 나눈 시트와 layers.json 도 만든다")
    p_render.set_defaults(func=cmd_render)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
