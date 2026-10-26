"""T자 일러스트에 뼈대를 심어 동작을 만들고, 프레임마다 64칸 픽셀 그림으로 찍는 도구.

사용법:
    python animate.py rig 캐릭터폴더 [-c config.json]
        front.png / back.png / left.png / right.png 에서 관절을 찾아 rig_<방향>.json 과 확인 그림 check/rig_<방향>.png 를 만든다.
    python animate.py check 캐릭터폴더
        rig_<방향>.json 을 손으로 고친 뒤 확인 그림만 다시 그린다.
    python animate.py parts 캐릭터폴더 [--force]
        뼈대를 따라 부위별 조각 그림을 만든다 (parts.py).
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
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # 내장 Python은 스크립트 폴더를 자동으로 찾지 않는다
import pixelize as px  # noqa: E402
VIEWS = ("front", "back", "left", "right", "front_left", "back_left", "front_right", "back_right")
# 옆 그림: 왼쪽을 보면 캐릭터의 왼쪽(_l)이, 오른쪽을 보면 오른쪽(_r)이 보는 사람 쪽(가까운 쪽)이다.
SIDE_VIEWS = {"left": "l", "right": "r"}
# 대각선 그림(45도): 두 팔다리가 다 보이지만 한쪽이 보는 사람 쪽이다. 왼쪽 대각선은 캐릭터의 왼쪽이 가깝다.
DIAG_VIEWS = {"front_left": "l", "back_left": "l", "front_right": "r", "back_right": "r"}
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
LIMB_PARTS = ["upper_arm", "forearm", "hand", "thigh", "shin", "foot"]


def far_of(view):
    """옆 그림에서 보이는 팔다리 → 뒤쪽 복사본 뼈. 옆 그림이 아니면 빈 dict.
    옆 그림은 두 다리·두 팔이 겹쳐 있어서, 보이는 쪽 조각을 어둡게 복사해 뒤쪽 팔다리로 쓴다."""
    near = SIDE_VIEWS.get(view)
    if not near:
        return {}
    far = "r" if near == "l" else "l"
    return {f"{p}_{near}": f"{p}_{far}" for p in LIMB_PARTS}


# 그리는 순서 (큰 수가 앞). rig 파일에서 고칠 수 있다.
DEFAULT_DEPTH = {
    "front": {"thigh": 1, "shin": 1, "foot": 1, "torso": 2, "head": 3, "upper_arm": 4, "forearm": 4, "hand": 4},
    "back": {"thigh": 1, "shin": 1, "foot": 1, "torso": 2, "head": 3, "upper_arm": 4, "forearm": 4, "hand": 4},
    "left": {"torso": 3, "head": 4},
    "right": {"torso": 3, "head": 4},
    **{v: {"thigh": 1, "shin": 1, "foot": 1, "torso": 2, "head": 3, "upper_arm": 4, "forearm": 4, "hand": 4}
       for v in ("front_left", "back_left", "front_right", "back_right")},
}
DIAG_FAR_DEPTH = {"upper_arm": 1.5, "forearm": 1.5, "hand": 1.5, "thigh": 0.8, "shin": 0.8, "foot": 0.8}  # 대각선의 먼 쪽 팔다리
SIDE_LIMB_DEPTH = {"upper_arm": (0, 5), "forearm": (0, 5), "hand": (0, 5),
                   "thigh": (1, 2), "shin": (1, 2), "foot": (1, 2)}  # (뒤쪽, 앞쪽)

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
    """pixelize 설정 + 동작용 설정. frame / frame_foot_margin 이 있으면 동작 시트의 칸 크기와 발 여백을 그 값으로 바꾼다
    (캐릭터 키 char_height 는 그대로라, 칸만 커지고 위·옆에 여백이 생긴다. 정지 그림용 pixelize.py 는 canvas 를 그대로 쓴다)."""
    cfg = px.load_config(path)
    for key, value in ANIM_DEFAULTS.items():
        cfg.setdefault(key, value)
    if cfg.get("frame"):
        cfg["canvas"] = cfg["frame"]
        cfg["foot_margin"] = cfg.get("frame_foot_margin", cfg["foot_margin"])
    return cfg


def length_unit(cfg):
    """동작·붙인 물건의 길이 단위 (그리기 픽셀). "키 48칸 캐릭터의 1칸"이다.
    칸(frame)을 키워도 캐릭터 크기는 그대로이므로 길이도 그대로이고, 캐릭터 키를 바꾸면(32칸·128칸 판) 함께 늘고 준다."""
    return cfg["char_height"] * cfg["render_scale"] / 48


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

    if view in SIDE_VIEWS:
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
    if view in SIDE_VIEWS:
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
        if view in SIDE_VIEWS and kind in SIDE_LIMB_DEPTH:
            depth[name] = SIDE_LIMB_DEPTH[kind][1 if name.endswith("_" + SIDE_VIEWS[view]) else 0]
        elif view in DIAG_VIEWS and kind in DIAG_FAR_DEPTH and not name.endswith("_" + DIAG_VIEWS[view]):
            depth[name] = DIAG_FAR_DEPTH[kind]
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
        if args.views and view not in args.views:
            continue
        img = load_clean(path, cfg)
        joints = detect_joints(img, view, cfg["alpha_threshold"])
        rig = {"image": path.name, "view": view, "joints": joints,
               "base_pose": default_base_pose(joints, view), "depth": default_depth(view)}
        rig_path = args.character / f"rig_{view}.json"
        write_rig(rig_path, rig)
        print(f"  {rig_path}  (확인 그림: {save_rig_check(args.character, rig, cfg)})")


def write_rig(path, rig):
    text = json.dumps(rig, indent=2, ensure_ascii=False)
    text = re.sub(r"\[\s+(-?[\d.]+),\s+(-?[\d.]+)\s+\]", r"[\1, \2]", text)  # 좌표는 한 줄로
    path.write_text(text, encoding="utf-8")


def cmd_mirror(args):
    """한 방향 그림을 좌우로 뒤집어 다른 방향을 만든다 (예: 왼쪽을 보는 옆 그림 → 오른쪽을 보는 옆 그림).
    그림·덮개 그림을 뒤집고, rig 의 관절·추가 뼈 좌표를 뒤집는다. 앞뒤 순서는 새 방향의 보이는 쪽에 맞춰 다시 정한다.
    붙인 물건(칼)은 같은 손에 그대로 두고, 그 손 바로 앞(+0.5)에 그린다 — 오른손잡이는 어느 방향에서도 오른손잡이다.
    나중에 그 방향 그림을 따로 그려 넣으면 그 그림으로 rig 부터 다시 만들면 된다.
    옆 그림이 아니면(대각선 등) 뒤집은 그림에서는 왼쪽·오른쪽 팔다리가 바뀌므로 관절 이름의 _l/_r 도 바꾼다."""
    char, src, dst = args.character, args.source, args.target
    rig = json.loads((char / f"rig_{src}.json").read_text(encoding="utf-8"))
    img = Image.open(char / rig["image"])
    w = img.width
    img.transpose(Image.FLIP_LEFT_RIGHT).save(char / f"{dst}.png")
    flip = lambda p: [round(w - 1 - p[0], 1), p[1]]
    swap = (lambda n: n) if src in SIDE_VIEWS else (
        lambda n: n[:-2] + {"_l": "_r", "_r": "_l"}[n[-2:]] if n.endswith(("_l", "_r")) else n)

    out = {k: v for k, v in rig.items() if k not in ("joints", "extra_bones", "layers", "attachments", "depth", "base_pose")}
    out.update(image=f"{dst}.png", view=dst, joints={swap(n): flip(p) for n, p in rig["joints"].items()})
    out["base_pose"] = {swap(b): ({**ch, "rot": -ch["rot"]} if "rot" in ch else ch) for b, ch in rig.get("base_pose", {}).items()}
    depth = default_depth(dst)
    out["extra_bones"] = [{**e, "name": swap(e["name"]), "start": flip(e["start"]), "end": flip(e["end"])}
                          for e in rig.get("extra_bones", [])]
    for e in out["extra_bones"]:
        if "claim" in e:  # 색으로 가져오는 상자도 좌우로 뒤집는다
            x0, y0, x1, y1 = e["claim"]["box"]
            e["claim"] = {**e["claim"], "box": [w - x1, y0, w - x0, y1]}
    out["depth"] = depth
    layers = []
    for layer in rig.get("layers", []):
        name = layer["image"].replace(f"_{src}", f"_{dst}")
        Image.open(char / layer["image"]).transpose(Image.FLIP_LEFT_RIGHT).save(char / name)
        layers.append({**layer, "image": name})
    if layers:
        out["layers"] = layers
    if rig.get("attachments"):
        out["attachments"] = [{**a, "depth": depth.get(a["bone"], 4) + 0.5} for a in rig["attachments"]]
    write_rig(char / f"rig_{dst}.json", out)
    cfg = load_config(args.config)
    print(f"  {char / f'{dst}.png'}, {char / f'rig_{dst}.json'}  (확인 그림: {save_rig_check(char, out, cfg)})")
    print(f"  다음: animate.bat parts {char} --views {dst}")


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

def canvas_mapping(img, cfg, name):
    """원본 그림 → 그리기 캔버스 자리 잡기. 키가 char_height 가 되도록 줄이고, 발끝을 바닥 기준선에 맞춰 가운데 둔다.
    돌려주는 값: 그림 영역(x0, y0, x1, y1), 배율 f, 캔버스 크기, 줄인 크기(w, h), 놓을 자리(ox, oy)."""
    k = cfg["render_scale"]
    x0, y0, x1, y1 = content_box(img, cfg["alpha_threshold"])
    f = cfg["char_height"] * k / (y1 - y0)
    size = cfg["canvas"] * k
    w, h = round((x1 - x0) * f), round((y1 - y0) * f)
    ox, oy = (size - w) // 2, size - cfg["foot_margin"] * k - h
    if w > size:
        sys.exit(f"{name}: 그림이 너무 넓습니다 ({w} > {size}). char_height 를 줄이세요")
    return x0, y0, x1, y1, f, size, w, h, ox, oy


def place_attachments(attachments, bones, joints, depth_of, unit):
    """붙인 물건마다 (손잡이 자리, 끝 자리)를 캔버스 좌표로 돌려주고, slot 마다 뼈를 하나씩 bones 에 더한다.
    붙일 뼈 위 'at' 비율 자리가 손잡이, 그 뼈 방향에서 'angle'도 돌린 쪽이 끝.
    같은 slot 의 물건들은 뼈 하나(이름 = slot)를 함께 쓴다."""
    placed = []
    for a in attachments:
        slot = a.get("slot", a["name"])
        j0, j1 = next((b[2], b[3]) for b in bones if b[0] == a["bone"])
        s, e = joints[j0], joints[j1]
        grip = s + (e - s) * a.get("at", 0.5)
        ang = math.atan2(e[1] - s[1], e[0] - s[0]) + math.radians(a.get("angle", 0))
        end = grip + a["length"] * unit * np.array([math.cos(ang), math.sin(ang)])
        placed.append((grip, end))
        if slot not in [b[0] for b in bones]:
            joints[slot + "_start"], joints[slot + "_end"] = grip, end
            bones.append((slot, a["bone"], slot + "_start", slot + "_end"))
            depth_of[slot] = a.get("depth", 5)
    return placed


class Model:
    """한 방향 그림의 쉼 자세 그림, 관절, 메시, 무게를 그리기 크기에 맞춰 준비한다.

    층: 0번은 몸(rig 의 image), 그 뒤로 rig 의 layers(갑옷·투구처럼 몸 그림에 맞춰 그린 덮개),
    그 뒤로 attachments(칼처럼 휘지 않는 물건). 같은 slot 의 층들은 서로 바꿔 끼우는 변형이고,
    합친 그림에는 slot 마다 첫 번째 층만 들어간다."""

    def __init__(self, char_dir, rig, cfg):
        k, thr = cfg["render_scale"], cfg["alpha_threshold"]
        img = load_clean(char_dir / rig["image"], cfg)
        x0, y0, x1, y1, f, size, w, h, ox, oy = canvas_mapping(img, cfg, rig["image"])

        def place(im):
            canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            canvas.paste(im.crop((x0, y0, x1, y1)).resize((w, h), Image.LANCZOS), (ox, oy))
            return canvas

        self.k, self.size, self.thr, self.view = k, size, thr, rig["view"]
        self.unit = length_unit(cfg)
        self.rest = place(img)
        self.layers = [{"name": "body", "slot": "body"}]
        rests = [self.rest]
        for layer in rig.get("layers", []):
            limg = px.remove_background(Image.open(char_dir / layer["image"]), cfg["bg_tolerance"])
            if limg.size != img.size:
                sys.exit(f"{layer['image']}: 크기가 {rig['image']} 와 같아야 합니다 ({limg.size} != {img.size})")
            rests.append(place(limg))
            self.layers.append({"name": layer["name"], "slot": layer.get("slot", layer["name"]),
                                "in_sheet": layer.get("in_sheet", True)})
        self.layer_images = rests[1:]
        self.texture = np.concatenate([np.asarray(r) for r in rests], axis=1)  # 층 그림을 가로로 이어 붙인다
        self.to_canvas = lambda p: (np.asarray(p, float) - [x0, y0]) * f + [ox, oy]
        self.to_source = lambda p: (np.asarray(p, float) - [ox, oy]) / f + [x0, y0]

        bones, joints, depth_of = rig_bones(rig)
        self.joints = {n: self.to_canvas(p) for n, p in joints.items()}
        attachments = rig.get("attachments", [])
        placed = place_attachments(attachments, bones, self.joints, depth_of, self.unit)
        self.weapon_bones = list(dict.fromkeys(a.get("slot", a["name"]) for a in attachments))
        self.bones = bones
        names = [b[0] for b in bones]
        self.base_pose = rig.get("base_pose", {})

        # 메시는 모든 층을 합친 모양에 씌워서, 어느 층이든 같은 그물로 휜다.
        step = cfg["mesh_step"]
        alpha = self.texture[..., 3].reshape(size, len(rests), size) >= thr
        self.verts, self.tris = build_mesh(alpha.any(1), step)
        far_map = far_of(self.view)
        skip = set(far_map.values()) | {a.get("slot", a["name"]) for a in attachments}
        self.weights = skin_weights(self.verts, self.tris, self.joints, step, skip, bones)

        depth = np.array([depth_of[n] for n in names], float)
        dominant = self.weights[self.tris].sum(1).argmax(1)
        self.tri_depth = depth[dominant]
        self.tri_tint = np.zeros(len(self.tris), bool)
        self.tri_src = self.tris  # 텍스처 좌표로 쓸 쉼 자세 점 번호
        self.all_verts, self.all_weights = self.verts, self.weights

        if far_map:
            # 보이는 팔다리 삼각형을 복사해 뒤쪽 팔다리 뼈에 붙인다.
            n = len(self.verts)
            limb = np.isin(dominant, [names.index(b) for b in far_map])
            w2 = self.weights.copy()
            for near, far_name in far_map.items():
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
        # 합친 시트: slot 마다 첫 번째 층. 단 "in_sheet": false 인 층(예: 일부 방향에만 그림이 있는 투구)은 뺀다
        self.default_layers = sorted(i for i in first.values() if self.layers[i].get("in_sheet", True))

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


def bone_matrices(bones, joints, pose, unit, with_chain=False):
    """뼈마다 '쉼 자세 좌표 → 움직인 좌표' 행렬. 회전은 화면에서 시계 방향이 양수(도).
    len 은 뼈 방향으로 늘이고 줄이는 비율, x/y 는 캐릭터 키 48 기준 칸 단위 이동 (unit = 그리기 픽셀 / 그 1칸).
    tilt 는 보는 사람 쪽(+)/반대쪽(-)으로 기울인 각도(도) — 뼈 방향으로 cos(tilt) 만큼 짧아져 보인다.
    turn 은 뼈를 제 축으로 돌린 각도(도) — 뼈와 수직 방향으로 cos(turn) 만큼 좁아져 보인다.
    tilt·turn 은 그 뼈의 모양만 줄이고, 자식 뼈는 모양은 그대로 둔 채 붙는 자리만 따라온다
    (상체를 숙이면 머리는 내려오지만 납작해지지 않는다). len 은 예전처럼 자식에게도 이어진다."""
    root = pose.get("root", {})
    # root 는 몸 전체: x/y 로 옮기고, rot 로 몸통 가운데(골반과 목의 중간)를 중심으로 돌린다 (구르기 등)
    center = (joints["pelvis"] + joints["neck"]) / 2
    root_m = (_move(root.get("x", 0) * unit, root.get("y", 0) * unit) @ _move(*center)
              @ _rot(root.get("rot", 0)) @ _move(-center[0], -center[1]))
    mats, chain = {}, {}   # mats: 그 뼈를 그릴 행렬, chain: 자식이 이어받을 행렬 (tilt·turn 모양 줄임이 빠짐)
    for name, parent, j0, j1 in bones:
        p = pose.get(name, {})
        s, e = joints[j0], joints[j1]
        a = math.degrees(math.atan2(e[1] - s[1], e[0] - s[0]))
        if parent:
            # 부모의 모양 줄임 때문에 옮겨진 붙는 자리만큼 옮겨서 이어받는다
            held = mats[parent] @ [*s, 1] - chain[parent] @ [*s, 1]
            base = _move(held[0], held[1]) @ chain[parent]
        else:
            base = root_m
        stretch = _rot(a) @ np.diag([p.get("len", 1.0), 1, 1]) @ _rot(-a)
        foreshort = _rot(a) @ np.diag([math.cos(math.radians(p.get("tilt", 0))),
                                       math.cos(math.radians(p.get("turn", 0))), 1]) @ _rot(-a)
        head = _move(p.get("x", 0) * unit, p.get("y", 0) * unit) @ _move(*s) @ _rot(p.get("rot", 0)) @ stretch
        chain[name] = base @ head @ _move(-s[0], -s[1])
        mats[name] = base @ head @ foreshort @ _move(-s[0], -s[1])
    return (mats, chain) if with_chain else mats


def pick_variant(piece, pose):
    """붙인 물건의 각도별 그림 중 지금 그릴 것인가. 물건이 보는 사람 쪽(z > 0)으로 tilt 만큼 기울었으면
    그 기울기 구간(show_from ≤ tilt < show_to)의 그림을, 아니면 기본 그림(show_from < 0)을 그린다."""
    show = piece.get("show")
    if show is None:
        return True
    p = pose.get(piece["bone"], {})
    tilt = p.get("tilt", 0) if p.get("z", 0) > 0 and p.get("tilt", 0) <= 90 else 0
    return show[0] <= tilt < show[1]


def z_offsets(bones, pose):
    """뼈마다 앞뒤 순서 이동값 z (자식은 부모의 z 를 이어받아 더한다). 조각을 쌓는 순서 = depth + z."""
    out = {}
    for name, parent, _, _ in bones:
        out[name] = pose.get(name, {}).get("z", 0) + (out[parent] if parent else 0)
    return out


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

def load_anim(path, anims_dir, view=None):
    """동작 파일을 읽는다. common/ 폴더의 공통 동작이면 view 방향의 동작으로 바꿔 돌려준다."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if Path(path).parent.name == "common":
        import common_anim
        data = common_anim.view_anim(data, view)
    if "mirror_of" in data:
        # 좌우를 뒤집은 동작: 회전과 x 이동의 부호를 바꾼다 (예: 앞 그림 동작을 뒤 그림에 쓰기)
        base = load_anim(anims_dir / f"{data['mirror_of']}.json", anims_dir)
        data = {**base, **{k: v for k, v in data.items() if k != "mirror_of"}, "keys": mirror_keys(base["keys"])}
        for s in data.get("view_swap", []):
            s["view"] = {"front": "back", "back": "front"}.get(s["view"], s["view"])
    data.setdefault("frames", 8)
    data.setdefault("fps", 10)
    data.setdefault("loop", True)
    data.setdefault("interp", "ease")
    return data


def anim_files(anims_dir, view):
    """이 방향에서 만들 동작 {이름: 파일}. 같은 이름의 공통 동작(common/)이 있으면 공통 동작을 쓴다."""
    files = {p.stem: p for p in sorted((anims_dir / view).glob("*.json"))}
    for p in sorted((anims_dir / "common").glob("*.json")):
        files[p.stem] = p
    return dict(sorted(files.items()))


def anim_path(anims_dir, key):
    """'front/attack' 같은 이름 → 파일. 공통 동작이 있으면 그 파일."""
    view, name = key.split("/")
    common = anims_dir / "common" / f"{name}.json"
    return common if common.exists() else anims_dir / view / f"{name}.json"


def frame_times(anim):
    """프레임마다의 동작 시각 (0~1). 반복 동작은 끝 프레임이 처음과 겹치지 않게 i/n, 아니면 i/(n-1)."""
    n = anim["frames"]
    return [i / n if anim["loop"] else i / max(1, n - 1) for i in range(n)]


def frame_durations(anim):
    """프레임마다 머무는 시간(ms). frame_ms 가 없으면 fps 로 고르게."""
    n = anim["frames"]
    ms = anim.get("frame_ms")
    if ms is None:
        return [round(1000 / anim["fps"])] * n
    if isinstance(ms, (int, float)):
        return [int(ms)] * n
    if len(ms) != n:
        sys.exit(f"frame_ms 개수({len(ms)})가 frames({n})와 다릅니다")
    return [int(v) for v in ms]


def mirror_keys(keys):
    """좌우를 뒤집은 열쇠 프레임: 회전과 x 이동의 부호를 바꾼다."""
    return {bone: {ch: [[k[0], -k[1] if ch in ("rot", "x") else k[1], *k[2:]] for k in vals]
                   for ch, vals in chans.items()}
            for bone, chans in keys.items()}


def frame_setup(anim, models, view, t):
    """이 시각에 그릴 (모델, 자세).
    view_swap 구간이면 다른 방향 그림으로 그린다 — 앞으로 구를 때 중간에 등이 보이는 것처럼, 그 방향 그림에 없는 면을
    다른 방향 그림의 조각으로 채우는 방법이다. mirror 면 좌우를 뒤집은 값으로, add 는 그 위에 더할 값(예: 몸 전체 180도)."""
    for s in anim.get("view_swap", []):
        if s["from"] <= t < s["to"] and s["view"] in models:
            m = models[s["view"]]
            keys = mirror_keys(anim["keys"]) if s.get("mirror") else anim["keys"]
            pose = anim_pose({**anim, "keys": keys}, m, t)
            for bone, chans in s.get("add", {}).items():
                p = pose.setdefault(bone, {})
                for ch, v in chans.items():
                    p[ch] = p.get(ch, 1.0) * v if ch == "len" else p.get(ch, 0.0) + v
            return m, pose
    return models[view], anim_pose(anim, models[view], t)


def anim_pose(anim, model, t):
    """이 모델로 그릴 시각 t 의 자세. 공통 동작이면 그 방향으로 바꾼 자세 위에 방향별 2D 값을 더한다."""
    if "common" in anim:
        import common_anim
        return pose_at(anim, common_anim.convert(anim, model, t), t)
    return pose_at(anim, model.base_pose, t)


TRAIL_INNER = 0.35                 # 칼날의 손잡이 쪽 35%는 궤적에서 뺀다 (반달 모양이 되게)
TRAIL_ARC = 100                    # 궤적은 칼날 바로 뒤, 지나온 길의 마지막 이 각도(도)만 칠한다
TRAIL_COLOR = (235, 240, 255, 200)


def weapon_trail(anim, models, view, t0, t1, size, spin):
    """t0 → t1 사이에 무기 날이 지나간 자리 중 마지막 TRAIL_ARC 도. 같은 방향 그림으로 그리는 구간이 아니면 None."""
    if frame_setup(anim, models, view, t0)[0] is not frame_setup(anim, models, view, t1)[0]:
        return None
    t0 = t1 - (t1 - t0) * min(1.0, TRAIL_ARC / spin)
    mask = np.zeros((size, size), np.uint8)
    prev = None
    for s in np.linspace(0, 1, 17):
        m, pose = frame_setup(anim, models, view, t0 + s * (t1 - t0))
        mats = bone_matrices(m.bones, m.joints, pose, m.unit)
        segs = []
        for slot in m.weapon_bones:
            st, en = m.joints[slot + "_start"], m.joints[slot + "_end"]
            inner = st + TRAIL_INNER * (en - st)
            segs.append(((mats[slot] @ [*inner, 1])[:2], (mats[slot] @ [*en, 1])[:2]))
        if prev:
            for (a0, e0), (a1, e1) in zip(prev, segs):
                cv2.fillPoly(mask, [np.round([a0, e0, e1, a1]).astype(np.int32)], 1)
        prev = segs
    return mask > 0


def sample(keys, t):
    """열쇠 프레임 [[시간(0~1), 값], ...] 사이를 부드럽게 잇는다."""
    keys = sorted(keys, key=lambda k: k[0])
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0, *_), (t1, v1, *_) in zip(keys, keys[1:]):
        if t <= t1:
            u = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
            u = u * u * (3 - 2 * u)
            return v0 + (v1 - v0) * u
    return keys[-1][1]


def sample_spline(keys, t, loop):
    """열쇠 사이를 매끄럽게 이어지는 곡선(Catmull-Rom)으로 잇는다. 열쇠에서 멈추지 않고 흐른다.
    [시간, 값, "ease"] 처럼 ease 가 붙은 열쇠는 기울기를 0으로 둬서 그 자리에서 멈췄다 출발한다.
    반복 동작은 처음과 끝 열쇠의 기울기를 반대편 열쇠로 이어서 정한다. 편집기(editor/editor.js)와 똑같이 계산한다."""
    keys = sorted(keys, key=lambda k: k[0])
    n = len(keys)
    if n == 1 or t <= keys[0][0] and not loop:
        return keys[0][1]
    if t >= keys[-1][0] and not loop:
        return keys[-1][1]
    ts = [k[0] for k in keys]
    vs = [k[1] for k in keys]

    def neighbor(i):  # (시각, 값). 반복이면 반대편으로 이어 붙인다 (0 과 1 열쇠가 같은 자리면 하나 건너뛴다)
        if 0 <= i < n:
            return ts[i], vs[i]
        if not loop:
            return None
        same = n > 1 and abs(ts[0] - (ts[-1] - 1)) < 1e-9
        if i < 0:
            j = n + i - (1 if same else 0)
            return ts[j] - 1, vs[j]
        j = i - n + (1 if same else 0)
        return ts[j] + 1, vs[j]

    def tangent(i):
        if len(keys[i]) > 2 and keys[i][2] == "ease":
            return 0.0
        a, b = neighbor(i - 1), neighbor(i + 1)
        if a is None or b is None:
            return 0.0
        return (b[1] - a[1]) / (b[0] - a[0]) if b[0] != a[0] else 0.0

    if loop and (t < ts[0] or t > ts[-1]):  # 반복 동작에서 첫 열쇠 앞이나 끝 열쇠 뒤
        t0, v0, i0 = ts[-1] - 1, vs[-1], n - 1
        t1, v1, i1 = ts[0], vs[0], 0
        if t > ts[-1]:
            t0, t1 = ts[-1], ts[0] + 1
        m0, m1 = tangent(i0), tangent(i1)
    else:
        i0 = max(i for i in range(n) if ts[i] <= t)
        if i0 == n - 1:
            return vs[-1]
        i1 = i0 + 1
        t0, v0, t1, v1 = ts[i0], vs[i0], ts[i1], vs[i1]
        m0, m1 = tangent(i0), tangent(i1)
    h = t1 - t0
    if h <= 0:
        return v1
    u = (t - t0) / h
    return ((2 * u ** 3 - 3 * u ** 2 + 1) * v0 + (u ** 3 - 2 * u ** 2 + u) * h * m0
            + (-2 * u ** 3 + 3 * u ** 2) * v1 + (u ** 3 - u ** 2) * h * m1)


def pose_at(anim, base_pose, t):
    """기본 자세 + 동작 값. interp 가 "spline" 이면 이어지는 곡선, 아니면 열쇠마다 감속·가속(예전 방식)."""
    spline = anim.get("interp") == "spline"
    pose = {bone: dict(chans) for bone, chans in base_pose.items()}
    for bone, chans in anim["keys"].items():
        p = pose.setdefault(bone, {})
        for ch, keys in chans.items():
            v = sample_spline(keys, t, anim.get("loop", False)) if spline else sample(keys, t)
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


TRACK_POINTS = {"hand_l": "hand_tip_l", "hand_r": "hand_tip_r", "foot_l": "toe_l", "foot_r": "toe_r",
                "head": "head_center"}  # 따라갈 뼈 → 그 뼈 끝 관절
# 기준값은 지금 동작들의 실제 값을 재서 정했다 (보통 속도 구르기의 발끝이 프레임당 약 22칸, 칼 회전은 최대 약 285도).
JUMP_LIMIT = 25         # 손끝·발끝·머리가 한 프레임 사이에 이보다 멀리(64칸 기준 픽셀) 가면 순간이동 경고
SPIN_LIMIT = 100        # 무기가 한 프레임 사이에 이보다 많이(도) 돌면 경고 → 중간 프레임이나 궤적 효과가 필요
GAP_MIN, GAP_DROP = 0.6, 0.3  # 관절 원이 이 비율보다 덜 칠해지고, 기본 자세보다 이만큼 줄면 틈 경고
GAP_APART = 1.5         # 부모·자식 뼈가 들고 간 관절 위치가 이보다(64칸 기준 픽셀) 벌어지면 경고


def track_points(model, pose):
    """손끝·발끝·머리의 위치 (64칸 기준 픽셀)."""
    mats = bone_matrices(model.bones, model.joints, pose, model.unit)
    names = {b[0] for b in model.bones}
    return {bone: (mats[bone] @ [*model.joints[joint], 1])[:2] / model.unit
            for bone, joint in TRACK_POINTS.items() if bone in names}


def chain_angle(model, pose, bone):
    """뼈 사슬(몸통 → … → 이 뼈)의 회전값 합. 위치로 재면 180도를 넘는 회전의 방향을 알 수 없어서 값으로 더한다.
    몸 전체(root) 회전은 넣지 않는다 — 구르기처럼 몸과 함께 도는 것은 무기가 건너뛰는 것이 아니다."""
    parent = {b[0]: b[1] for b in model.bones}
    total = 0
    while bone:
        total += pose.get(bone, {}).get("rot", 0)
        bone = parent[bone]
    return total


def weapon_spin(model, pose, prev_pose, bone, screen=False):
    """한 프레임 사이에 무기가 돈 각도. 방향별 동작은 회전값(chain_angle)으로 잰다.
    공통 동작(screen=True)은 화면에 보이는 칼날 방향으로 잰다 — 칼이 보는 사람 쪽을 지날 때
    회전값이 180도 건너뛰어도(짧아진 칼을 뒤집어 그리는 것) 화면에서는 이어져 보이기 때문이다.
    칼이 반보다 짧아 보이는(보는 사람 쪽을 향한) 프레임이 끼면 방향이 의미 없으니 0 으로 본다."""
    if not screen:
        return abs(chain_angle(model, pose, bone) - chain_angle(model, prev_pose, bone))
    j0, j1 = next((b[2], b[3]) for b in model.bones if b[0] == bone)
    full = float(np.linalg.norm(np.subtract(model.joints[j1], model.joints[j0])))
    angles = []
    for p in (pose, prev_pose):
        m = bone_matrices(model.bones, model.joints, p, model.unit)[bone]
        s, e = m @ [*model.joints[j0], 1], m @ [*model.joints[j1], 1]
        if math.hypot(e[0] - s[0], e[1] - s[1]) < 0.5 * full:
            return 0.0
        angles.append(math.degrees(math.atan2(e[1] - s[1], e[0] - s[0])))
    return abs((angles[0] - angles[1] + 180) % 360 - 180)


def joint_cover(model, pose, img):
    """관절마다 (둘레 원 안이 칠해진 비율, 부모·자식 뼈가 들고 간 관절 위치의 거리(64칸 기준 픽셀)).
    조각 방식만 잰다 (그물 방식은 늘어나서 틈이 생기지 않는다). 원이 절반 넘게 칸 밖이면 뺀다 (잘림 경고가 따로 난다)."""
    if not hasattr(model, "gap_joints"):
        return {}
    mats = bone_matrices(model.bones, model.joints, pose, model.unit)
    parent = {b[0]: b[1] for b in model.bones}
    solid = np.asarray(img)[..., 3] >= model.thr
    size = solid.shape[0]
    out = {}
    for bone, joint, r in model.gap_joints:
        # 허벅지는 뼈대에서는 부모가 없지만 실제로는 골반에 붙어 있다
        holder = parent[bone] or ("hips" if "hips" in parent else None)
        child = (mats[bone] @ [*model.joints[joint], 1])[:2]
        held = (mats[holder] @ [*model.joints[joint], 1])[:2] if holder else child
        cx, cy = (child + held) / 2
        yy, xx = np.mgrid[int(cy - r):int(cy + r) + 1, int(cx - r):int(cx + r) + 1]
        disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        inside = disc & (xx >= 0) & (xx < size) & (yy >= 0) & (yy < size)
        if inside.sum() < disc.sum() / 2:
            continue
        out[bone] = (solid[yy[inside], xx[inside]].sum() / inside.sum(),
                     float(np.linalg.norm(child - held)) / model.unit)
    return out


def frame_warnings(frame_models, poses, frames, loop, trailed=(), screen_spin=False):
    """프레임마다 경고 목록: 칸 밖 잘림, 관절 벌어짐·틈, 순간이동, 무기 큰 회전.
    frame_models: 프레임마다 그린 모델 (view_swap 으로 바뀔 수 있다). 모델이 바뀌는 프레임 사이는 비교하지 않는다.
    trailed: 궤적 효과를 그린 프레임 (무기 큰 회전을 경고하지 않는다)."""
    for m in set(frame_models):
        if not hasattr(m, "_ref_cover"):
            base = {bone: dict(ch) for bone, ch in m.base_pose.items()}
            m._ref_cover = joint_cover(m, base, m.render(base))
    pts = [track_points(m, p) for m, p in zip(frame_models, poses)]
    out = []
    for i, (model, pose, img) in enumerate(zip(frame_models, poses, frames)):
        warn = []
        cut = clipped_sides(img, model.k)
        if cut:
            warn.append(f"칸 밖으로 잘림 ({', '.join(cut)})")
        for bone, (cover, apart) in joint_cover(model, pose, img).items():
            ref = model._ref_cover.get(bone, (1, 0))[0]
            if apart > GAP_APART:
                warn.append(f"{bone} 관절 벌어짐 ({apart:.1f}칸)")
            elif cover < GAP_MIN and ref - cover > GAP_DROP:
                warn.append(f"{bone} 관절 틈 (칠해짐 {cover:.0%}, 기본 {ref:.0%})")
        prev = i - 1 if i > 0 else (len(poses) - 1 if loop else None)
        if prev is not None and frame_models[prev] is model:
            for bone, p in pts[i].items():
                d = float(np.linalg.norm(p - pts[prev][bone]))
                if d > JUMP_LIMIT:
                    warn.append(f"{bone} 순간이동 ({prev}→{i}번 프레임 {d:.0f}칸)")
            for bone in ([] if i in trailed else model.weapon_bones):
                spin = weapon_spin(model, pose, poses[prev], bone, screen_spin)
                if spin > SPIN_LIMIT:
                    warn.append(f"{bone} 한 프레임에 {spin:.0f}도 회전 ({prev}→{i}번) → 중간 프레임이나 궤적 효과 필요")
        out.append(warn)
    return out


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


def save_image(img, path, **kwargs):
    """그림 뷰어가 파일을 열어 두면 윈도우에서 저장이 막힐 수 있다.
    잠깐 기다렸다 다시 시도하고, 끝내 막히면 이름 뒤에 _new 를 붙여 저장한 뒤 계속 진행한다."""
    for _ in range(5):
        try:
            img.save(path, **kwargs)
            return path
        except OSError:
            time.sleep(0.5)
    alt = path.with_name(f"{path.stem}_new{path.suffix}")
    img.save(alt, **kwargs)
    print(f"  주의 {path} 가 다른 프로그램에 열려 있어 저장하지 못했습니다 → {alt.name} 로 저장")
    return alt


def _font(size):
    try:
        return ImageFont.truetype("malgun.ttf", size)  # 한글이 나오는 윈도우 기본 글꼴
    except OSError:
        return ImageFont.load_default()


def draw_frames_check(pix_frames, warns, name, scale=5):
    """프레임마다 5배로 키운 확인 그림. 경고가 난 프레임은 빨간 테두리와 경고 내용을 아래에 적는다."""
    canvas = pix_frames[0].width
    cell = canvas * scale
    font, spacing = _font(13), 3
    texts = ["\n".join(_wrap(w, font, cell - 8) for w in ws) for ws in warns]
    measure = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    text_h = max([measure.multiline_textbbox((0, 0), t, font=font, spacing=spacing)[3] for t in texts if t] + [0])
    sheet = Image.new("RGBA", (cell * len(pix_frames), 24 + cell + 30 + text_h), (255, 255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((4, 4), name, fill=(0, 0, 0, 255), font=font)
    for i, (f, ws, text) in enumerate(zip(pix_frames, warns, texts)):
        x = i * cell
        sheet.paste(on_gray(f, (190, 190, 190, 255)).resize((cell, cell), Image.NEAREST), (x, 24))
        if ws:
            d.rectangle([x + 1, 25, x + cell - 2, 24 + cell - 2], outline=(220, 30, 30, 255), width=4)
        d.text((x + 4, 28 + cell), f"#{i}", fill=(0, 0, 0, 255), font=font)
        if text:
            d.multiline_text((x + 4, 46 + cell), text, fill=(200, 20, 20, 255), font=font, spacing=spacing)
    return sheet


def _wrap(text, font, width):
    """단어 단위로, 실제 글자 폭(width 화소)에 맞춰 줄을 바꾼다."""
    lines, cur = [], ""
    for word in text.split(" "):
        trial = f"{cur} {word}".strip()
        if cur and font.getlength(trial) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return "\n".join(lines + [cur])


def write_layer_sheets(out, name, model, part_frames, hi_frames, palette, cfg):
    """(층, depth) 마다 시트를 하나씩 쓴다. 파일 이름 앞 두 자리는 depth 순위(작을수록 뒤).
    겹칠 경계에 검은 줄이 생기지 않도록 층 시트에는 외곽선을 두르지 않는다."""
    canvas = cfg["canvas"]
    layer_dir = out / "layers" / name
    shutil.rmtree(layer_dir, ignore_errors=True)  # 열려 있는 파일은 남을 수 있다 (save_image 가 처리)
    layer_dir.mkdir(parents=True, exist_ok=True)
    # view_swap 프레임은 다른 방향 그림의 depth 를 쓰므로 실제로 나온 depth 까지 합친다
    rank = {d: i for i, d in enumerate(sorted(set(model.depth_levels()) | {d for _, d in part_frames}))}
    thr, k, n = cfg["alpha_threshold"], model.k, len(hi_frames)

    def per_cell(mask):  # 64칸의 칸마다 그림이 차지한 비율
        return mask.reshape(canvas, k, canvas, k).mean(axis=(1, 3))

    # 한 칸이 여러 층에 나뉘어 걸치면 어느 층도 절반이 안 돼 구멍이 난다.
    # 그래서 칠할지는 합친 그림과 똑같은 규칙(그 칸의 절반 이상이 진함)으로 정하고,
    # 그 칸은 그림이 가장 많이 걸친 층이 맡는다. 가장자리의 옅은 픽셀도 세어야 빈칸이 생기지 않는다.
    filled = [per_cell(np.asarray(h)[..., 3] >= thr) >= 0.5 for h in hi_frames]
    default_keys = [key for key in part_frames if key[0] in model.default_layers]
    alpha = {key: {i: np.asarray(img)[..., 3] for i, img in frames.items()} for key, frames in part_frames.items()}
    any_cell = {key: {i: per_cell(a > 0) for i, a in frames.items()} for key, frames in alpha.items()}

    def keep_mask(key, i):
        own = any_cell[key][i]
        others = [any_cell[o][i] for o in default_keys if o != key and i in any_cell[o]]
        best = own >= np.max(others, axis=0) if others else True
        alone = per_cell(alpha[key][i] >= thr) >= 0.5  # 바꿔 끼우는 변형은 합친 그림에 없으므로 제 모양으로도 칠한다
        return alone | (filled[i] & best & (own > 0))

    parts = []
    stacked = [Image.new("RGBA", (canvas, canvas)) for _ in range(n)]
    for (layer_i, depth) in sorted(part_frames, key=lambda kd: (rank[kd[1]], model.slots.index(model.layers[kd[0]]["slot"]), kd[0])):
        frames = part_frames[(layer_i, depth)]
        small = [px.downscale(frames[i], (canvas, canvas), palette, 1, keep_mask((layer_i, depth), i)) if i in frames
                 else Image.new("RGBA", (canvas, canvas)) for i in range(n)]
        if not any(np.asarray(s)[..., 3].any() for s in small):
            continue
        layer = model.layers[layer_i]
        sheet = Image.new("RGBA", (canvas * n, canvas))
        for i, s in enumerate(small):
            sheet.paste(s, (i * canvas, 0))
        file = save_image(sheet, layer_dir / f"{rank[depth]:02d}_{layer['name']}.png").name
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

    models = {}
    for view, rig in rigs.items():
        if not args.mesh and (args.character / "parts" / view / "parts.json").exists():
            import parts  # 조각이 있으면 조각 방식 (parts.py)
            print(f"  {view}: 조각 방식")
            models[view] = parts.PartModel(args.character, rig, cfg)
        else:
            print(f"  {view}: 그물 방식 (메시와 무게 준비 중...)")
            models[view] = Model(args.character, rig, cfg)

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
    warnings_path = out / "check" / "warnings.json"
    warnings_all = json.loads(warnings_path.read_text(encoding="utf-8")) if warnings_path.exists() else {}
    canvas = cfg["canvas"]
    for view, model in models.items():
        for stem, path in anim_files(anims_dir, view).items():
            key = f"{view}/{stem}"
            if args.only and key not in args.only:
                continue
            anim = load_anim(path, anims_dir, view)
            n = anim["frames"]
            hi_frames, pix_frames, poses, frame_models, trailed = [], [], [], [], set()
            part_frames = {}  # (층, depth) → {프레임 번호: 그림}
            times = frame_times(anim)
            for i, t in enumerate(times):
                fm, pose = frame_setup(anim, models, view, t)
                poses.append(pose)
                frame_models.append(fm)
                if args.layers:
                    hi, pieces = fm.render(pose, parts=True)
                    for key, img in pieces.items():
                        part_frames.setdefault(key, {})[i] = img
                else:
                    hi = fm.render(pose)
                # 베는 프레임에는 칼날이 지나간 자리를 몸 뒤에 칠한다 (궤적 효과, 합친 시트에만).
                # "trail": true 면 무기가 크게 돈 프레임 모두, [3] 처럼 목록이면 그 프레임만.
                trail_on = anim.get("trail")
                # 궤적을 그릴 프레임을 직접 고른 경우(목록)는 칼이 짧아 보이는 프레임이 끼어도 그리도록 회전값으로 잰다
                same = i > 0 and frame_models[i - 1] is fm
                spin = max([weapon_spin(fm, pose, poses[i - 1], w, "common" in anim and trail_on is True)
                            for w in fm.weapon_bones] + [0]) if same else 0
                if (trail_on is True and spin > SPIN_LIMIT) or (isinstance(trail_on, list) and i in trail_on and spin > 0):
                    mask = weapon_trail(anim, models, view, times[i - 1], t, fm.size, spin)
                    if mask is not None:
                        trail = np.zeros((fm.size, fm.size, 4), np.uint8)
                        trail[mask] = TRAIL_COLOR
                        hi = Image.alpha_composite(Image.fromarray(trail), hi)
                        trailed.add(i)
                small = px.downscale(hi, (canvas, canvas), palette, cfg["alpha_threshold"])
                if cfg["outline"]:
                    small = px.add_outline(small, px.hex_to_rgb(cfg["outline_color"]))
                hi_frames.append(hi)
                pix_frames.append(small)

            name = f"{view}_{stem}"
            warns = frame_warnings(frame_models, poses, hi_frames, anim["loop"], trailed, "common" in anim)
            for i, ws in enumerate(warns):
                for w in ws:
                    print(f"  주의 {name} {i}번 프레임: {w}")
            warnings_all[name] = {str(i): ws for i, ws in enumerate(warns) if ws}
            save_image(draw_frames_check(pix_frames, warns, name), out / "check" / f"{name}_frames.png")
            sheet = Image.new("RGBA", (canvas * n, canvas))
            for i, f in enumerate(pix_frames):
                sheet.paste(f, (i * canvas, 0))
            save_image(sheet, out / f"{name}.png")
            gif = [on_gray(f).resize((canvas * 4, canvas * 4), Image.NEAREST).convert("P") for f in pix_frames]
            durations = frame_durations(anim)
            save_image(gif[0], out / "check" / f"{name}.gif", save_all=True, append_images=gif[1:],
                       duration=durations, loop=0)
            meta[name] = {"frames": n, "fps": anim["fps"], "frame_ms": durations, "loop": anim["loop"],
                          "frame_size": canvas, "pivot": [0.5, cfg["foot_margin"] / canvas]}
            stacked = None
            if args.layers:
                parts, stacked = write_layer_sheets(out, name, model, part_frames, hi_frames, palette, cfg)
                layers_meta[name] = {**meta[name], "parts": parts}
            save_image(draw_anim_check(hi_frames, pix_frames, stacked), out / "check" / f"{name}.png")
            print(f"  {name}.png ({n}프레임)" + (f", 층 시트 {len(parts)}장" if args.layers else ""))
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    warnings_path.write_text(json.dumps(warnings_all, indent=2, ensure_ascii=False), encoding="utf-8")
    count = sum(len(ws) for frames in warnings_all.values() for ws in frames.values())
    print(f"경고 {count}개 (목록: {warnings_path}, 프레임별 큰 그림: {out / 'check'}/*_frames.png)")
    if args.layers:
        layers_path.write_text(json.dumps(layers_meta, indent=2, ensure_ascii=False), encoding="utf-8")


def cmd_angles(args):
    """동작의 프레임마다 팔·무기가 화면에서 실제로 가리키는 각도와 무기 끝 위치를 표로 출력한다.
    회전값을 머릿속으로 더해 자세를 짐작하지 말고, 이 표로 의도와 맞는지 확인한다.
    (서 있을 때도 아래팔은 위팔보다 바깥으로 기울어 있어서, '회전 0 = 똑바로 아래'가 아니다.)"""
    cfg = load_config(args.config)
    models = {}
    for view in VIEWS:
        rig_path = args.character / f"rig_{view}.json"
        if rig_path.exists():
            rig = json.loads(rig_path.read_text(encoding="utf-8"))
            if (args.character / "parts" / view / "parts.json").exists():
                import parts
                models[view] = parts.PartModel(args.character, rig, cfg)
            else:
                models[view] = Model(args.character, rig, cfg)
    for key in args.motions:
        view = key.split("/")[0]
        anim = load_anim(anim_path(args.anims, key), args.anims, view)
        n = anim["frames"]
        print(f"{key}  (화면 각도: 0=오른쪽, 90=아래, 180=왼쪽, -90=위 / 끝 위치: 캐릭터 키 48 기준 칸)")
        for i in range(n):
            t = i / n if anim["loop"] else i / max(1, n - 1)
            m, pose = frame_setup(anim, models, view, t)
            mats = bone_matrices(m.bones, m.joints, pose, m.unit)
            cols = []
            for name, _, j0, j1 in m.bones:
                if name.startswith(("upper_arm", "forearm")) or name in m.weapon_bones:
                    s = (mats[name] @ [*m.joints[j0], 1])[:2]
                    e = (mats[name] @ [*m.joints[j1], 1])[:2]
                    a = math.degrees(math.atan2(e[1] - s[1], e[0] - s[0]))
                    cols.append(f"{name} {a:5.0f}" + (f" 끝({e[0] / m.unit:.0f},{e[1] / m.unit:.0f})"
                                                     if name in m.weapon_bones else ""))
            print(f"  #{i} [{m.view}] " + " | ".join(cols))


def main():
    parser = argparse.ArgumentParser(description="T자 일러스트로 뼈대 동작을 만들어 픽셀 스프라이트로 찍습니다.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_rig = sub.add_parser("rig", help="관절을 찾아 rig_<방향>.json 을 만든다")
    p_rig.add_argument("character", type=Path, help="front.png / back.png / left.png / right.png 가 든 캐릭터 폴더")
    p_rig.add_argument("-c", "--config", type=Path, default=HERE / "config.json")
    p_rig.set_defaults(func=cmd_rig)

    p_parts = sub.add_parser("parts", help="뼈대를 따라 부위별 조각 그림을 만든다 (가려진 곳·관절 끝 자동 채우기)")
    p_parts.add_argument("character", type=Path)
    p_parts.add_argument("-c", "--config", type=Path, default=HERE / "config.json")
    p_parts.add_argument("--force", action="store_true", help="이미 있는 조각을 덮어쓴다")
    p_parts.add_argument("--views", nargs="*", help="이 방향만 (예: right)")
    p_rig.add_argument("--views", nargs="*", help="이 방향만 (예: front_left). 이미 고친 다른 방향 rig 를 덮어쓰지 않으려면 꼭 준다")

    p_mirror = sub.add_parser("mirror", help="한 방향 그림을 좌우로 뒤집어 다른 방향을 만든다 (예: left right)")
    p_mirror.add_argument("character", type=Path)
    p_mirror.add_argument("source", choices=VIEWS)
    p_mirror.add_argument("target", choices=VIEWS)
    p_mirror.add_argument("-c", "--config", type=Path, default=HERE / "config.json")
    p_mirror.set_defaults(func=cmd_mirror)
    p_parts.set_defaults(func=lambda a: __import__("parts").cmd_parts(a))

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
    p_render.add_argument("--mesh", action="store_true",
                          help="조각(parts/)이 있어도 예전 그물 방식으로 그린다")
    p_render.add_argument("--layers", action="store_true",
                          help="층(몸·덮개·물건)과 앞뒤 순서별로 나눈 시트와 layers.json 도 만든다")
    p_render.set_defaults(func=cmd_render)

    p_angles = sub.add_parser("angles", help="동작의 프레임마다 팔·무기의 실제 화면 각도와 무기 끝 위치를 출력한다")
    p_angles.add_argument("character", type=Path)
    p_angles.add_argument("motions", nargs="+", help="예: front/attack side/skill")
    p_angles.add_argument("-c", "--config", type=Path, default=HERE / "config.json")
    p_angles.add_argument("-a", "--anims", type=Path, default=HERE / "anims")
    p_angles.set_defaults(func=cmd_angles)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
