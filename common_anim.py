"""공통 동작: 동작 하나를 몸 기준 각도로 적어 두고, 앞·뒤·왼쪽·오른쪽 그림마다 2D 자세로 바꾼다.
앞에서 팔을 앞으로 들면 옆 그림에서도 같은 만큼 들린다 — 방향마다 따로 만들 필요가 없다.

몸 좌표: X = 캐릭터의 왼쪽, Y = 아래, Z = 캐릭터의 앞.
뼈마다 쉼 자세(rig 의 base_pose 까지 적용한 자세) 위에서 돌린다. 값은 모두 도(°).
  swing  : 앞뒤로 휘두르기 (X 축 회전). +면 팔다리 끝이 앞으로 간다, 몸통·머리는 +면 앞으로 숙인다. 옆 그림에서 그대로 보인다.
  spread : 옆으로 벌리기 (Z 축 회전). 팔다리는 +면 몸 바깥쪽 (왼쪽·오른쪽이 서로 거울). 앞 그림에서 그대로 보인다.
           몸통·머리 같은 가운데 뼈는 +면 앞 그림에서 시계 방향.
  twist  : 뼈를 제 축으로 비틀기. 팔다리는 +면 바깥쪽으로 (좌우 거울).
  돌리는 순서: twist → spread → swing. 그래서 swing 90(팔을 앞으로 듦)에서 spread 를 바꾸면 팔이 수평으로 쓸고 지나간다
  (가로 베기). spread 90(옆으로 듦)에서는 swing 이 제 축 돌리기가 된다.
  len    : 뼈 길이 비율 (2D 와 같음, 자식에게 이어짐).
root (몸 전체): x = 캐릭터 왼쪽으로, y = 아래로, fwd = 앞으로 (칸 단위, 2D 의 x/y 와 같은 크기),
  swing = 앞으로 숙이기, spread = 앞 그림에서 시계 방향으로 기울이기, twist = 제자리에서 캐릭터 왼쪽으로 돌기,
  ground = 바닥 붙이기 (0~1). 1 이면 몸(칼·망토 빼고)의 가장 낮은 곳이 쉼 자세의 발 높이에 닿도록 몸 전체를 올리고 내린다
  (무릎을 굽히면 몸이 저절로 내려앉고, 구르면 몸이 바닥을 따라 구른다). y 는 그 위에 더한다. 공중에 뜨는 구간은 0 으로.

따라 움직임: 동작 파일의 "follow": {"cape": {"of": "torso", "lag": 2, "gain": 1}} — cape 는 root~torso 가 돈 만큼
  lag 프레임 늦게 따라 돈다 (그동안 원래 방향을 붙잡고 있다가 뒤늦게 따라온다). 그 방향 rig 에 없는 뼈는 건너뛴다.

뼈 조각은 쉼 자세에서 화면에 평평하게 놓인 카드로 보고, 3D 로 돌린 카드를 그 방향 화면에 비춘 모양을
2D 의 rot(돌리기)·tilt(뼈 방향으로 짧아짐)·turn(폭이 좁아짐)으로 나눈다. 앞뒤 순서(z)는 3D 로 보는 사람 쪽에 나온 만큼.
편집기(editor/editor.js)가 똑같이 계산한다."""
import math

import numpy as np

import animate as an

# 몸 좌표 → (화면 x, 화면 y, 보는 사람 쪽)
VIEW_AXES = {
    "front": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float),
    "back": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], float),
    "left": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float),
    "right": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float),
}
# 대각선(45도): 이웃한 두 방향의 가운데. 예) 앞왼쪽은 앞(화면 x = X)과 왼쪽(화면 x = -Z) 사이
_H = 2 ** -0.5
VIEW_AXES.update({
    "front_left": np.array([[_H, 0, -_H], [0, 1, 0], [_H, 0, _H]]),
    "back_left": np.array([[-_H, 0, -_H], [0, 1, 0], [_H, 0, -_H]]),
    "front_right": np.array([[_H, 0, _H], [0, 1, 0], [-_H, 0, _H]]),
    "back_right": np.array([[-_H, 0, _H], [0, 1, 0], [-_H, 0, -_H]]),
})
BONE_CHANNELS = ("swing", "spread", "twist", "len")
ROOT_CHANNELS = ("x", "y", "fwd", "swing", "spread", "twist")
UP_KINDS = ("torso", "head")  # 위로 뻗은 뼈: swing + 가 앞으로 숙이기가 되게 부호를 바꾼다
TWIST_AXIS = {"torso": (0, -1, 0), "head": (0, -1, 0), "foot": (0, 0, 1)}  # 나머지는 아래 (0, 1, 0)
Z_SCALE = 8  # 몸통 길이만큼 보는 사람 쪽으로 나오면 앞뒤 순서가 8 만큼 앞으로


def rot_x(d):
    c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(d):
    c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(d):
    c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_axis(u, d):
    u = np.asarray(u, float)
    k = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    r = math.radians(d)
    return np.eye(3) + math.sin(r) * k + (1 - math.cos(r)) * (k @ k)


def kind_of(name):
    return name.rsplit("_", 1)[0] if name.endswith(("_l", "_r")) else name


def local_rotation(name, ch):
    """뼈 하나의 몸 기준 회전과, 그중 각 방향 화면 안에서 도는 몫 (앞 그림 기준 spread, 옆 그림 기준 swing)."""
    m = -1 if name.endswith("_l") else 1
    up = -1 if kind_of(name) in UP_KINDS else 1
    spread, swing, twist = m * ch.get("spread", 0), up * ch.get("swing", 0), m * ch.get("twist", 0)
    axis = TWIST_AXIS.get(kind_of(name), (0, 1, 0))
    return rot_x(swing) @ rot_z(spread) @ rot_axis(axis, twist), spread, swing


def in_plane(view, spread, swing):
    # 회전값이 한 바퀴 건너뛰지 않게 고를 때 기준으로만 쓴다 (대각선은 두 방향의 평균)
    return {"front": spread, "back": -spread, "left": swing, "right": -swing,
            "front_left": _H * (spread + swing), "back_left": _H * (swing - spread),
            "front_right": _H * (spread - swing), "back_right": -_H * (spread + swing)}[view]


def wrap(d):
    return (d + 180) % 360 - 180


def split_card(n2, near):
    """2x2 화면 변환 n2 (뼈 방향 기준) → (rot, 뼈 방향 비율, 폭 비율). 폭 비율은 0 이상으로 둔다.
    rot 는 near 에 가장 가까운 값으로 고른다 (프레임 사이에 360도 튀지 않게)."""
    a = n2[0, 0] ** 2 + n2[1, 1] ** 2
    b = n2[1, 0] ** 2 + n2[0, 1] ** 2
    c = n2[0, 0] * n2[1, 0] - n2[1, 1] * n2[0, 1]
    r = 0.5 * math.atan2(2 * c, a - b)
    cs, sn = math.cos(r), math.sin(r)
    along = cs * n2[0, 0] + sn * n2[1, 0]
    across = -sn * n2[0, 1] + cs * n2[1, 1]
    if across < 0:
        r, along, across = r + math.pi, -along, -across
    r = math.degrees(r)
    return near + wrap(r - near), along, across


def neutral(model):
    """쉼 자세(base_pose 적용)의 뼈 시작·끝 자리. 모델마다 한 번만 계산한다."""
    cached = getattr(model, "_neutral", None)
    if cached is None:
        mats = an.bone_matrices(model.bones, model.joints, model.base_pose, model.unit)
        cached = {name: ((mats[name] @ [*model.joints[j0], 1])[:2], (mats[name] @ [*model.joints[j1], 1])[:2])
                  for name, _, j0, j1 in model.bones}
        model._neutral = cached
    return cached


def sample_channels(keys, t, anim):
    spline = anim.get("interp") == "spline"
    return {bone: {ch: an.sample_spline(k, t, anim.get("loop", False)) if spline else an.sample(k, t)
                   for ch, k in chans.items()}
            for bone, chans in keys.items()}


def chain_of(model, bone):
    """root 다음부터 bone 까지 뼈 이름 (부모 먼저)."""
    parents = {name: parent for name, parent, _, _ in model.bones}
    out = []
    while bone:
        out.insert(0, bone)
        bone = parents.get(bone)
    return out


def turned(vals, chain):
    """root 와 chain 뼈들이 실제로 돈 각도의 합 (swing 축, spread 축). 위로 뻗은 뼈·root 의 swing 부호를 맞춘다."""
    rc = vals.get("root", {})
    swing, spread = -rc.get("swing", 0), rc.get("spread", 0)
    for name in chain:
        ch = vals.get(name, {})
        swing += (-1 if kind_of(name) in UP_KINDS else 1) * ch.get("swing", 0)
        spread += (-1 if name.endswith("_l") else 1) * ch.get("spread", 0)
    return swing, spread


def apply_follow(anim, model, t, vals):
    """follow 뼈에 '조금 전 시각에 부모 쪽이 돈 각도 - 지금 돈 각도' 를 더한다."""
    names = {b[0] for b in model.bones}
    n = anim.get("frames", 8)
    step = 1 / n if anim.get("loop") else 1 / max(1, n - 1)
    for bone, f in anim.get("follow", {}).items():
        if bone not in names or f["of"] not in names:
            continue
        chain = chain_of(model, f["of"])
        t0 = t - f.get("lag", 2) * step
        t0 = t0 % 1 if anim.get("loop") else max(0.0, t0)
        now, before = turned(vals, chain), turned(sample_channels(anim["common"], t0, anim), chain)
        gain = f.get("gain", 1.0)
        ch = vals.setdefault(bone, {})
        ch["swing"] = ch.get("swing", 0) + gain * (before[0] - now[0]) * (-1 if kind_of(bone) in UP_KINDS else 1)
        ch["spread"] = ch.get("spread", 0) + gain * (before[1] - now[1]) * (-1 if bone.endswith("_l") else 1)


def ground_shift(model, pose):
    """몸(기본 뼈)의 가장 낮은 곳이 쉼 자세의 가장 낮은 곳(발)에 오려면 몸을 아래로 옮길 칸 수."""
    rest = neutral(model)
    feet = [b for b in an.BONE_NAMES if b in rest]
    mats = an.bone_matrices(model.bones, model.joints, pose, model.unit)
    by_name = {name: (j0, j1) for name, _, j0, j1 in model.bones}
    low_rest = max(max(rest[b][0][1], rest[b][1][1]) for b in feet)
    low_now = max((mats[b] @ [*model.joints[j], 1])[1] for b in feet for j in by_name[b])
    return (low_rest - low_now) / model.unit


def convert(anim, model, t):
    """공통 동작의 시각 t 자세 → 이 모델(방향) 의 2D 자세 (base_pose 포함)."""
    view = model.view
    va = VIEW_AXES[view]
    vals = sample_channels(anim["common"], t, anim)
    apply_follow(anim, model, t, vals)
    rest = neutral(model)
    joints = model.joints
    center = (np.asarray(joints["pelvis"], float) + joints["neck"]) / 2
    torso_len = float(np.linalg.norm(np.subtract(joints["neck"], joints["pelvis"])))

    rc = vals.get("root", {})
    g_root = rot_z(rc.get("spread", 0)) @ rot_x(-rc.get("swing", 0)) @ rot_y(rc.get("twist", 0))
    m3_root = va @ g_root @ va.T
    root_r, _, _ = split_card(m3_root[:2, :2], in_plane(view, rc.get("spread", 0), -rc.get("swing", 0)))
    pos = va @ [rc.get("x", 0), rc.get("y", 0), rc.get("fwd", 0)]

    pose = {bone: dict(chans) for bone, chans in model.base_pose.items()}
    root = pose.setdefault("root", {})
    root["x"] = root.get("x", 0) + pos[0]
    root["y"] = root.get("y", 0) + pos[1]
    root["rot"] = root.get("rot", 0) + root_r

    g, near, rot_g, m3s, starts, z_total = {}, {}, {}, {}, {}, {}
    for name, parent, _, _ in model.bones:
        ch = vals.get(name, {})
        local, spread, swing = local_rotation(name, ch)
        g[name] = (g[parent] if parent else g_root) @ local
        m3 = va @ g[name] @ va.T
        m3s[name] = m3
        near[name] = (near[parent] if parent else root_r) + in_plane(view, spread, swing)
        s0, e0 = rest[name]
        ang = math.degrees(math.atan2(e0[1] - s0[1], e0[0] - s0[0]))
        rot_n = an._rot(-ang)[:2, :2] @ m3[:2, :2] @ an._rot(ang)[:2, :2]
        rot_g[name], along, across = split_card(rot_n, near[name])
        if parent:
            starts[name] = starts[parent] + m3s[parent] @ [*(s0 - rest[parent][0]), 0]
        else:
            starts[name] = np.array([*center, 0]) + m3_root @ [*(s0 - center), 0]
        mid = starts[name] + m3 @ [*((e0 - s0) / 2), 0]
        z_total[name] = Z_SCALE * mid[2] / torso_len

        p = pose.setdefault(name, {})
        p["rot"] = p.get("rot", 0) + rot_g[name] - (rot_g[parent] if parent else root_r)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, along))))
        turn = math.degrees(math.acos(max(-1.0, min(1.0, across))))
        if tilt > 1e-6:
            p["tilt"] = p.get("tilt", 0) + tilt
        if turn > 1e-6:
            p["turn"] = p.get("turn", 0) + turn
        z = z_total[name] - (z_total[parent] if parent else 0)
        if abs(z) > 1e-6:
            p["z"] = p.get("z", 0) + z
        if "len" in ch:
            p["len"] = p.get("len", 1.0) * ch["len"]
        if not parent:
            # 몸 전체 회전의 짧아짐(숙이면 위아래로 납작)을 2D 의 root 돌리기로는 못 하니, 붙는 자리를 옮겨 맞춘다
            want = m3_root[:2, :2] @ (s0 - center)
            got = an._rot(root_r)[:2, :2] @ (s0 - center)
            dx, dy = an._rot(-root_r)[:2, :2] @ (want - got) / model.unit
            if abs(dx) + abs(dy) > 1e-6:
                p["x"] = p.get("x", 0) + dx
                p["y"] = p.get("y", 0) + dy
    if rc.get("ground", 0):
        root["y"] += rc["ground"] * ground_shift(model, pose)
    return pose


def view_anim(data, view):
    """공통 동작 파일 → 한 방향의 동작. views.<방향> 의 값(trail, view_swap, 2D keys 등)이 그 방향에만 덧붙는다.
    2D keys 는 바꾼 자세 위에 더해진다 (그 방향만 손볼 때).
    views.<방향>.common 은 공통 채널을 그 방향에서만 바꿔 쓴다 (뼈·채널 단위로 통째로 바꿈).
    예: 앞 그림 구르기는 몸 전체를 앞으로 돌리면 납작한 그림이 뒤집히기만 하니, 앞 그림에서는 root swing 을 작게."""
    extra = data.get("views", {}).get(view, {})
    out = {k: v for k, v in data.items() if k not in ("views", "keys")}
    out.update({k: v for k, v in extra.items() if k not in ("keys", "common")})
    out["common"] = {bone: dict(chans) for bone, chans in data.get("keys", {}).items()}
    for bone, chans in extra.get("common", {}).items():
        out["common"].setdefault(bone, {}).update(chans)
    out["keys"] = extra.get("keys", {})
    return out
