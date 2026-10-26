// 동작 편집기. 자세 계산은 animate.py 의 같은 이름 함수와 똑같이 맞춘다
// (bone_matrices, sample, sample_spline, pose_at, mirror_keys, frame_setup, frame_times, frame_durations).
"use strict";

const CHANNELS = ["rot", "len", "x", "y", "tilt", "turn", "z"];
const DEFAULT = { rot: 0, len: 1, x: 0, y: 0, tilt: 0, turn: 0, z: 0, swing: 0, spread: 0, twist: 0, fwd: 0, ground: 0 };
const CHANNEL_UI = {   // [최소, 최대, 간격, 설명]
  rot: [-360, 360, 1, "회전 (화면 시계 방향 +)"],
  len: [0.3, 1.6, 0.01, "뼈 방향 길이 비율"],
  x: [-30, 30, 0.5, "옆 이동 (칸)"],
  y: [-30, 30, 0.5, "위아래 이동 (칸, 아래 +)"],
  tilt: [-80, 80, 1, "앞뒤 기울기 — 보는 쪽 +, 뼈 방향으로 짧아짐 (Alt+위아래 끌기)"],
  turn: [-80, 80, 1, "제 축으로 돌리기 — 폭이 좁아짐"],
  z: [-5, 5, 0.5, "앞뒤 순서 — 앞 +, 딸린 뼈도 함께"],
};
// 공통 동작(anims/common/)의 채널 — common_anim.py 의 설명과 같다
const BONE_COMMON = ["swing", "spread", "twist", "len"];
const ROOT_COMMON = ["x", "y", "fwd", "swing", "spread", "twist", "ground"];
const COMMON_UI = {
  swing: [-360, 360, 1, "앞뒤로 휘두르기 — 팔다리 끝이 앞으로 +, 몸통·머리·몸 전체는 앞으로 숙이기 + (옆 그림에서 끌기)"],
  spread: [-360, 360, 1, "옆으로 벌리기 — 팔다리는 바깥쪽 +, 가운데 뼈는 앞 그림에서 시계 방향 + (앞 그림에서 끌기)"],
  twist: [-180, 180, 1, "제 축으로 비틀기 — 팔다리는 바깥쪽 +, 몸 전체는 캐릭터 왼쪽으로 돌기 +"],
  len: CHANNEL_UI.len,
  x: [-30, 30, 0.5, "캐릭터 왼쪽으로 이동 (칸) — 앞 그림에서 오른쪽"],
  y: CHANNEL_UI.y,
  fwd: [-30, 30, 0.5, "앞으로 이동 (칸) — 옆 그림에서 얼굴 쪽"],
  ground: [0, 1, 0.05, "바닥 붙이기 — 1 이면 가장 낮은 발이 바닥에 닿게 몸을 올리고 내림 (공중은 0)"],
};

const st = {
  info: null, view: "front", scenes: {}, key: null, common: false,
  raw: null,          // 파일 내용 (편집 대상)
  anim: null,         // 계산에 쓰는 동작 (mirror_of 를 풀고 기본값을 채운 것)
  readOnly: false, frame: 0, bone: "upper_arm_r",
  playing: false, dirty: false, drag: null,
  undo: [], redo: [], committed: null, inGesture: false,   // 되돌리기
};
const $ = (id) => document.getElementById(id);
const stage = $("stage"), sctx = stage.getContext("2d");

// ---------------------------------------------------------------- 행렬

const mat = {
  mul(a, b) {
    const r = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
      r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    return r;
  },
  rot(deg) {
    const c = Math.cos(deg * Math.PI / 180), s = Math.sin(deg * Math.PI / 180);
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]];
  },
  move(x, y) { return [[1, 0, x], [0, 1, y], [0, 0, 1]]; },
  diag(a) { return [[a, 0, 0], [0, 1, 0], [0, 0, 1]]; },
  apply(m, p) { return [m[0][0] * p[0] + m[0][1] * p[1] + m[0][2], m[1][0] * p[0] + m[1][1] * p[1] + m[1][2]]; },
};

// animate.bone_matrices 와 같다
function boneMatrices(sc, pose) {
  const J = sc.joints, u = sc.unit;
  const root = pose.root || {};
  const cx = (J.pelvis[0] + J.neck[0]) / 2, cy = (J.pelvis[1] + J.neck[1]) / 2;
  const rootM = [mat.move((root.x || 0) * u, (root.y || 0) * u), mat.move(cx, cy), mat.rot(root.rot || 0), mat.move(-cx, -cy)]
    .reduce(mat.mul);
  const mats = {}, chain = {};   // mats: 그릴 행렬, chain: 자식이 이어받을 행렬 (tilt·turn 모양 줄임이 빠짐)
  const rad = Math.PI / 180;
  for (const [name, parent, j0, j1] of sc.bones) {
    const p = pose[name] || {};
    const s = J[j0], e = J[j1];
    const a = Math.atan2(e[1] - s[1], e[0] - s[0]) / rad;
    let base = rootM;
    if (parent) {
      const d = mat.apply(mats[parent], s), c = mat.apply(chain[parent], s);
      base = mat.mul(mat.move(d[0] - c[0], d[1] - c[1]), chain[parent]);
    }
    const stretch = [mat.rot(a), mat.diag(p.len === undefined ? 1 : p.len), mat.rot(-a)].reduce(mat.mul);
    const foreshort = [mat.rot(a), [[Math.cos((p.tilt || 0) * rad), 0, 0], [0, Math.cos((p.turn || 0) * rad), 0], [0, 0, 1]],
      mat.rot(-a)].reduce(mat.mul);
    const head = [base, mat.move((p.x || 0) * u, (p.y || 0) * u), mat.move(s[0], s[1]), mat.rot(p.rot || 0), stretch].reduce(mat.mul);
    chain[name] = mat.mul(head, mat.move(-s[0], -s[1]));
    mats[name] = [head, foreshort, mat.move(-s[0], -s[1])].reduce(mat.mul);
  }
  return mats;
}

// ---------------------------------------------------------------- 열쇠 곡선

function sampleEase(keys, t) {           // animate.sample (예전 방식: 열쇠마다 감속·가속)
  keys = [...keys].sort((a, b) => a[0] - b[0]);
  if (t <= keys[0][0]) return keys[0][1];
  for (let i = 0; i + 1 < keys.length; i++) {
    const [t0, v0] = keys[i], [t1, v1] = keys[i + 1];
    if (t <= t1) {
      let u = t1 > t0 ? (t - t0) / (t1 - t0) : 1;
      u = u * u * (3 - 2 * u);
      return v0 + (v1 - v0) * u;
    }
  }
  return keys[keys.length - 1][1];
}

function sampleSpline(keys, t, loop) {   // animate.sample_spline
  keys = [...keys].sort((a, b) => a[0] - b[0]);
  const n = keys.length;
  if (n === 1 || (t <= keys[0][0] && !loop)) return keys[0][1];
  if (t >= keys[n - 1][0] && !loop) return keys[n - 1][1];
  const ts = keys.map((k) => k[0]), vs = keys.map((k) => k[1]);
  const neighbor = (i) => {
    if (i >= 0 && i < n) return [ts[i], vs[i]];
    if (!loop) return null;
    const same = n > 1 && Math.abs(ts[0] - (ts[n - 1] - 1)) < 1e-9;
    if (i < 0) { const j = n + i - (same ? 1 : 0); return [ts[j] - 1, vs[j]]; }
    const j = i - n + (same ? 1 : 0); return [ts[j] + 1, vs[j]];
  };
  const tangent = (i) => {
    if (keys[i].length > 2 && keys[i][2] === "ease") return 0;
    const a = neighbor(i - 1), b = neighbor(i + 1);
    if (!a || !b) return 0;
    return b[0] !== a[0] ? (b[1] - a[1]) / (b[0] - a[0]) : 0;
  };
  let t0, v0, t1, v1, m0, m1;
  if (loop && (t < ts[0] || t > ts[n - 1])) {
    t0 = ts[n - 1] - 1; v0 = vs[n - 1]; t1 = ts[0]; v1 = vs[0];
    if (t > ts[n - 1]) { t0 = ts[n - 1]; t1 = ts[0] + 1; }
    m0 = tangent(n - 1); m1 = tangent(0);
  } else {
    let i0 = 0;
    for (let i = 0; i < n; i++) if (ts[i] <= t) i0 = i;
    if (i0 === n - 1) return vs[n - 1];
    t0 = ts[i0]; v0 = vs[i0]; t1 = ts[i0 + 1]; v1 = vs[i0 + 1];
    m0 = tangent(i0); m1 = tangent(i0 + 1);
  }
  const h = t1 - t0;
  if (h <= 0) return v1;
  const u = (t - t0) / h, u2 = u * u, u3 = u2 * u;
  return (2 * u3 - 3 * u2 + 1) * v0 + (u3 - 2 * u2 + u) * h * m0 + (-2 * u3 + 3 * u2) * v1 + (u3 - u2) * h * m1;
}

function sampleChannel(anim, keys, t) {
  return anim.interp === "spline" ? sampleSpline(keys, t, !!anim.loop) : sampleEase(keys, t);
}

function poseAt(anim, base, t) {          // animate.pose_at
  const pose = {};
  for (const [b, ch] of Object.entries(base)) pose[b] = { ...ch };
  for (const [bone, chans] of Object.entries(anim.keys || {})) {
    const p = pose[bone] || (pose[bone] = {});
    for (const [ch, keys] of Object.entries(chans)) {
      if (!keys.length) continue;
      const v = sampleChannel(anim, keys, t);
      p[ch] = ch === "len" ? (p[ch] === undefined ? 1 : p[ch]) * v : (p[ch] || 0) + v;
    }
  }
  return pose;
}

function mirrorKeys(keys) {               // animate.mirror_keys
  const out = {};
  for (const [bone, chans] of Object.entries(keys)) {
    out[bone] = {};
    for (const [ch, vals] of Object.entries(chans))
      out[bone][ch] = vals.map((k) => [k[0], (ch === "rot" || ch === "x") ? -k[1] : k[1], ...k.slice(2)]);
  }
  return out;
}

// ---------------------------------------------------------------- 공통 동작 → 방향별 자세 (common_anim.py 와 같다)

const VIEW_AXES = {
  front: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], back: [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
  left: [[0, 0, -1], [0, 1, 0], [1, 0, 0]], right: [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
};
const UP_KINDS = ["torso", "head"];
const TWIST_AXIS = { torso: [0, -1, 0], head: [0, -1, 0], foot: [0, 0, 1] };
const Z_SCALE = 8;
const DEG = Math.PI / 180;
const rot3 = {
  x(d) { const c = Math.cos(d * DEG), s = Math.sin(d * DEG); return [[1, 0, 0], [0, c, -s], [0, s, c]]; },
  y(d) { const c = Math.cos(d * DEG), s = Math.sin(d * DEG); return [[c, 0, s], [0, 1, 0], [-s, 0, c]]; },
  z(d) { const c = Math.cos(d * DEG), s = Math.sin(d * DEG); return [[c, -s, 0], [s, c, 0], [0, 0, 1]]; },
  axis(u, d) {
    const k = [[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]], kk = mat.mul(k, k);
    const s = Math.sin(d * DEG), c1 = 1 - Math.cos(d * DEG);
    return [0, 1, 2].map((i) => [0, 1, 2].map((j) => (i === j ? 1 : 0) + s * k[i][j] + c1 * kk[i][j]));
  },
};
const transpose = (m) => m[0].map((_, j) => m.map((r) => r[j]));
const mulVec = (m, v) => m.map((r) => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
const rot2 = (d) => { const c = Math.cos(d * DEG), s = Math.sin(d * DEG); return [[c, -s], [s, c]]; };
const mul2 = (a, b) => [[a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
                        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]]];
const top2 = (m) => [[m[0][0], m[0][1]], [m[1][0], m[1][1]]];
const kindOf = (name) => /_[lr]$/.test(name) ? name.slice(0, -2) : name;
const wrap180 = (d) => ((d + 180) % 360 + 360) % 360 - 180;

function localRotation(name, ch) {
  const m = name.endsWith("_l") ? -1 : 1, up = UP_KINDS.includes(kindOf(name)) ? -1 : 1;
  const spread = m * (ch.spread || 0), swing = up * (ch.swing || 0), twist = m * (ch.twist || 0);
  const axis = TWIST_AXIS[kindOf(name)] || [0, 1, 0];
  return [[rot3.x(swing), rot3.z(spread), rot3.axis(axis, twist)].reduce(mat.mul), spread, swing];
}

function inPlane(view, spread, swing) {
  return { front: spread, back: -spread, left: swing, right: -swing }[view];
}

function splitCard(n, near) {
  const a = n[0][0] ** 2 + n[1][1] ** 2, b = n[1][0] ** 2 + n[0][1] ** 2;
  const c = n[0][0] * n[1][0] - n[1][1] * n[0][1];
  let r = 0.5 * Math.atan2(2 * c, a - b);
  const cs = Math.cos(r), sn = Math.sin(r);
  let along = cs * n[0][0] + sn * n[1][0], across = -sn * n[0][1] + cs * n[1][1];
  if (across < 0) { r += Math.PI; along = -along; across = -across; }
  r = r / DEG;
  return [near + wrap180(r - near), along, across];
}

function neutralOf(sc) {
  if (!sc._neutral) {
    const mats = boneMatrices(sc, sc.base_pose);
    sc._neutral = {};
    for (const [name, , j0, j1] of sc.bones)
      sc._neutral[name] = [mat.apply(mats[name], sc.joints[j0]), mat.apply(mats[name], sc.joints[j1])];
  }
  return sc._neutral;
}

function sampleCommon(anim, t) {
  const vals = {};
  for (const [bone, chans] of Object.entries(anim.common || {})) {
    vals[bone] = {};
    for (const [ch, keys] of Object.entries(chans)) if (keys.length) vals[bone][ch] = sampleChannel(anim, keys, t);
  }
  return vals;
}

function chainOf(sc, bone) {               // common_anim.chain_of
  const parents = {};
  for (const [name, parent] of sc.bones) parents[name] = parent;
  const out = [];
  while (bone) { out.unshift(bone); bone = parents[bone]; }
  return out;
}

function turned(vals, chain) {             // common_anim.turned
  const rc = vals.root || {};
  let swing = -(rc.swing || 0), spread = rc.spread || 0;
  for (const name of chain) {
    const ch = vals[name] || {};
    swing += (UP_KINDS.includes(kindOf(name)) ? -1 : 1) * (ch.swing || 0);
    spread += (name.endsWith("_l") ? -1 : 1) * (ch.spread || 0);
  }
  return [swing, spread];
}

function applyFollow(anim, sc, t, vals) {  // common_anim.apply_follow
  const names = new Set(sc.bones.map((b) => b[0]));
  const n = anim.frames || 8, step = anim.loop ? 1 / n : 1 / Math.max(1, n - 1);
  for (const [bone, f] of Object.entries(anim.follow || {})) {
    if (!names.has(bone) || !names.has(f.of)) continue;
    const chain = chainOf(sc, f.of);
    let t0 = t - (f.lag === undefined ? 2 : f.lag) * step;
    t0 = anim.loop ? ((t0 % 1) + 1) % 1 : Math.max(0, t0);
    const now = turned(vals, chain), before = turned(sampleCommon(anim, t0), chain);
    const gain = f.gain === undefined ? 1 : f.gain;
    const ch = vals[bone] || (vals[bone] = {});
    ch.swing = (ch.swing || 0) + gain * (before[0] - now[0]) * (UP_KINDS.includes(kindOf(bone)) ? -1 : 1);
    ch.spread = (ch.spread || 0) + gain * (before[1] - now[1]) * (bone.endsWith("_l") ? -1 : 1);
  }
}

// animate.BONE_NAMES — 바닥 붙이기는 칼·망토 같은 추가 뼈를 빼고 이 뼈들의 가장 낮은 곳으로 잰다
const BODY_BONES = ["torso", "head", "upper_arm_l", "forearm_l", "hand_l", "upper_arm_r", "forearm_r", "hand_r",
  "thigh_l", "shin_l", "foot_l", "thigh_r", "shin_r", "foot_r"];

function groundShift(sc, pose) {           // common_anim.ground_shift
  const rest = neutralOf(sc), feet = BODY_BONES.filter((b) => rest[b]);
  const mats = boneMatrices(sc, pose);
  let lowRest = -Infinity, lowNow = -Infinity;
  for (const [name, , j0, j1] of sc.bones) {
    if (!feet.includes(name)) continue;
    lowRest = Math.max(lowRest, rest[name][0][1], rest[name][1][1]);
    for (const j of [j0, j1]) lowNow = Math.max(lowNow, mat.apply(mats[name], sc.joints[j])[1]);
  }
  return (lowRest - lowNow) / sc.unit;
}

function convertCommon(anim, sc, t) {
  const view = sc.view, va = VIEW_AXES[view], vaT = transpose(va);
  const vals = sampleCommon(anim, t);
  applyFollow(anim, sc, t, vals);
  const rest = neutralOf(sc), J = sc.joints;
  const center = [(J.pelvis[0] + J.neck[0]) / 2, (J.pelvis[1] + J.neck[1]) / 2];
  const torsoLen = Math.hypot(J.neck[0] - J.pelvis[0], J.neck[1] - J.pelvis[1]);
  const rc = vals.root || {};
  const gRoot = [rot3.z(rc.spread || 0), rot3.x(-(rc.swing || 0)), rot3.y(rc.twist || 0)].reduce(mat.mul);
  const m3Root = [va, gRoot, vaT].reduce(mat.mul);
  const [rootR] = splitCard(top2(m3Root), inPlane(view, rc.spread || 0, -(rc.swing || 0)));
  const pos = mulVec(va, [rc.x || 0, rc.y || 0, rc.fwd || 0]);

  const pose = {};
  for (const [b, ch] of Object.entries(sc.base_pose)) pose[b] = { ...ch };
  const root = pose.root || (pose.root = {});
  root.x = (root.x || 0) + pos[0];
  root.y = (root.y || 0) + pos[1];
  root.rot = (root.rot || 0) + rootR;

  const g = {}, near = {}, rotG = {}, m3s = {}, starts = {}, zTotal = {};
  for (const [name, parent] of sc.bones) {
    const ch = vals[name] || {};
    const [local, spread, swing] = localRotation(name, ch);
    g[name] = mat.mul(parent ? g[parent] : gRoot, local);
    const m3 = [va, g[name], vaT].reduce(mat.mul);
    m3s[name] = m3;
    near[name] = (parent ? near[parent] : rootR) + inPlane(view, spread, swing);
    const [s0, e0] = rest[name];
    const ang = Math.atan2(e0[1] - s0[1], e0[0] - s0[0]) / DEG;
    const rotN = mul2(mul2(rot2(-ang), top2(m3)), rot2(ang));
    let along, across;
    [rotG[name], along, across] = splitCard(rotN, near[name]);
    if (parent) {
      const ps = rest[parent][0], d = mulVec(m3s[parent], [s0[0] - ps[0], s0[1] - ps[1], 0]);
      starts[name] = starts[parent].map((v, i) => v + d[i]);
    } else {
      const d = mulVec(m3Root, [s0[0] - center[0], s0[1] - center[1], 0]);
      starts[name] = [center[0] + d[0], center[1] + d[1], d[2]];
    }
    const half = mulVec(m3, [(e0[0] - s0[0]) / 2, (e0[1] - s0[1]) / 2, 0]);
    zTotal[name] = Z_SCALE * (starts[name][2] + half[2]) / torsoLen;

    const p = pose[name] || (pose[name] = {});
    p.rot = (p.rot || 0) + rotG[name] - (parent ? rotG[parent] : rootR);
    const tilt = Math.acos(Math.max(-1, Math.min(1, along))) / DEG;
    const turn = Math.acos(Math.max(-1, Math.min(1, across))) / DEG;
    if (tilt > 1e-6) p.tilt = (p.tilt || 0) + tilt;
    if (turn > 1e-6) p.turn = (p.turn || 0) + turn;
    const z = zTotal[name] - (parent ? zTotal[parent] : 0);
    if (Math.abs(z) > 1e-6) p.z = (p.z || 0) + z;
    if (ch.len !== undefined) p.len = (p.len === undefined ? 1 : p.len) * ch.len;
    if (!parent) {
      const off = [s0[0] - center[0], s0[1] - center[1]];
      const want = top2(m3Root), got = rot2(rootR), back = rot2(-rootR);
      const diff = [0, 1].map((i) => (want[i][0] - got[i][0]) * off[0] + (want[i][1] - got[i][1]) * off[1]);
      const dx = (back[0][0] * diff[0] + back[0][1] * diff[1]) / sc.unit;
      const dy = (back[1][0] * diff[0] + back[1][1] * diff[1]) / sc.unit;
      if (Math.abs(dx) + Math.abs(dy) > 1e-6) { p.x = (p.x || 0) + dx; p.y = (p.y || 0) + dy; }
    }
  }
  if (rc.ground) root.y += rc.ground * groundShift(sc, pose);
  return pose;
}

function viewAnim(data, view) {           // common_anim.view_anim
  const extra = (data.views || {})[view] || {};
  const out = {};
  for (const [k, v] of Object.entries(data)) if (k !== "views" && k !== "keys") out[k] = v;
  for (const [k, v] of Object.entries(extra)) if (k !== "keys") out[k] = v;
  out.common = data.keys || {};
  out.keys = extra.keys || {};
  return out;
}

function animPose(anim, sc, t) {          // animate.anim_pose
  return poseAt(anim, anim.common ? convertCommon(anim, sc, t) : sc.base_pose, t);
}

function frameSetup(anim, view, t) {      // animate.frame_setup → {scene, pose, swap}
  for (const s of anim.view_swap || []) {
    if (s.from <= t && t < s.to && st.scenes[s.view]) {
      const sc = st.scenes[s.view];
      const keys = s.mirror ? mirrorKeys(anim.keys) : anim.keys;
      const pose = animPose({ ...anim, keys }, sc, t);
      for (const [bone, chans] of Object.entries(s.add || {})) {
        const p = pose[bone] || (pose[bone] = {});
        for (const [ch, v] of Object.entries(chans))
          p[ch] = ch === "len" ? (p[ch] === undefined ? 1 : p[ch]) * v : (p[ch] || 0) + v;
      }
      return { sc, pose, swap: s };
    }
  }
  const sc = st.scenes[view];
  return { sc, pose: animPose(anim, sc, t), swap: null };
}

function frameTimes(anim) {
  const n = anim.frames;
  return [...Array(n).keys()].map((i) => anim.loop ? i / n : i / Math.max(1, n - 1));
}

function frameDurations(anim) {
  const n = anim.frames, ms = anim.frame_ms;
  if (ms === undefined || ms === null) return Array(n).fill(Math.round(1000 / anim.fps));
  if (typeof ms === "number") return Array(n).fill(ms);
  return ms;
}

// ---------------------------------------------------------------- 그리기

function zOffsets(sc, pose) {             // animate.z_offsets
  const out = {};
  for (const [name, parent] of sc.bones) out[name] = ((pose[name] || {}).z || 0) + (parent ? out[parent] : 0);
  return out;
}

function drawScene(ctx, sc, pose, scale, alpha = 1) {
  const mats = boneMatrices(sc, pose);
  const z = zOffsets(sc, pose);
  // 쌓는 순서 = depth + z (같으면 층 번호, 그다음 원래 순서) — parts.PartModel.render 와 같다
  const order = sc.pieces.map((p, i) => i).sort((i, j) => {
    const a = sc.pieces[i], b = sc.pieces[j];
    return (a.depth + (z[a.bone] || 0)) - (b.depth + (z[b.bone] || 0)) || a.layer - b.layer || i - j;
  });
  ctx.save();
  ctx.globalAlpha = alpha;
  for (const p of order.map((i) => sc.pieces[i])) {
    if (!sc.defaultLayers.has(p.layer) || !p.img.complete) continue;
    // OpenCV(warpAffine)는 화소 가운데를, 캔버스는 화소 모서리를 기준으로 그린다. 반 화소 옮겨 맞춘다.
    const m = [mat.move(0.5, 0.5), mats[p.bone], p.rest, mat.move(-0.5, -0.5)].reduce(mat.mul);
    ctx.setTransform(scale * m[0][0], scale * m[1][0], scale * m[0][1], scale * m[1][1], scale * m[0][2], scale * m[1][2]);
    ctx.drawImage(p.img, 0, 0);
  }
  ctx.restore();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  return mats;
}

function segments(sc, mats) {
  return sc.bones.map(([name, , j0, j1]) => ({
    name, a: mat.apply(mats[name], sc.joints[j0]), b: mat.apply(mats[name], sc.joints[j1]),
  }));
}

function render() {
  if (!st.anim) return;
  const scale = stage.width / st.scenes[st.view].size;
  sctx.setTransform(1, 0, 0, 1, 0, 0);
  sctx.fillStyle = "#2b2f38";
  sctx.fillRect(0, 0, stage.width, stage.height);
  const times = frameTimes(st.anim);
  const sc0 = st.scenes[st.view];
  if ($("showGround").checked) {
    sctx.strokeStyle = "#475063";
    sctx.beginPath(); sctx.moveTo(0, sc0.foot_line * scale); sctx.lineTo(stage.width, sc0.foot_line * scale); sctx.stroke();
    const cell = sc0.size / sc0.canvas * scale;  // 최종 1칸 눈금 (8칸마다)
    sctx.fillStyle = "#353b47";
    for (let x = 0; x < stage.width; x += cell * 8) sctx.fillRect(x, 0, 1, stage.height);
  }
  if ($("showOnion").checked) {
    for (const [d, a] of [[-1, 0.25], [1, 0.18]]) {
      const i = st.frame + d;
      const j = st.anim.loop ? (i + st.anim.frames) % st.anim.frames : i;
      if (j < 0 || j >= st.anim.frames || j === st.frame) continue;
      const f = frameSetup(st.anim, st.view, times[j]);
      drawScene(sctx, f.sc, f.pose, scale, a);
    }
  }
  const f = frameSetup(st.anim, st.view, times[st.frame]);
  const mats = drawScene(sctx, f.sc, f.pose, scale);
  st.cur = { ...f, mats, scale };
  if ($("showBones").checked) {
    for (const s of segments(f.sc, mats)) {
      const sel = s.name === st.bone;
      sctx.strokeStyle = sel ? "#4fb3ff" : "rgba(255,255,255,0.35)";
      sctx.lineWidth = sel ? 3 : 1.5;
      sctx.beginPath(); sctx.moveTo(s.a[0] * scale, s.a[1] * scale); sctx.lineTo(s.b[0] * scale, s.b[1] * scale); sctx.stroke();
      sctx.fillStyle = sel ? "#4fb3ff" : "rgba(255,255,255,0.5)";
      sctx.beginPath(); sctx.arc(s.a[0] * scale, s.a[1] * scale, sel ? 4 : 2.5, 0, 7); sctx.fill();
    }
  }
  renderPixel(f);
  renderViews4();
  renderPanel();
}

function renderViews4() {    // 공통 동작: 지금 프레임을 네 방향으로 작게 보여 준다 (누르면 그 방향으로 바꿈)
  const box = $("views4");
  box.style.display = st.common ? "" : "none";
  $("stageWrap").classList.toggle("with4", st.common);
  if (!st.common) return;
  if (box.dataset.views !== st.info.views.join()) {
    box.dataset.views = st.info.views.join();
    box.innerHTML = st.info.views.map((v) => `<div class="v4" data-view="${v}"><canvas id="v4_${v}" width="120" height="120"></canvas><span>${v}</span></div>`).join("");
    for (const el of box.querySelectorAll(".v4")) el.onclick = () => selectView(el.dataset.view);
  }
  const t = t_now();
  for (const v of st.info.views) {
    const c = $(`v4_${v}`), ctx = c.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#2b2f38"; ctx.fillRect(0, 0, c.width, c.height);
    const f = frameSetup(withDefaults(viewAnim(st.raw, v)), v, t);
    drawScene(ctx, f.sc, f.pose, c.width / f.sc.size);
    c.parentElement.classList.toggle("cur", v === st.view);
  }
}

const offscreen = document.createElement("canvas");
function renderPixel(f) {
  // 그리기 크기로 그린 뒤 칸마다 가운데 픽셀만 골라 64칸(=canvas칸)으로 줄인 대략 모습
  const sc = f.sc, size = sc.size, k = sc.k, n = sc.canvas;
  offscreen.width = offscreen.height = size;
  const o = offscreen.getContext("2d");
  o.clearRect(0, 0, size, size);
  drawScene(o, sc, f.pose, 1);
  const src = o.getImageData(0, 0, size, size).data;
  const pc = $("pixel"), p = pc.getContext("2d");
  const img = p.createImageData(n, n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const si = ((y * k + (k >> 1)) * size + (x * k + (k >> 1))) * 4, di = (y * n + x) * 4;
    if (src[si + 3] >= 128) { img.data[di] = src[si]; img.data[di + 1] = src[si + 1]; img.data[di + 2] = src[si + 2]; img.data[di + 3] = 255; }
  }
  const tmp = document.createElement("canvas");
  tmp.width = tmp.height = n;
  tmp.getContext("2d").putImageData(img, 0, 0);
  p.imageSmoothingEnabled = false;
  p.fillStyle = "#2b2f38"; p.fillRect(0, 0, pc.width, pc.height);
  p.drawImage(tmp, 0, 0, pc.width, pc.height);
}

// ---------------------------------------------------------------- 열쇠 편집

function keysOf(bone, ch, create) {
  const K = st.raw.keys || (st.raw.keys = {});
  if (!K[bone]) { if (!create) return null; K[bone] = {}; }
  if (!K[bone][ch]) { if (!create) return null; K[bone][ch] = []; }
  return K[bone][ch];
}

function chansOf(bone) { return st.common ? (bone === "root" ? ROOT_COMMON : BONE_COMMON) : CHANNELS; }
function uiOf(ch) { return (st.common ? COMMON_UI : CHANNEL_UI)[ch]; }

function valueAt(bone, ch, t) {
  const keys = ((st.common ? st.anim.common : st.anim.keys)[bone] || {})[ch];
  return keys && keys.length ? sampleChannel(st.anim, keys, t) : DEFAULT[ch];
}

const near = (a, b) => Math.abs(a - b) < 1e-4;
const round = (v, d = 2) => Math.round(v * 10 ** d) / 10 ** d;

function setKey(bone, ch, t, v) {
  if (st.readOnly) return;
  const keys = keysOf(bone, ch, true);
  const k = keys.find((k) => near(k[0], t));
  if (k) k[1] = round(v, ch === "len" ? 3 : 2);
  else { keys.push([round(t, 4), round(v, ch === "len" ? 3 : 2)]); keys.sort((a, b) => a[0] - b[0]); }
  changed();
}

function hasKey(bone, t) {
  const chans = (st.raw.keys || {})[bone] || {};
  return Object.values(chans).some((ks) => ks.some((k) => near(k[0], t)));
}

function keyIsEase(bone, t) {
  const chans = (st.raw.keys || {})[bone] || {};
  return Object.values(chans).some((ks) => ks.some((k) => near(k[0], t) && k[2] === "ease"));
}

function anyKey(t) {
  return Object.keys(st.raw.keys || {}).some((b) => hasKey(b, t));
}

function addKeyAll(bone, t) {           // 지금 보이는 값으로 열쇠: 회전은 늘, 나머지는 기본값이 아니거나 이미 열쇠가 있는 채널
  const always = st.common ? ["swing", "spread"] : ["rot"];
  for (const ch of chansOf(bone)) {
    const v = valueAt(bone, ch, t);
    if (always.includes(ch) || v !== DEFAULT[ch] || keysOf(bone, ch, false)) setKey(bone, ch, t, v);
  }
}

function deleteKeys(bone, t) {
  if (st.readOnly) return;
  const chans = (st.raw.keys || {})[bone];
  if (!chans) return;
  for (const ch of Object.keys(chans)) {
    chans[ch] = chans[ch].filter((k) => !near(k[0], t));
    if (!chans[ch].length) delete chans[ch];
  }
  if (!Object.keys(chans).length) delete st.raw.keys[bone];
  changed();
}

function setEase(bone, t, on) {
  if (st.readOnly) return;
  const chans = (st.raw.keys || {})[bone] || {};
  for (const ks of Object.values(chans)) for (const k of ks)
    if (near(k[0], t)) { if (on) k[2] = "ease"; else k.length = 2; }
  changed();
}

// ---------------------------------------------------------------- 되돌리기 / 다시 하기
// 바뀌기 전 상태(committed)를 쌓아 둔다. 끌기·슬라이더처럼 이어지는 조작은 끝날 때 한 번만 쌓는다.

const UNDO_LIMIT = 200;
const snap = () => JSON.stringify(st.raw);

function pushUndo(before) {
  st.undo.push(before);
  if (st.undo.length > UNDO_LIMIT) st.undo.shift();
  st.redo = [];
}

function beginGesture() { st.inGesture = true; }

function endGesture() {
  if (!st.inGesture) return;
  st.inGesture = false;
  const now = snap();
  if (now !== st.committed) { pushUndo(st.committed); st.committed = now; updateUndoButtons(); }
}

function restore(state) {
  st.raw = JSON.parse(state);
  st.committed = state;
  st.dirty = true;
  resolveAnim();
  st.frame = Math.min(st.frame, st.anim.frames - 1);
  renderSettings(); setStatus(); render(); renderTimeline(); updateUndoButtons();
}

function undo() {
  if (st.inGesture || !st.undo.length) return;
  st.redo.push(snap());
  restore(st.undo.pop());
}

function redo() {
  if (st.inGesture || !st.redo.length) return;
  st.undo.push(snap());
  restore(st.redo.pop());
}

function updateUndoButtons() {
  $("undoBtn").disabled = !st.undo.length;
  $("redoBtn").disabled = !st.redo.length;
}

function changed() {
  if (!st.inGesture) {
    const now = snap();
    if (now !== st.committed) { pushUndo(st.committed); st.committed = now; }
  }
  st.dirty = true;
  resolveAnim();
  setStatus();
  render();
  renderTimeline();
  updateUndoButtons();
}

function withDefaults(a) {               // animate.load_anim 처럼 기본값을 채운다
  a = JSON.parse(JSON.stringify(a));
  a.frames = a.frames || 8; a.fps = a.fps || 10;
  if (a.loop === undefined) a.loop = true;
  if (!a.interp) a.interp = "ease";
  a.keys = a.keys || {};
  return a;
}

function resolveAnim() {
  // 공통 동작은 지금 방향으로 바꾸고, mirror_of 는 서버가 풀어 준 값을 쓴다
  st.anim = withDefaults(st.common ? viewAnim(st.raw, st.view) : st.readOnly ? st.resolved : st.raw);
}

// ---------------------------------------------------------------- 패널

function renderPanel() {
  const t = frameTimes(st.anim)[st.frame];
  const sign = st.cur && st.cur.swap && st.cur.swap.mirror ? -1 : 1;
  const rows = $("chanRows");
  const rowsKey = `${st.common}|${st.bone}`;
  if (rows.dataset.bone !== rowsKey) {
    rows.dataset.bone = rowsKey;
    rows.innerHTML = "";
    for (const ch of chansOf(st.bone)) {
      const r = document.createElement("div");
      r.className = "row";
      const range = uiOf(ch);
      r.title = range[3];
      r.innerHTML = `<label>${ch}</label><span class="keydot" id="kd_${ch}"></span>
        <input type="range" id="rg_${ch}" min="${range[0]}" max="${range[1]}" step="${range[2]}">
        <input type="number" id="nb_${ch}" step="${range[2]}">`;
      rows.appendChild(r);
      const apply = (v) => { if (!isNaN(v)) setKey(st.bone, ch, t_now(), v); };
      const rg = r.querySelector(`#rg_${ch}`);
      rg.addEventListener("pointerdown", beginGesture);
      rg.addEventListener("input", (e) => apply(parseFloat(e.target.value)));
      rg.addEventListener("change", endGesture);
      r.querySelector(`#nb_${ch}`).addEventListener("change", (e) => apply(parseFloat(e.target.value)));
    }
  }
  for (const ch of chansOf(st.bone)) {
    const v = valueAt(st.bone, ch, t);
    const ks = keysOf(st.bone, ch, false);
    $(`kd_${ch}`).classList.toggle("on", !!(ks && ks.some((k) => near(k[0], t))));
    if (document.activeElement !== $(`nb_${ch}`)) $(`nb_${ch}`).value = round(v, 3);
    if (document.activeElement !== $(`rg_${ch}`)) $(`rg_${ch}`).value = v;
    $(`rg_${ch}`).disabled = $(`nb_${ch}`).disabled = st.readOnly;
  }
  $("easeChk").checked = keyIsEase(st.bone, t);
  $("easeChk").disabled = st.readOnly || !hasKey(st.bone, t);
  $("boneSel").value = st.bone;
  const f = st.cur;
  $("chanHelp").style.display = st.common ? "none" : "";
  $("commonHelp").style.display = st.common ? "" : "none";
  $("swapInfo").textContent = f && f.swap ? `이 프레임은 ${f.swap.view} 그림으로 그립니다 (view_swap${f.swap.mirror ? ", 좌우 뒤집은 값" : ""}).` +
    (sign < 0 ? " 끌기 방향은 자동으로 뒤집어 저장합니다." : "") : "";
}

function t_now() { return frameTimes(st.anim)[st.frame]; }

function renderSettings() {
  $("framesIn").value = st.anim.frames;
  $("loopIn").checked = !!st.anim.loop;
  $("interpIn").value = st.anim.interp;
  const tr = st.anim.trail;
  $("trailIn").value = Array.isArray(tr) ? tr.join(", ") : tr === true ? "true" : "";
  for (const id of ["framesIn", "loopIn", "interpIn", "allMsIn", "allMsBtn", "trailIn"]) $(id).disabled = st.readOnly;
}

function renderTimeline() {
  const tl = $("timeline");
  tl.innerHTML = "";
  const times = frameTimes(st.anim), durs = frameDurations(st.anim);
  const trail = Array.isArray(st.anim.trail) ? st.anim.trail : [];
  times.forEach((t, i) => {
    const c = document.createElement("div");
    c.className = "cell" + (i === st.frame ? " cur" : "");
    c.dataset.i = i;
    const swap = (st.anim.view_swap || []).some((s) => s.from <= t && t < s.to);
    if (swap) c.classList.add("swap");
    if (trail.includes(i)) c.classList.add("trail");
    const dots = [];
    if (anyKey(t)) dots.push('<span class="dot any"></span>');
    if (hasKey(st.bone, t)) dots.push(`<span class="dot sel${keyIsEase(st.bone, t) ? " ease" : ""}"></span>`);
    c.innerHTML = `<div class="num">#${i}</div><div class="dots">${dots.join("")}</div>
      <input type="number" min="10" max="2000" value="${durs[i]}" title="이 프레임에 머무는 시간(ms)">`;
    c.addEventListener("click", (e) => { if (e.target.tagName !== "INPUT") goFrame(i); });
    const inp = c.querySelector("input");
    inp.disabled = st.readOnly;
    inp.addEventListener("change", (e) => {
      const ms = frameDurations(st.anim).slice();
      ms[i] = Math.max(10, parseInt(e.target.value) || ms[i]);
      st.raw.frame_ms = ms;
      changed();
    });
    tl.appendChild(c);
  });
  const slider = $("frameSlider");
  slider.max = st.anim.frames - 1;
  if (document.activeElement !== slider) slider.value = st.frame;
  const total = durs.reduce((a, b) => a + b, 0);
  $("timeInfo").textContent = `${st.anim.frames}프레임 · 한 번 ${total}ms · ${st.anim.loop ? "반복" : "한 번"} · 곡선: ${st.anim.interp}` +
    (st.anim.view_swap ? " · 보라 줄 = 다른 방향 그림" : "") + (trail.length ? " · 흰 줄 = 궤적" : "");
}

function goFrame(i) {
  i = Math.max(0, Math.min(st.anim.frames - 1, i));
  if (i === st.frame) return;
  st.frame = i;
  render(); renderTimeline();
}

// 끌어서 프레임 넘겨 보기: 아래 슬라이더, 또는 시간표 칸 위를 누른 채 좌우로 끌기
$("frameSlider").addEventListener("pointerdown", () => { if (st.playing) togglePlay(); });
$("frameSlider").addEventListener("input", (e) => goFrame(parseInt(e.target.value)));
let scrubbing = false;
$("timeline").addEventListener("pointerdown", (e) => {
  if (e.target.tagName === "INPUT" || !e.target.closest(".cell")) return;
  scrubbing = true;
  if (st.playing) togglePlay();
  e.preventDefault();
});
window.addEventListener("pointermove", (e) => {
  if (!scrubbing) return;
  const cell = document.elementFromPoint(e.clientX, e.clientY);
  const c = cell && cell.closest && cell.closest(".cell");
  if (c) goFrame(parseInt(c.dataset.i));
});
window.addEventListener("pointerup", () => { scrubbing = false; });

function setStatus(msg) {
  const s = $("status");
  s.textContent = msg || (st.readOnly ? "읽기 전용 (뒤집어 쓰는 동작)" : st.dirty ? "저장 안 됨" : "저장됨");
  s.className = st.dirty ? "dirty" : "";
}

// ---------------------------------------------------------------- 마우스

function pickBone(px, py) {
  if (!st.cur) return null;
  let best = null, bestD = 14;
  for (const s of segments(st.cur.sc, st.cur.mats)) {
    const ax = s.a[0] * st.cur.scale, ay = s.a[1] * st.cur.scale, bx = s.b[0] * st.cur.scale, by = s.b[1] * st.cur.scale;
    const dx = bx - ax, dy = by - ay, L = dx * dx + dy * dy;
    const u = L ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / L)) : 0;
    const d = Math.hypot(px - (ax + u * dx), py - (ay + u * dy));
    if (d < bestD) { bestD = d; best = s.name; }
  }
  return best;
}

function mousePos(e) {
  const r = stage.getBoundingClientRect();
  return [(e.clientX - r.left) * stage.width / r.width, (e.clientY - r.top) * stage.height / r.height];
}

// 공통 동작에서 끌기 → 채널과 부호. 화면 안에서 돌리기는 앞·뒤 그림이면 spread, 옆 그림이면 swing.
// 앞뒤로 기울이기(Alt)는 그 반대 채널이고, 위로 끌면 보는 사람 쪽으로 온다.
function commonDragChannel(view, bone, depth) {
  const side = view === "left" || view === "right", flip = view === "back" || view === "right" ? -1 : 1;
  if (bone === "root") return depth ? [side ? "spread" : "swing", flip] : [side ? "swing" : "spread", side ? -flip : flip];
  const m = bone.endsWith("_l") ? -1 : 1, up = UP_KINDS.includes(kindOf(bone)) ? -1 : 1;
  if (depth) return side ? ["spread", -up * m * flip] : ["swing", flip];
  return side ? ["swing", up * flip] : ["spread", m * flip];
}

stage.addEventListener("mousedown", (e) => {
  if (!st.anim) return;
  const [px, py] = mousePos(e);
  const t = t_now();
  const sign = st.cur.swap && st.cur.swap.mirror ? -1 : 1;
  beginGesture();   // 끌기 한 번 = 되돌리기 한 번
  if (e.shiftKey) {
    // 공통 동작: 앞·뒤 그림의 옆 이동은 x, 옆 그림은 fwd (앞 = 얼굴 쪽)
    const v = st.cur.sc.view, side = v === "left" || v === "right";
    const xch = st.common && side ? "fwd" : "x";
    const xsign = st.common ? ({ front: 1, back: -1, left: -1, right: 1 })[v] : sign;
    st.drag = { kind: "root", px, py, xch, x0: valueAt("root", xch, t), y0: valueAt("root", "y", t), sign: xsign };
    return;
  }
  const b = pickBone(px, py);
  if (b) { st.bone = b; }
  if (e.altKey) {   // Alt+위아래 끌기: 고른 뼈를 앞뒤로 기울이기 (위로 끌면 보는 쪽 +)
    e.preventDefault();
    const [ch, s] = st.common ? commonDragChannel(st.cur.sc.view, st.bone, true) : ["tilt", 1];
    st.drag = { kind: "tilt", py, ch, s, v0: valueAt(st.bone, ch, t) };
    render(); renderTimeline();
    return;
  }
  let pivot;
  if (st.bone === "root") {   // 몸 전체 회전의 중심 = 골반과 목의 중간
    const J = st.cur.sc.joints, c = [(J.pelvis[0] + J.neck[0]) / 2, (J.pelvis[1] + J.neck[1]) / 2];
    const p = mat.apply(st.cur.mats.torso, c);
    pivot = [p[0] * st.cur.scale, p[1] * st.cur.scale];
  } else {
    const seg = segments(st.cur.sc, st.cur.mats).find((s) => s.name === st.bone);
    pivot = [seg.a[0] * st.cur.scale, seg.a[1] * st.cur.scale];
  }
  const [ch, s] = st.common ? commonDragChannel(st.cur.sc.view, st.bone, false) : ["rot", sign];
  st.drag = { kind: "rot", pivot, last: Math.atan2(py - pivot[1], px - pivot[0]) * 180 / Math.PI,
              acc: 0, ch, v0: valueAt(st.bone, ch, t), sign: s };
  render(); renderTimeline();
});

window.addEventListener("mousemove", (e) => {
  if (!st.drag || st.readOnly) return;
  const [px, py] = mousePos(e);
  const t = t_now(), d = st.drag;
  if (d.kind === "root") {
    const u = st.cur.sc.unit * st.cur.scale;
    setKey("root", d.xch, t, d.x0 + d.sign * (px - d.px) / u);
    setKey("root", "y", t, d.y0 + (py - d.py) / u);
    return;
  }
  if (d.kind === "tilt") {
    const lim = d.ch === "tilt" ? 80 : 360;
    setKey(st.bone, d.ch, t, Math.max(-lim, Math.min(lim, d.v0 + d.s * (d.py - py) * 0.5)));
    return;
  }
  const a = Math.atan2(py - d.pivot[1], px - d.pivot[0]) * 180 / Math.PI;
  let delta = a - d.last;
  if (delta > 180) delta -= 360;
  if (delta < -180) delta += 360;
  d.last = a; d.acc += delta;
  setKey(st.bone, d.ch, t, d.v0 + d.sign * d.acc);
});
window.addEventListener("mouseup", () => { st.drag = null; endGesture(); });

// ---------------------------------------------------------------- 재생

let playTimer = null;
function playStep() {   // 지금 프레임을 그 프레임 시간(ms)만큼 보여 준 뒤 다음 프레임으로
  if (!st.playing) return;
  render(); renderTimeline();
  playTimer = setTimeout(() => {
    st.frame = (st.frame + 1) % st.anim.frames;
    playStep();
  }, frameDurations(st.anim)[st.frame]);
}
function togglePlay() {
  st.playing = !st.playing;
  $("playBtn").textContent = st.playing ? "■ 멈춤" : "▶ 재생";
  clearTimeout(playTimer);
  if (st.playing) playStep();
}

// ---------------------------------------------------------------- 불러오기·저장

async function api(path, opt) {
  const r = await fetch(path, opt);
  const j = await r.json();
  if (j.error) throw new Error(j.error);
  return j;
}

async function loadScene(view) {
  if (st.scenes[view]) return;
  const sc = await api(`/api/scene?view=${view}`);
  sc.defaultLayers = new Set(sc.default_layers);
  await Promise.all(sc.pieces.map((p) => new Promise((res) => {
    p.img = new Image(); p.img.onload = res; p.img.onerror = res; p.img.src = p.tex;
  })));
  st.scenes[view] = sc;
}

async function loadAnim(key) {
  if (st.dirty && !confirm("저장하지 않은 변경이 있습니다. 버리고 다른 동작을 열까요?")) {
    $("animSel").value = st.key; return;
  }
  const j = await api(`/api/anim?key=${key}`);
  st.key = key; st.raw = j.raw; st.resolved = j.resolved;
  st.common = key.startsWith("common/");
  st.undo = []; st.redo = []; st.committed = snap(); st.inGesture = false;
  st.readOnly = !st.common && "mirror_of" in j.raw;
  st.frame = 0; st.dirty = false;
  const swaps = (j.resolved.view_swap || []).map((s) => s.view);
  for (const v of swaps) await loadScene(v);
  resolveAnim();
  const banner = $("banner");
  if (st.readOnly) {
    banner.style.display = "block";
    banner.innerHTML = `이 동작은 <b>${j.raw.mirror_of}</b> 를 좌우로 뒤집어 씁니다. 그쪽을 고치면 여기도 바뀝니다.
      <br><button id="detachBtn">따로 편집하기 (뒤집은 값으로 분리)</button>`;
    $("detachBtn").onclick = () => {
      const r = JSON.parse(JSON.stringify(j.resolved));
      st.raw = r; st.readOnly = false; banner.style.display = "none"; changed();
    };
  } else if (!st.common && (st.info.anims.common || []).includes(key.split("/")[1])) {
    banner.style.display = "block";
    banner.textContent = `같은 이름의 공통 동작(common/${key.split("/")[1]})이 있어서, 픽셀로 만들 때는 이 파일 대신 공통 동작을 씁니다.`;
  } else banner.style.display = "none";
  $("animSel").value = key;
  renderSettings(); setStatus(); render(); renderTimeline();
}

async function save() {
  if (st.readOnly) return;
  const out = JSON.parse(JSON.stringify(st.raw));
  if (out.frame_ms && out.frame_ms.length !== out.frames) out.frame_ms = frameDurations({ ...st.anim, frame_ms: out.frame_ms }).slice(0, out.frames);
  await api(`/api/anim?key=${st.key}`, { method: "POST", body: JSON.stringify(out) });
  st.dirty = false; setStatus("저장됨");
}

async function renderPixels() {
  if (st.dirty) await save();
  setStatus("픽셀로 만드는 중…");
  $("renderLog").textContent = "만드는 중…";
  if (st.common) $("renderLog").textContent = "네 방향을 모두 만드는 중…";
  const r = await api(`/api/render?key=${st.key}&view=${st.view}`, { method: "POST" });
  const warns = r.log.split("\n").filter((l) => l.includes("주의") || l.includes("Error") || l.includes("Traceback"));
  $("renderLog").textContent = r.ok ? (warns.length ? warns.join("\n") : "경고 없음") : r.log;
  $("renderLog").style.color = r.ok && !warns.length ? "var(--dim)" : "var(--warn)";
  $("renderImg").src = `/api/output?path=${encodeURIComponent(r.frames)}&_=${Date.now()}`;
  setStatus();
}

async function init() {
  st.info = await api("/api/info");
  $("charName").textContent = st.info.character;
  const vs = $("viewSel");
  vs.innerHTML = st.info.views.map((v) => `<option>${v}</option>`).join("");
  for (const v of st.info.views) await loadScene(v);
  vs.onchange = () => selectView(vs.value);
  await selectView(st.info.views[0]);
  setStatus();
}

function fillAnimSel() {
  const common = st.info.anims.common || [], own = st.info.anims[st.view] || [];
  $("animSel").innerHTML =
    (common.length ? `<optgroup label="네 방향 공통">${common.map((n) => `<option value="common/${n}">${n}</option>`).join("")}</optgroup>` : "") +
    `<optgroup label="${st.view} 전용">${own.map((n) => `<option value="${st.view}/${n}">${n}</option>`).join("")}</optgroup>`;
  if (st.key) $("animSel").value = st.key;
}

async function selectView(view) {
  st.view = view;
  $("viewSel").value = view;
  fillAnimSel();
  $("boneSel").innerHTML = ["root", ...st.scenes[view].bones.map((b) => b[0])].map((b) => `<option>${b}</option>`).join("");
  if (!st.scenes[view].bones.some((b) => b[0] === st.bone)) st.bone = "torso";
  if (st.common) {   // 공통 동작은 그대로 두고 보는 방향만 바꾼다
    $("animSel").value = st.key;
    resolveAnim(); renderSettings(); render(); renderTimeline();
    return;
  }
  const names = st.info.anims[view] || [], common = st.info.anims.common || [];
  const cur = st.key && st.key.split("/")[1];
  const pick = names.includes(cur) ? cur : names.includes("attack") ? "attack" : names[0];
  if (!st.key && common.length) await loadAnim(`common/${common.includes("attack") ? "attack" : common[0]}`);
  else if (pick) await loadAnim(`${view}/${pick}`);
}

// ---------------------------------------------------------------- 버튼

$("animSel").onchange = (e) => loadAnim(e.target.value);
$("boneSel").onchange = (e) => { st.bone = e.target.value; render(); renderTimeline(); };
$("saveBtn").onclick = () => save().catch((e) => alert(e.message));
$("renderBtn").onclick = () => renderPixels().catch((e) => { alert(e.message); setStatus(); });
$("playBtn").onclick = togglePlay;
$("undoBtn").onclick = undo;
$("redoBtn").onclick = redo;
$("keyBtn").onclick = () => addKeyAll(st.bone, t_now());
$("delKeyBtn").onclick = () => deleteKeys(st.bone, t_now());
$("easeChk").onchange = (e) => setEase(st.bone, t_now(), e.target.checked);
for (const id of ["showBones", "showOnion", "showGround"]) $(id).onchange = render;
$("framesIn").onchange = (e) => {
  const n = Math.max(1, parseInt(e.target.value) || st.anim.frames);
  const ms = frameDurations(st.anim);
  st.raw.frames = n;
  st.raw.frame_ms = [...Array(n).keys()].map((i) => ms[Math.min(i, ms.length - 1)]);
  st.frame = Math.min(st.frame, n - 1);
  changed(); renderSettings();
};
$("loopIn").onchange = (e) => { st.raw.loop = e.target.checked; changed(); };
$("interpIn").onchange = (e) => { st.raw.interp = e.target.value; changed(); };
$("allMsBtn").onclick = () => {
  const v = parseInt($("allMsIn").value);
  if (v > 0) { st.raw.frame_ms = Array(st.anim.frames).fill(v); changed(); }
};
$("trailIn").onchange = (e) => {
  // 공통 동작이면 궤적은 방향마다 따로 (views.<방향>.trail)
  let target = st.raw;
  if (st.common) { const vs = st.raw.views || (st.raw.views = {}); target = vs[st.view] || (vs[st.view] = {}); }
  const s = e.target.value.trim();
  if (!s) delete target.trail;
  else if (s === "true") target.trail = true;
  else target.trail = s.split(/[ ,]+/).map(Number).filter((n) => !isNaN(n));
  if (st.common && !Object.keys(target).length) delete st.raw.views[st.view];
  changed();
};
$("newBtn").onclick = async () => {
  const name = prompt("새 동작 이름 (영문·숫자·_):");
  if (!name) return;
  const common = confirm("네 방향 공통 동작으로 만들까요?\n확인 = 공통 (한 번 움직이면 네 방향 모두 따라감)\n취소 = 지금 방향(" + st.view + ") 전용");
  const key = `${common ? "common" : st.view}/${name}`;
  if (st.dirty && !confirm("저장하지 않은 변경이 있습니다. 버리고 새 동작을 열까요?")) return;
  const data = { frames: 16, frame_ms: Array(16).fill(60), loop: false, interp: "spline", keys: {} };
  await api(`/api/anim?key=${key}`, { method: "POST", body: JSON.stringify(data) });
  st.info = await api("/api/info");
  st.dirty = false;
  fillAnimSel();
  await loadAnim(key);
};
window.addEventListener("keydown", (e) => {
  // Ctrl+S: 저장 (글자 칸에서 입력 중이어도 그 값을 먼저 반영하고 저장)
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
    e.preventDefault();
    if (document.activeElement && document.activeElement.blur) document.activeElement.blur();
    if (st.readOnly) setStatus("읽기 전용이라 저장하지 않았습니다");
    else save().catch((err) => alert(err.message));
    return;
  }
  // Ctrl+Z / Ctrl+Y (Ctrl+Shift+Z): 글자 칸에서 입력 중일 때는 브라우저 기본 동작에 맡긴다
  const typing = e.target.tagName === "INPUT" && (e.target.type === "number" || e.target.type === "text");
  if ((e.ctrlKey || e.metaKey) && !typing) {
    const k = e.key.toLowerCase();
    if (k === "z" && !e.shiftKey) { e.preventDefault(); undo(); return; }
    if (k === "y" || (k === "z" && e.shiftKey)) { e.preventDefault(); redo(); return; }
  }
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (!st.anim) return;
  if (e.key === "ArrowRight") { st.frame = (st.frame + 1) % st.anim.frames; render(); renderTimeline(); }
  else if (e.key === "ArrowLeft") { st.frame = (st.frame - 1 + st.anim.frames) % st.anim.frames; render(); renderTimeline(); }
  else if (e.key === " ") { e.preventDefault(); togglePlay(); }
  else if (e.key === "k" || e.key === "K") addKeyAll(st.bone, t_now());
  else if (e.key === "Delete") deleteKeys(st.bone, t_now());
  else return;
});
window.addEventListener("beforeunload", (e) => { if (st.dirty) { e.preventDefault(); e.returnValue = ""; } });

init().catch((e) => setStatus("오류: " + e.message));
