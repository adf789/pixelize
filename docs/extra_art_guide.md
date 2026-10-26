# 더 넣을 그림 안내 (대각선 방향 · 부위 여러 각도)

> 진행 상황 (warrior): 대각선 두 장과 칼 세 장(옆·45도·정면)을 넣어 연결했다. 주먹 정면 그림은 아직 쓰지 않는다.
> 원본 시트는 `input/warrior_frontleftside.png`, `input/worrior_backleftside.png`.

지금은 방향마다 평평한 그림 한 장을 오려 돌려서, 보는 사람 쪽을 향한 칼·팔이 납작하고 짧아진다.
아래 두 가지 그림을 더 넣으면 그 한계를 줄일 수 있다. 그림이 들어오면 도구가 쓰도록 연결한다.

---

## ① 대각선 방향 그림 (8방향용)

탑뷰 게임은 위·아래·좌·우에 대각선 4방향을 더한 8방향을 많이 쓴다.
대각선 2장만 뽑으면 나머지 2장은 좌우로 뒤집어 만든다 (`animate.bat mirror`).

| 파일 | 캐릭터가 보는 쪽 | 비고 |
| --- | --- | --- |
| `characters/<이름>/front_left.png` | 앞과 왼쪽 사이 (화면 왼쪽 아래) | 얼굴이 보이는 3/4 앞모습 |
| `characters/<이름>/back_left.png` | 뒤와 왼쪽 사이 (화면 왼쪽 위) | 등이 보이는 3/4 뒷모습 |
| (만들어 줌) `front_right.png`, `back_right.png` | 오른쪽 대각선 | 위 두 장을 뒤집어 만든다 |

### 지킬 것

- **지금 그림과 같은 캐릭터·같은 크기**: 키(머리 꼭대기~발끝)가 지금 그림과 같게. 지금 그림은 높이 850픽셀, 캐릭터가 거의 꽉 찬다.
- **배경**: 지금과 같은 연한 회색 단색 (RGB 211, 211, 212 근처) 또는 흰색. 그림자·바닥 없음.
- **자세**: 팔을 몸에서 20~30도 벌린 자세(A자). 팔이 몸통에 붙으면 조각을 나누기 어렵다.
  두 다리도 조금 벌려 서로 겹치지 않게.
- **칼은 빼고** 맨손으로 (칼은 지금처럼 따로 붙인다).
- 투구 덮개 그림(`helmet_*`)을 쓰려면 같은 방향의 투구 그림도 같은 자리에 맞춰 한 장씩.
- 전신, 발끝까지 잘리지 않게. 굵은 외곽선, 평평한 색 (README 의 "AI 그림을 뽑을 때 지킬 것"과 같다).

### 지시문 예시

지금 `front.png` 를 **참고 그림(레퍼런스)** 으로 함께 넣으면 옷·색이 덜 어긋난다.

```
same character as the reference image, three-quarter front view, facing front-left (45 degrees),
full body, A-pose with arms slightly away from the body, legs slightly apart, no weapon,
top-down RPG game character, chibi proportions, thick dark outline, flat colors, simple shapes,
no texture, no shadow, plain light gray background
```

뒷모습은 `three-quarter back view, facing back-left (45 degrees), back of the character visible` 로 바꾼다.

---

## ② 부위 여러 각도 그림

칼·주먹이 보는 사람 쪽을 향할 때, 짧게 누른 그림 대신 **그 각도에서 본 그림**으로 바꿔 끼운다.
도구는 그 부위가 보는 사람 쪽으로 얼마나 기울었는지 보고 가장 가까운 그림을 고른다
(대략 0~35도는 옆 그림, 35~65도는 45도 그림, 65도 넘으면 끝에서 본 그림).

| 파일 | 내용 | 그리는 법 |
| --- | --- | --- |
| `sword.png` (있음) | 칼을 옆에서 본 모습 | 지금 것 |
| `sword_45.png` | 칼끝이 보는 사람 쪽으로 45도 기운 모습 | 칼날이 원근으로 짧아지고 끝이 가까워 보인다 |
| `sword_tip.png` | 칼끝이 보는 사람을 똑바로 향한 모습 | 손잡이·코등이가 십자 모양으로 보이고 칼날은 거의 점 |
| `hand_fist_front.png` | 주먹을 보는 사람 쪽으로 내민 모습 (주먹 정면) | 앞 그림의 손 크기에 맞춰 |

### 지킬 것

- **투명 배경 PNG** (안 되면 단색 배경 — 도구가 지운다).
- **크기를 지금 그림에 맞춘다**: 칼은 `sword.png`(세로 520픽셀 칼)와 같은 축척, 손은 `front.png` 의 손 크기와 같게.
- 칼은 **손잡이가 그림 가운데**에 오게 둔다. 손잡이·칼끝 자리는 그림을 받은 뒤 도구 설정에 적는다.
- 같은 칼을 세 각도로 뽑아야 하므로, 한 장에 나란히 뽑은 뒤 잘라 쓰면 모양이 덜 어긋난다.

### 지시문 예시

```
game item sprite sheet, the same fantasy longsword shown three times side by side:
1) side view lying flat, 2) tilted 45 degrees toward the viewer with perspective foreshortening,
3) pointing straight at the viewer, only the hilt and cross-guard visible like a cross,
thick dark outline, flat colors, simple shapes, no texture, no shadow, plain light gray background
```

```
game character part sprite, a gloved fist punching straight toward the viewer, front view of the knuckles,
same glove as the reference image, thick dark outline, flat colors, no shadow, plain light gray background
```

---

## 넣는 곳과 다음 순서

1. 그림을 위 이름으로 `characters/warrior/` 에 넣는다 (여러 장을 한 장에 뽑았으면 그대로 넣어도 된다 — 잘라 준다).
2. 도구를 연결한다:
   - 대각선: 방향 목록에 `front_left`, `back_left` (+ 뒤집은 오른쪽) 를 더하고, 관절 찾기 → 조각 → 공통 동작이 그 방향으로도 바뀌게 한다.
   - 부위 각도: 칼·손을 기운 정도에 따라 바꿔 끼우는 기능을 넣는다.
3. 네 방향과 같은 확인(프레임별 큰 그림, 경고, 편집기와 계산 일치)을 한다.
