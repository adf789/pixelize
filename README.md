# pixelize

AI로 뽑은 캐릭터 일러스트를 **64×64 픽셀 그림**으로 바꾸는 도구.
모든 그림에 같은 규칙(키, 발 위치, 색 수, 외곽선)을 적용해서 결과를 일정하게 맞춘다.

## 설치

`python\`(Python 3.12 내장판)과 `models\`(자세 인식 모델)는 용량이 커서 저장소에 넣지 않는다.
새로 받았을 때:

1. Python 3.12 내장판(Windows embeddable)을 `python\` 에 풀고, `python312._pth` 의 `import site` 줄 앞 `#` 을 지운다.
2. `get-pip.py` 로 pip 를 깐 뒤:
   ```
   python\python.exe -m pip install -r requirements.txt
   ```
3. 모델은 `animate.bat rig` 를 처음 돌릴 때 `models\` 에 저절로 받아진다 (약 220MB).

## 사용

```
pixelize.bat input\hero -o output\hero -c config.json
```

- `input/hero/` 에 `hero_front.png`, `hero_side.png`, `hero_back.png` 처럼 **한 캐릭터의 그림을 한 폴더에** 넣는다.
- 한 폴더의 그림은 **팔레트(색 묶음)를 함께 쓴다.** 그래서 앞/옆/뒤 색이 서로 어긋나지 않는다.
- 결과:
  - `output/hero/hero_front.png` … 실제로 Unity에 넣을 64×64 그림
  - `output/hero/preview/` … 눈으로 보기 쉽게 8배로 키운 그림
  - `output/hero/palette.hex` … 이번에 만든 팔레트

## 처리 순서

1. 배경 지우기 — 네 모서리 색을 배경으로 보고, 테두리와 이어진 비슷한 색을 투명하게 만든다. 이미 투명 배경이면 건너뛴다.
2. 작은 조각 지우기 — 본체와 떨어진 작은 덩어리(그림 아래 "Front" 글자, 티끌)를 지운다.
3. 팔레트 정하기 — 원본 그림들의 색에서 팔레트를 만든다 (또는 정해 둔 팔레트를 쓴다).
4. 크기 맞추기 — 원본 색을 팔레트 색으로 바꾼 뒤, 캐릭터 키가 `char_height` 칸이 되도록 칸마다 **가장 많이 나온 색 하나**를 고른다. 평균을 내지 않으므로 색이 섞여 탁해지지 않는다. 발끝은 바닥에서 `foot_margin` 칸 위에 맞춰 가운데 둔다.
5. 외곽선 — 바깥쪽에 1칸 테두리를 두른다.

## 설정 (`config.json`)

| 항목 | 기본값 | 뜻 |
| --- | --- | --- |
| `canvas` | 64 | 그림 한 장의 가로·세로 칸 수 |
| `char_height` | 48 | 캐릭터 키(칸). 무기 휘두르기와 효과가 들어갈 자리를 남기려고 64보다 작게 잡음 |
| `foot_margin` | 4 | 바닥에서 발끝까지 띄울 칸 수. 모든 캐릭터의 발이 같은 줄에 선다 |
| `colors` | 24 | 팔레트를 새로 만들 때 쓸 색 수 |
| `palette` | null | 정해 둔 팔레트 파일(`.hex`) 경로. 설정 파일 위치 기준. 비우면 그림에서 새로 만든다 |
| `alpha_threshold` | 128 | 이 값보다 진하면 "보임"으로 친다 (0~255) |
| `bg_tolerance` | 40 | 배경색과 얼마나 달라도 배경으로 볼지 (0~255) |
| `min_part_ratio` | 0.02 | 가장 큰 덩어리의 이 비율보다 작은 떨어진 조각은 지운다. 떨어진 무기 등이 지워지면 낮춘다 |
| `outline` | true | 외곽선을 두를지 |
| `outline_color` | `#1b1325` | 외곽선 색 |
| `preview_scale` | 8 | 미리 보기 확대 배율 |

### 게임 전체 색을 맞추는 방법

1. 첫 캐릭터를 돌리면 `palette.hex` 가 나온다. (또는 Lospec 같은 곳의 `.hex` 팔레트를 받아도 된다.)
2. 이 파일을 `palettes/game.hex` 로 옮기고 `config.json` 의 `"palette": "palettes/game.hex"` 로 지정한다.
3. 그 뒤로 만드는 캐릭터·몬스터는 모두 같은 색만 쓴다.

## AI 그림을 뽑을 때 지킬 것

64칸으로 줄이면 아주 작아지므로, **단순한 그림일수록 결과가 좋다.**

- 단색 배경 (흰색이나 연한 회색), 그림자 없음
- 전신, 발끝까지 잘리지 않게
- 굵은 외곽선, 평평한 색, 큰 덩어리 (잔무늬·질감 빼기)
- 머리를 크게 (2~3등신) — 작은 그림에서 얼굴이 보이려면 필요하다
- 앞/옆/뒤를 **한 장에 나란히** 뽑은 뒤 잘라 쓰면 서로 모양이 덜 어긋난다
- 게임에 넣을 그림은 팔 벌린 자세(T자) 대신 **팔을 내린 자연스러운 서기 자세**로 뽑는다. T자 자세는 Spine 같은 뼈대 작업용이다
- 어두운 옷만 입히면 작아졌을 때 덩어리로 뭉친다. 밝은 색 부분(천, 장식)을 크게 넣으면 잘 읽힌다

지시문 예시:

```
character turnaround sheet, front view, side view, back view, same character,
top-down RPG game character, chibi proportions, full body, standing pose,
thick dark outline, flat colors, simple shapes, no texture, no shadow,
plain white background, <캐릭터 설명>
```

## 알아 둘 점

- 자동 결과는 **바탕 그림**이다. 눈·손·무기 같은 작은 부분은 Aseprite에서 몇 칸 손보는 게 좋다.
- 왼쪽 그림은 따로 만들지 않고 옆 그림을 Unity에서 `SpriteRenderer.flipX` 로 뒤집어 쓴다.
- Unity 그림 설정: `Filter Mode = Point`, `Compression = None`, `Pixels Per Unit = 64`, 기준점(Pivot)은 발끝 위치인 `Custom (0.5, 0.0625)` (= `foot_margin` 4 ÷ 64).

---

# animate — 뼈대 동작 → 픽셀 스프라이트

T자 일러스트에 뼈대를 심어 걷기·뛰기·공격 같은 동작을 만들고, 프레임마다 위의 픽셀화 규칙(키, 발 위치, 팔레트, 외곽선)으로 찍는다.
Claude 에게는 **`/sprite-animate` 스킬**(`.claude/skills/sprite-animate/SKILL.md`)로 맡기면 된다.

## 흐름

```
animate.bat rig    characters\warrior                     # 1. 관절 찾기 → rig_<방향>.json, check\rig_<방향>.png
(확인 그림을 보고 rig_<방향>.json 고치기)
animate.bat check  characters\warrior                     # 2. 고친 rig 로 확인 그림만 다시
animate.bat render characters\warrior -o output\warrior   # 3. 동작 만들기
animate.bat render characters\warrior -o output\warrior --only side/attack   # 일부만
```

## 원리

- **관절 찾기**: 자세 인식 모델(RTMW, `models\` 에 약 220MB 내려받음)이 어깨·팔꿈치·손목·골반·무릎·발목·발끝·얼굴을 찾는다.
- **그림을 자르지 않는다**: 그림 위에 촘촘한 그물(메시)을 씌우고, 그물 점마다 "그물을 따라 걸어서 가장 가까운 뼈 두 개"를 따라가게 한다. 고무판처럼 휘므로 관절에 틈이 생기지 않는다.
- **T자 팔 내리기**: 팔이 45도 넘게 벌어져 있으면 `base_pose` 로 자동으로 내린다. 모든 동작은 이 자세 위에 더해진다.
- **옆 그림**: 두 다리가 겹쳐 있어 따로 움직일 수 없으므로, 보이는 팔다리(`_l`)를 복사해 어둡게 칠한 뒤쪽 팔다리(`_r`)로 쓴다.
- **추가 뼈(`extra_bones`)**: 망토처럼 팔다리에 붙어 있지만 따로 움직일 부분에 뼈를 더한다. 없으면 팔을 들 때 망토가 같이 끌려 올라간다.
- **붙인 물건(`attachments`)**: 칼처럼 휘면 안 되는 그림 한 장을 손 뼈에 붙인다. 그물로 휘지 않고 네모 그대로 손을 따라 움직이며, 동작에서 `sword` 같은 이름으로 손목처럼 돌릴 수 있다. 그 그림의 색도 팔레트에 들어간다. 형식은 스킬 문서 참고.
- **잘림 경고**: 그림이 64칸 가장자리에 닿으면 `주의 … 칸 밖으로 잘림` 을 출력한다.

## 파일

| 경로 | 내용 |
| --- | --- |
| `characters\<이름>\front.png` 등 | 원본 일러스트 (앞·뒤는 T자, 옆은 왼쪽을 봄) |
| `characters\<이름>\rig_<방향>.json` | 관절 좌표(원본 픽셀 기준), 기본 자세, 앞뒤 순서, 추가 뼈, 붙인 물건 |
| `characters\<이름>\sword.png` | 손에 붙일 칼 그림 (지금 것은 코드로 그린 시험용. AI로 뽑은 그림으로 바꾸고 `grip`/`tip` 좌표만 고치면 된다) |
| `characters\<이름>\check\rig_<방향>.png` | 관절·뼈·무게 확인 그림 (100칸 눈금) |
| `anims\<방향>\<동작>.json` | 동작 열쇠 프레임. 형식은 스킬 문서 참고. 뒤 그림은 `{"mirror_of": "front/walk"}` 로 앞 동작을 뒤집어 쓴다 |
| `output\<이름>\<방향>_<동작>.png` | 64칸 × 프레임 수 가로 띠 스프라이트 시트 |
| `output\<이름>\animations.json` | 동작마다 프레임 수, fps, 반복 여부, 기준점 |
| `output\<이름>\check\` | 동작 확인 그림(위: 원본 휘기, 아래: 픽셀)과 미리 보기 GIF |

설정(`config.json`)에 두 가지가 더 있다: `render_scale`(64칸의 몇 배 크기로 그린 뒤 줄일지, 기본 8), `mesh_step`(그물 한 칸 크기, 기본 4. 작을수록 부드럽지만 느리다).

## Unity 에 넣기

- 시트를 `Sprite Mode = Multiple`, `Grid By Cell Size 64×64` 로 자른다.
- 기준점은 `animations.json` 의 `pivot` (발끝).
- 왼쪽 방향은 옆 시트를 `SpriteRenderer.flipX` 로 뒤집어 쓴다 (옆 그림이 왼쪽을 보므로, 오른쪽으로 갈 때 뒤집는다).
