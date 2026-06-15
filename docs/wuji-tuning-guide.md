# 4.3 retargeting参数微调

retargeting参数配置文件，在本仓库 `public/example/config/` 路径下，**按手型选择对应文件**：

| 手型 | 左手 | 右手 |
|------|------|------|
| Wuji Hand（默认） | `adaptive_analytical_wuji_glove_left.yaml` | `adaptive_analytical_wuji_glove_right.yaml` |
| Wuji Hand 2（WH120） | `adaptive_analytical_wuji_glove_wh120_left.yaml` | `adaptive_analytical_wuji_glove_wh120_right.yaml` |

> **务必确认改对了文件**：WH120 的配置在带 `wh120` 的文件里，它通过 `optimizer.urdf_path / mjcf_path / link_naming` 指向 Wuji Hand 2 模型；改错文件（例如给 WH120 调参却编辑了默认 Wuji Hand 的 yaml）不会生效。下面 `tuning_tool.py` 的启动也要相应用 `--config` 指定 WH120 文件（见 §1）。

所有可调参数都在 YAML 的 `retarget` 块下。所有参数调整都建议先启动 `tuning_tool.py`，再修改 YAML，保存后通过可视化结果确认效果。优先调整几何修正 `segment_scaling`。

---

## 1. tuning_tool 启动和使用方法

`tuning_tool.py` 是调参时的主要观察工具，同步显示数采手套收集的手部运动和wuji-hand的运动以及其中的映射关系，查看目标骨架是否合理，wuji-hand 动作是否符合期望检查调参效果

启动方式：

```bash
cd public/example

# === Wuji Hand（默认手型）===
# 不带 --config 时，--wuji-glove 默认加载 config/adaptive_analytical_wuji_glove_<hand>.yaml
python tuning_tool.py --wuji-glove --hand right --glove-sn <YOUR_SN>
python tuning_tool.py --wuji-glove --hand left  --glove-sn <YOUR_SN>

# === Wuji Hand 2（WH120）===
# 必须用 --config 显式指定 wh120 文件，否则会回落到默认 Wuji Hand 配置
python tuning_tool.py --wuji-glove --hand right --glove-sn <YOUR_SN> \
  --config config/adaptive_analytical_wuji_glove_wh120_right.yaml
python tuning_tool.py --wuji-glove --hand left  --glove-sn <YOUR_SN> \
  --config config/adaptive_analytical_wuji_glove_wh120_left.yaml
```

> WH120 调参时，启动命令的 `--config` 与你编辑的 yaml **必须是同一个 wh120 文件**，否则看到的是默认 Wuji Hand 的效果。

使用方法：

1. 启动 `tuning_tool.py`。
2. 打开**与启动命令一致**的那个 YAML：默认 Wuji Hand 用 `adaptive_analytical_wuji_glove_<hand>.yaml`，WH120 用 `adaptive_analytical_wuji_glove_wh120_<hand>.yaml`。
3. 修改 `retarget` 下的参数并保存。
4. 工具会热加载配置；观察三层 skeleton 的变化。
5. 如果结果变好，再继续小步调整；如果变差，回退上一次修改。

颜色含义：

| 颜色 | 含义 | 调参时怎么看 |
|------|------|--------------|
| 橙色 | 原始输入 21 点 | 判断手套输入姿态是否正常 |
| 青色 | 橙色线经参数缩放后的目标 | 主要看 `segment_scaling` 调完后目标是否合理 |
| 白色 | wuji-hand 骨架| 检查wuji-hand是否跟上目标青色线的输入目标 |
（注：军绿色为橙色和青色重合的显示结果，在手指segment_scaling = 1.0 时出现，即不对手套输入进行放缩映射处理）

---

## 2. 最常用： `segment_scaling`

几何缩放：按手指、按目标向量（wrist→PIP / wrist→DIP / wrist→TIP）微调长度，修正某根手指长度与wuji-hand骨架不一致的问题。所有几何修正都通过 `segment_scaling` 完成。

参数位置：

```yaml
retarget:
  segment_scaling:
    thumb:  [1.0, 1.0, 1.0]
    index:  [1.0, 1.0, 1.0]
    middle: [1.0, 1.0, 1.0]
    ring:   [1.0, 1.0, 1.0]
    pinky:  [1.05, 1.05, 1.1]
```

| 参数 | 含义 | 默认 | 调小 / 调大 |
|------|------|------|-------------|
| `segment_scaling` | 每指对 wrist→PIP / wrist→DIP / wrist→TIP 三个目标向量分别缩放，用于适配用户单根手指比例（也支持 `[MCP, PIP, DIP, TIP]` 4 元素格式） | 左手 pinky `[1.05, 1.0, 1.1]`；右手 pinky `[1.05, 1.05, 1.1]`；其他多为 `1.0` | ↓该手指目标更短 / ↑该手指目标更长 |

使用建议：

- 某根手指整体偏短、够不到(即青色线段短于白色骨架)，调大该手指 3 个值；某根手指过长或过度伸展(即青色线段长于白色骨架)，调小该手指 3 个值。
- 保存修改后看青色 target 是否变得跟白色骨架贴合再继续。
- YAML 里另有一个 `scaling` 字段不是几何缩放，归在"捏合调节"，详见 §3、§4，常规保持 `1.0`。

---

## 3. 参数分类

`retarget` 块下的参数按用途可以分五类：

- 关节权重：`w_pos` / `w_dir` / `w_full_hand`，分别决定算法对"指尖位置 / 指尖朝向 / 整指姿态"的偏好优先级。其中 `w_pos` / `w_dir` 主要影响捏合状态下的指尖精度，`w_full_hand` 主要影响张开手时的整指形状。
- Huber 阈值：`huber_delta` / `huber_delta_dir`，控制位置误差和朝向误差从二次惩罚进入线性惩罚的阈值，影响算法对大偏差的敏感程度。
- 软限位 / 姿态约束：`thumb_skip_pip` / `w_hyper` / `soft_min` / `w_couple` / `couple_ratio`，其中 `thumb_skip_pip` 会牺牲一部分大拇指关节贴合度来提升整体映射自然度；`w_hyper` / `soft_min` 用于减少反向弯曲；`w_couple` / `couple_ratio` 把 DIP 弯曲与 PIP 软联动到更像人手的比例。
- 平滑：`lp_alpha` / `norm_delta`，抑制输出抖动；前者作用在输出层，后者作用在优化项里。
- 模式 / 捏合调节：`pinch_thresholds` / `scaling` / `mediapipe_rotation`，分别为捏合 / 张开模式切换阈值、捏合状态下指尖目标的整体缩放、输入坐标系微调。

---

## 4. 参数速查表

| 参数 | 含义 | Wuji Glove 默认 | 调小 ↓ / 调大 ↑ |
|------|------|-----------------|-----------------|
| `segment_scaling` | 每指对 wrist→PIP / wrist→DIP / wrist→TIP 三个目标向量分别缩放（也支持 `[MCP, PIP, DIP, TIP]` 4 元素格式） | 左手 pinky `[1.05, 1.0, 1.1]`；右手 pinky `[1.05, 1.05, 1.1]`；其他多为 `1.0` | ↓该手指目标更短 / ↑该手指目标更长；最常调 |
| `w_pos` | 指尖位置损失权重，wrist→tip 向量误差 | `1.0` | ↓允许指尖偏位 / ↑指尖位置更准 |
| `w_dir` | 指尖朝向损失权重，DIP→TIP 方向误差 | `2.0` | ↓朝向更宽松 / ↑指向更精确 |
| `w_full_hand` | 整指姿态损失权重，wrist→PIP/DIP/TIP 三组向量 | `1.0` | ↓中段关节更自由 / ↑整指更贴输入 |
| `huber_delta` | 位置误差 Huber 阈值，单位 cm | `2.0` | ↓对大偏差更宽容 / ↑对大偏差更敏感 |
| `huber_delta_dir` | 朝向误差 Huber 阈值 | `0.5` | ↓朝向大误差更宽容 / ↑朝向大误差更敏感 |
| `thumb_skip_pip` | 是否把拇指 MCP `kp[2]` 从 FullHandVec 损失剔除 | `true` | 通常保持 `true`，用于容忍 SDK 拇指 MCP 不准 |
| `w_hyper` | 反向弯曲惩罚强度，PIP/DIP < `soft_min` 时生效 | `1.0` | ↓允许更多反弯 / ↑更强制不反弯 |
| `soft_min` | 软限位最小关节角阈值，单位 rad | `0.0` | 一般不改；`0.0` 表示不希望出现负角度 |
| `w_couple` | DIP 朝 `couple_ratio × PIP` 拉近的约束强度 | `0.1` | ↓DIP/PIP 更独立 / ↑DIP 更紧随 PIP |
| `couple_ratio` | DIP 相对 PIP 的弯曲比例 | `0.7` | ↓DIP 弯得更少 / ↑DIP 弯得更多 |
| `lp_alpha` | 输出关节角低通滤波系数 | `0.2` | ↓更平滑但延迟更大 / ↑更跟手但更容易抖 |
| `norm_delta` | 帧间关节速度正则项权重，软限速 | `0.04` | ↓更跟手但可能抖 / ↑更平滑但更保守 |
| `pinch_thresholds.d1/d2` | 拇指与指尖距离触发捏合的双阈值（cm）：dist ≥ `d2` 完全 FullHand；dist ≤ `d1` 捏合权重饱和（上限 0.7，剩 30% 仍是 FullHand） | `2.0 / 4.0` | ↓更晚进捏合模式 / ↑更早进捏合模式 |
| `scaling` | 仅缩放捏合模式下的指尖位置目标（不影响 FullHand 损失和青色 target 显示） | `1.0` | ↓捏合时指尖目标更近 / ↑捏合时指尖目标更远；常规保持 `1.0` |
| `mediapipe_rotation` | 输入 21 点整体旋转微调，单位度 | Wuji Hand：左手 `(5, -5, 20)`、右手 `(-5, -5, -20)`；WH120 因腕部 link 朝向不同，默认值见对应 `wh120` 文件 | 用于纠正手套坐标系朝向；不同手型与左右手的默认值已分别写在各自 yaml 里 |
| `wrist_offset_cm` / `thumb_offset_cm` | `wrist_offset_cm`：对除拇指外的四指 16 个关键点统一平移；`thumb_offset_cm`：对拇指 4 个关键点统一平移。单位 cm | `[0, 0, 0]` | 当前默认关闭；SDK 已做按 SN 标定，一般不建议客户自行修改 |

---

## 5. 实际调参顺序

1. 先启动 `tuning_tool.py`，重点观察青色 target 和 白色的wuji-hand骨架的变化。
2. 先调 `segment_scaling`：只修某一根手指偏长 / 偏短的问题，每次只改一根手指。
3. 有抖动再调 `lp_alpha` / `norm_delta`：`lp_alpha` 低一点更稳，`norm_delta` 高一点更保守。
4. 捏合误触发或不触发，再调 `pinch_thresholds.d1/d2`：调大 `d1/d2` 会更早进入捏合模式，提高对指尖捏合的偏好；调小 `d1/d2` 会更晚进入捏合模式，更偏向保持全手姿态。
5. `w_pos` / `w_dir` / `w_full_hand` / `w_hyper` / `w_couple` / `couple_ratio`。属于算法偏好权重，不建议优先改，除非已经确认几何尺度和平滑参数都没问题。注意 `w_pos` / `w_dir` 主要影响捏合状态下的指尖精度，`w_full_hand` 主要影响张开手时的整指形状。
