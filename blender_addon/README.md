# PHACK LAMM Blender 插件

基于 `LAMM_generate_local` 的 Blender 插件：随机生成 HACK 网格，并用 **10 个区域圆盘** 做局部 manipulation。

## 功能

1. **随机生成 Hack 网格**  
   从 `gaussian_id.pickle` 采样身份 latent，用 LAMM decode 得到网格（Blender 中头朝上）。

2. **局部区域圆盘编辑（10 patches）**  
   3D 视图右侧（靠近 PHACK 侧栏）有平面圆盘：点 **Drag Disk Pad** 后拖橙色点（水平→X，竖直→Z），**Depth** 控制 Y。  
   再点 **Apply**（或勾选 Live Apply）。

3. **随机区域编辑**  
   按 `displacement_stats` 采样（与 `generate_local` 一致），并回写到滑条。

## 安装

1. 确认本机已有 PHACK 训练产物：
   - `results/lamm_hack_manipulation/best_alpha_max.pth`
   - `results/lamm_hack_manipulation/gaussian_id.pickle`
   - `results/lamm_hack_manipulation/displacement_stats.pickle`

2. Blender **3.6+ / 4.x**：
   - `Edit → Preferences → Add-ons → Install...`
   - 选择目录 `blender_addon/phack_lamm`（或先将其打成 zip 再安装）
   - 勾选启用 **PHACK LAMM Hack Generator**
   - 若侧栏 PHACK 面板是空的：先取消勾选再重新勾选启用（或重启 Blender），让 PropertyGroup 重新注册

3. 打开 `View3D` 侧栏（快捷键 `N`）→ 标签 **PHACK**。

### 开发期免安装（推荐）

把插件目录软链到 Blender 的 addons 路径，例如：

```bash
# Blender 4.2 示例，按本机版本改路径
ln -s /media/ubuntu/SSD/PHACK_code/blender_addon/phack_lamm \
  ~/.config/blender/4.4/scripts/addons/phack_lamm
```

然后在 Preferences 里启用插件。

## 路径配置

侧栏里需要正确：

| 项 | 含义 |
|---|---|
| **Repo Root** | `PHACK_code` 根目录（默认会自动猜） |
| **Python** | 带 `torch` 与本仓库依赖的解释器（默认尝试 anaconda / `PHACK_PYTHON`） |
| **AE Checkpoint** | 生成用；留空 = `results/lamm_hack_ae/20260706_094816/best.pth` |
| **Manip Checkpoint** | 局部编辑用；留空 = `results/lamm_hack_manipulation/best_alpha_max.pth` |
| **Device** | `0` / `cpu` 等 |

推理在 **外部 Python 子进程** 中运行（不依赖 Blender 自带的 Python 是否安装了 torch）。

也可设置环境变量：

```bash
export PHACK_PYTHON=/home/ubuntu/anaconda3/bin/python3
```

## 使用流程

1. 设好 **Seed / k_std**，点 **Generate Random Hack Mesh**。
2. 选 **Region**，点 **Drag Disk Pad**，在视口右侧（侧栏旁）圆盘内拖动橙色点。
3. 用 **Depth / Scale / Amount** 微调，再点 **Apply**（或勾选 Live Apply）。
4. **Reset** / **Restore Source** 回到未编辑状态。

## 命令行自测（不打开 Blender）

```bash
cd /media/ubuntu/SSD/PHACK_code

# 随机身份
python blender_addon/phack_lamm/inference/runtime.py \
  --mode generate --seed 26 --out_npz /tmp/phack_source.npz

# 随机鼻子局部编辑
python blender_addon/phack_lamm/inference/runtime.py \
  --mode manipulate --source_npz /tmp/phack_source.npz \
  --random_region --region 8 --out_npz /tmp/phack_nose.npz

# 圆盘式平移（region 8 = Nose）
python blender_addon/phack_lamm/inference/runtime.py \
  --mode manipulate --source_npz /tmp/phack_source.npz \
  --offsets_json '{"8":[0,0.01,0]}' --out_npz /tmp/phack_disk.npz
```

## 目录结构

```
blender_addon/
  README.md
  phack_lamm/                 # Blender addon 包
    __init__.py
    properties.py
    operators.py
    panels.py
    mesh_utils.py
    inference/
      runtime.py              # LAMM 推理（generate / manipulate）
```

## 说明

- **生成**用 AE checkpoint；**局部编辑**用 manip checkpoint，并以残差合成  
  `result = source + (manip(δ) - manip(0))`，避免直接 decode manip 权重带来的 patch 断层。
- 圆盘：水平→X，竖直→Z，Depth→Y；位移在 Blender 坐标系，推理前转到模型 Y-up。
- 源身份缓存在 `/tmp/phack_lamm_blender/source.npz`。
