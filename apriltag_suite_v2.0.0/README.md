# AprilTag 上位机 + Python SDK · v2.0.0

从 v1.8.1 迁移到 **PyQt6**。完整上位机、可安装 Python 包和同事接入示例一起交付。识别、标定变换、相机控制与串口协议只有一份核心实现，界面调用 `apriltag_sdk`。

- 你继续使用完整上位机：启动 `apriltag_upper.py`。
- 同事需要全部界面：运行 `examples/embed_full_panel.py`，将视觉云台页嵌入自己的 PyQt6 上位机。
- 同事需要自定义界面：使用 `Locator` / `AprilTagSystem` 与可复用的 PyQt6 控件。
- 同事已经打开相机：把原始 BGR 图像和对应内参交给 `Locator`，不要再启动一条 D435 流。

## 1. 内容与入口

| 路径 | 内容 |
|---|---|
| `apriltag_upper.py` | 完整 PyQt6 上位机入口 |
| `apriltag_sdk/` | Python SDK 源码；核心不依赖 Qt |
| `apriltag_sdk/qt/` | PyQt6 适配器、完整界面、标定/RGB/3D 控件 |
| `dist/apriltag_robot_sdk-2.0.0-py3-none-any.whl` | 可安装 SDK；也包含完整上位机 |
| `dist/apriltag_robot_sdk-2.0.0.tar.gz` | 源码发行包 |
| `requirements-py38.txt` | Python 3.8 固定依赖版本 |
| `requirements-py314.txt` | Python 3.14 固定依赖版本 |
| `examples/` | 外部图像、D435 无界面、PyQt6、自带完整功能页接入示例 |
| `docs/INTEGRATION.md` | 给同事的接入文档，建议先读 |
| `docs/API.md` | 接口、单位、线程与生命周期约定 |
| `docs/MIGRATION.md` | 旧上位机设置迁移和行为说明 |
| `docs/VALIDATION.md` | 实际测试范围、环境及设备验证事项 |
| `apriltag_sdk/assets/calibration/` | 用户提供的两份参考标定文件，保持原内容 |
| `tags/` | 原项目 AprilTag 36h11 ID 0–3 图像 |

## 2. 安装与启动

使用 **64 位 CPython**。下面的命令在解压后的项目目录执行。推荐创建独立环境，避免旧 PySide6、不同 OpenCV 发行包混入同一环境。

### 你的 Python 3.14.6

Windows 命令提示符：

```bat
py -3.14 -m venv .venv
.venv\Scripts\activate
python -V
python -m pip install --upgrade pip
python -m pip install -r requirements-py314.txt
python -m pip install --no-deps dist\apriltag_robot_sdk-2.0.0-py3-none-any.whl
python apriltag_upper.py
```

`py -3.14` 选择已安装的 3.14 解释器，以 `python -V` 显示为准。没有 Windows `py` 启动器时，使用你实际 Python 的完整路径创建环境。

### 同事的 Python 3.8

```bat
py -3.8 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-py38.txt
python -m pip install --no-deps dist\apriltag_robot_sdk-2.0.0-py3-none-any.whl
python examples\pyqt6_host.py --simulate
```

不要在 Python 3.8 中无版本约束地强装最新 PyQt6。本项目为它固定了 PyQt6 6.7.1、Qt 6.7.3 与相应 RealSense/OpenCV 版本。

### 安装源码或仅安装需要的功能

```bash
# 当前目录作为可编辑源码，适合同事开发；自动按 Python 版本选择依赖
python -m pip install -e ".[qt,camera]"

# 仅识别外部图像/标定/串口，不安装 Qt 与 RealSense
python -m pip install dist/apriltag_robot_sdk-2.0.0-py3-none-any.whl

# 在已有 SDK 基础上补齐全部依赖，或单独交付 wheel 时使用
python -m pip install "./dist/apriltag_robot_sdk-2.0.0-py3-none-any.whl[qt,camera]"
```

正式复现本次验证环境，优先使用上面的固定依赖文件。Linux 用 `python3` 创建环境，再执行 `source .venv/bin/activate`。需要正常桌面环境及相机 USB 权限。

`install_current_env.bat` 会为**当前终端的 Python**安装固定依赖和源码包，不会自动切换 Python。`run.bat` 从包目录启动上位机。安装后也可以执行 `apriltag-upper`。

只保留一个提供 `cv2` 的 OpenCV 发行包。本项目使用 `opencv-contrib-python-headless`；它仍能识别 AprilTag，图像显示由 PyQt6 完成，不使用 `cv2.imshow`。

## 3. 无设备快速试用

```bash
python examples/external_image.py --synthetic --output preview.png
python examples/pyqt6_host.py --simulate
```

第一个示例生成 ID 3 测试图，完成真实 AprilTag 检测和 PnP，输出 JSON 与带标记的预览图。没有加载外参/提供 Cmd 时，`target_base_mm` 为 `None`。这些合成结果仅用于验证接口，不能替代设备定位精度验收。

## 4. 接入设备

```bash
python apriltag_upper.py
# 或不用界面，只运行相机识别
python examples/headless_d435.py --tag-size-mm 50 --use-bundled-calibration
```

上位机默认 Tag 边长仍为原版的 100 mm；**请按打印后黑色 Tag 外边界的实际边长填写**，不包括周围白边。例如黑色区域为 50 mm，就填 50。示例使用 50 mm。

RGB 参数是 RealSense 硬件参数，影响实际采集的图像，因而同时影响预览和识别输入。预览缩放、窗口大小、文字与坐标轴绘制只处理图像副本，不回写识别输入。距离来自彩色内参 + Tag 尺寸 + PnP，未使用深度流。

## 5. 保留的功能

- D435 原生彩色分辨率发现、BGR/RGB/YUYV/UYVY/MJPEG 等格式转换；采集、识别、预览分离，最新帧覆盖旧帧。
- AprilTag 36h11、960 像素检测宽度上限、原图亚像素角点、IPPE_SQUARE 位姿、5 次观测位置中值、每个 ID 独立目标偏移。
- Camera/机器人 Base 坐标，文件原值与编辑副本切换，编辑副本导出；标定链 3D 交互视图。
- **不按两份标定文件的名称、路径、来源字段或来源哈希限制组合**。仍检查文件类型、数值和支持的几何模型。
- 云台短按单步、长按连续、Home、最新目标限频发送、GBK 编码器反馈、Cmd/Encoder 监视。
- Legacy 坐标始终使用 **Cmd**；编码器只监视。Home 与标定模型的零位是不同设置。
- 原上位机 INI 设置保存；SDK 独立 JSON 设置保存；RGB 成功写入值按设备序列号恢复。

没有增加机械臂运动指令接口；SDK 输出定位数据，由同事上位机决定如何使用。

## 6. 给同事交付什么

建议把本目录完整交给同事。他只需开发运行时依赖时，也可以只拿 `dist/*.whl`、对应 `requirements-*.txt`、`docs/` 与 `examples/`。wheel 内已包含 SDK、PyQt6 完整界面和参考标定 JSON，第三方依赖由 pip 安装，不是离线安装包。

先看 [接入文档](docs/INTEGRATION.md)，需要查具体参数时看 [接口文档](docs/API.md)。旧设置迁移看 [迁移说明](docs/MIGRATION.md)。实际验证结论见 [测试报告](docs/VALIDATION.md)。
