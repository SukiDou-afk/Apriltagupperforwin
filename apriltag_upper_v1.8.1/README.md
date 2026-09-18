# AprilTag 上位机 v1.8.1

本版修复 v1.8 按历史引用文件名停用 Base 输出的问题。文件名、路径、来源指纹均不限制用户选择标定组合。保留 v1.8 的 D435 参数控制、标定编辑副本，以及 AprilTag 识别、每 ID 独立目标偏移、长按云台控制与编码器监视。

## 启动与升级

1. 将完整 ZIP 解压到新文件夹。已有环境可在该文件夹打开终端，执行 `python apriltag_upper.py`，或双击 `run.bat`。
2. 如需保留 v1.7/v1.8 的分辨率、每 ID 偏移、串口等设置，将旧目录的 `settings.ini` 复制到新目录，再启动。旧手动标定参数会清理，不再参与计算。
3. `run.bat` 优先使用本目录 `.venv`；找不到时使用当前 Python。Anaconda 用户请先激活原来的环境，再执行 Python 启动命令。
4. 新环境使用 Python 3.10 或 3.11，运行 `python -m pip install -r requirements.txt`。不需要 Torch/CUDA。
5. 界面中填写 Tag 黑色外框的实际边长。若使用黑框 50 mm 的标签，填 **50 mm**，不是白色底板 56 mm。首次默认值仍沿用 v1.7 的 100 mm，必须按实物核对；设置随后自动保存。

## 相机硬件参数

启动相机，点击“展开相机参数”。程序只列出当前 RGB sensor 实际支持的项目，查询范围、步进、只读状态和实际值；鼠标停在参数框上查看这些信息。

可支持的项目包括自动曝光、曝光、增益、自动白平衡、白平衡、亮度、对比度、饱和度、锐度、Gamma、色调、背光补偿、电源频率、自动曝光优先级。最终显示项以连接设备报告为准。

- 自动曝光开启时，曝光与增益不可编辑；先关闭自动曝光。白平衡同理。
- 输入数值后按 Enter 或移开焦点即提交。参数按 SDK 步进对齐，并读回实际值。
- 成功写入的值按相机序列号保存。下次启动同一相机时恢复，先设置自动开关，再设置允许修改的手动值。
- 写入失败显示原因，不保存失败值；相机采集继续。硬件参数在独立线程中处理。
- Exposure 使用 SDK 的原生单位与范围，不将它错误标成毫秒或微秒；可悬停查看 SDK 描述。白平衡显示 K。
- 自动曝光优先级可能降低帧率。调参时同时查看相机输入 FPS、界面 FPS、AprilTag Hz。

接口依据：[RealSense 官方 sensor-control 示例](https://github.com/realsenseai/librealsense/blob/master/examples/sensor-control/api_how_to.h)。本版没有加入软件图像增强，识别与角点精修沿用现有图像链路。

## 标定文件原值与编辑副本

点击“展开标定数据 / 编辑副本”，两个标签页分别选择相机→云台、云台→机械臂 JSON。

界面显示 parent/child、原文件 translation/RPY、原 fit_summary；相机文件还显示轴、方向和拟合零位，机械臂文件显示 source_pantilt_camera。下方表单编辑工作副本，旋转矩阵随 RPY 同步显示。

- translation 在界面显示 mm，JSON 内保存 m。
- RPY 在界面显示 deg，JSON 内保存 rad；采用 `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`。
- 修改 RPY 会同步生成旋转矩阵，清除旧 rotation_rotvec，避免多种旋转表示互相覆盖。仅修改平移或 model 时不重算原旋转。
- parent/child、来源和 fit 信息作为只读元数据；参与变换的数值可编辑。
- 选择“使用标定文件原始数据”，计算使用读入的原始快照；表单仍可编辑副本，但编辑不会改变当前结果。
- 选择“使用界面编辑后的数据”，Base 结果与三维视图立即使用副本。
- 两处数据源提示和底部状态栏显示当前生效来源。
- “副本重置为文件值”丢弃该副本修改；“重新读取”重新读磁盘文件并重置该副本。
- 编辑副本自动保存于 settings.ini。仅当原始文件内容指纹一致时，下次启动才恢复该副本。
- “将两份副本另存为 JSON…”会新建独立子文件夹，保存两份 JSON 并更新相机引用与内容指纹，不覆盖输入文件，也不自动切换运行来源。
- 原文件 fit_summary 是历史拟合记录；编辑后不会重新标定，不能将原残差视为编辑后的精度。

## 标定文件选择与检查

用户已确认本次上传的以下两份文件正确。本包在 `calibration_reference/` 中原样收录，首次启动且没有历史选择时默认加载：

- `camera_to_gimbal_default.json`：相机→云台。
- `cr10a_arm_to_pantilt_charuco_calibration.json`：云台→机械臂。

默认选择仅用于方便启动。用户可以随时更换任何符合程序数据格式的相应标定文件，不存在限定组合或名单。已有 settings.ini 记录的文件选择优先保留；如果旧路径在当前电脑上不存在，请重新选择文件。

`source_pantilt_camera` 和 `source_pantilt_camera_sha256` 仅是历史来源信息，不再触发匹配检查或停用输出。没有这两个字段也不影响 Base 计算。无需重命名上传文件，原 JSON 不会自动改写。

仍检查文件能否读取、文件类型是否放对栏位、坐标系字段是否符合当前 Legacy 模型、数值是否有限，以及旋转矩阵是否有效。缺少一份必要文件或数据无效时，才停止对应 Base 输出并提示原因。点坐标和三维视图继续使用同一组生效数据。

编辑副本的内容指纹仍用于判断“当前文件有没有变化”，避免将上次编辑副本误恢复到后来更换的文件上；它不限制两份标定文件之间的组合。

`historical_reference/` 保留 v1.7 随包示例供追溯，不自动选择。字段中的 fit_summary 是原拟合记录，不代表现场定位误差保证。

## Legacy 坐标模型

旧 pan_base 是联合拟合使用的虚拟参考系，不等于可直接尺量的真实轴承中心。Pan 固定绕 +Z，Tilt 轴由文件定义，模型不引入额外 Pan→Tilt 平移。

```text
p = radians(pan_sign  * (PanCmd  - pan_zero_deg))
q = radians(tilt_sign * (TiltCmd - tilt_zero_deg))
P_pan = Rz(p) @ R_tilt(q) @ (R_TC @ P_camera + t_TC)
P_arm = R_BP @ P_pan + t_BP
```

本版继续严格使用 **Pan/Tilt Cmd**，编码器只用于监视。Home Pan/Home Tilt 仅控制“回中”按钮，独立保存，不是拟合 zero，也不改变标定计算。

本版保持 v1.7 的目标点换算行为，未新增独立的 `vision_correction_offset` 文件加载接口。交接时记录的补偿为 `[0,0,0]`；若后续需要非零补偿，应另行接入并验证，不能默认为本版已读取。

每个 ID 的偏移仍在 Tag 坐标系下：`P_camera = R_camera_tag @ offset_tag + t_camera_tag`。Camera 光学坐标为 X 向右、Y 向下、Z 向前。

## 验证与现场检查

本包包含 16 项回归测试，已通过语法编译、模块导入、Qt 离屏 GUI 检查和 ZIP 完整性检查。测试详情见 `VALIDATION.md`。

用户已确认本次两份标定文件经过验证。此次软件修订在开发环境中核对数据结构、变换数值与 GUI 行为；没有连接真实 D435、Windows 串口或云台，不将模拟测试视为新的实物精度验证。

建议现场依次检查：

1. 启动相机，核对 Tag 实际边长、分辨率和 FPS；确认 Camera XYZ 正常。
2. 关闭自动曝光，小幅调整曝光/增益；确认画面改变、读回值正常、采集不中断。
3. 重启软件，确认同一相机的参数恢复。
4. 选择需要使用的两份标定 JSON，观察“Base 可计算”和 Base XYZ；不需要改文件名。
5. 在副本中将 Arm translation X 增加 10 mm，切换至编辑副本，固定目标的 Base X 应增加 10 mm；切回文件原值应恢复。验证完可重置副本。
6. 检查云台短按/长按/松开和编码器显示。观察到的编码器变化不应单独改变 Legacy 换算角度源。

开发者测试命令：`python -m pip install pytest`，随后运行 `python -m pytest -q tests`。硬件无需连接。
