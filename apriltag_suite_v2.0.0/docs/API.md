# SDK 接口参考 · 2.0.0

安装包名 `apriltag-robot-sdk`；Python 导入名 `apriltag_sdk`。推荐使用本文列出的公共入口。带 `_` 的成员是内部实现，不作为接入契约。核心包导入不加载 Qt / RealSense。

## Locator：外部图像与坐标计算

```python
from apriltag_sdk import Locator
loc = Locator(tag_size_mm=50.0, median_enabled=True, detect_max_width=960)
```

| 方法 | 输入和行为 |
|---|---|
| `configure(tag_size_mm=None, median_enabled=None, detect_max_width=None)` | 只更新提供的选项，并清空位置滤波历史；尺寸正且有限；检测宽度≥32 |
| `set_offsets(offsets_mm)` | 替换整个每ID偏移表，如 `{3:[20,0,10]}`；未配置ID偏移为0；传空字典清空 |
| `load_calibration(kind, path)` | kind=`camera`或`arm`；新文件无效时抛异常，并停用该槽旧标定 |
| `set_calibration_source(source)` | `original`文件原值 / `edited`编辑副本 |
| `edit_calibration(kind, xyz_mm=None, rpy_deg=None, model=None)` | 只改内存副本；XYZ毫米，RPY度，R=RzRyRx；model若提供则替换整个model字典 |
| `reset_calibration_copy(kind)` | 将该副本重置为本次加载的文件原值 |
| `set_calibration_documents(camera_document, arm_document, source="original")` | 从 CalibrationDocument/None 复制文档，用于与标定面板同步 |
| `calibration_chain()` | 当前变换链快照，平移单位米；不可用时含 blocked_reason |
| `export_calibrations(directory, source="edited")` | 创建新目录并导出 camera.json、arm.json；目录已存在会拒绝，不覆盖源文件 |
| `export_state()` / `restore_state(state)` | 可JSON持久化的状态；恢复会重新读取各路径；源内容改变则不恢复旧编辑副本 |
| `process(image_bgr, camera_matrix, dist_coeffs, pan_cmd_deg=None, tilt_cmd_deg=None, captured_at=None, frame_seq=None)` | 同步返回 DetectionBatch |

同一 Locator 的公开状态变更和计算串行加锁。状态改变应用到下一次 `process()`；不会修改已经交出的批次。如果改变实时流的内参、分辨率或更换相机，应通过 `configure()` 清空此前滤波历史。

`bundled_calibrations()` 返回包内 `(camera_path, arm_path)` 两个 pathlib.Path，仅是便于明确选择的参考文件路径。无隐式匹配约束。

### DetectionBatch

| 字段/属性 | 类型与含义 |
|---|---|
| `frame_seq` | int，调用者序号或 Locator 自增序号 |
| `captured_at` | float，主机 monotonic 秒；未传时取 process 开始时间 |
| `finished_at` | float，处理完成 monotonic 秒 |
| `detect_ms` | float，本次处理用时，包含串行锁等待和坐标转换 |
| `pan_cmd_deg`, `tilt_cmd_deg` | float/None，本批用于计算的 Cmd |
| `calibration_source` | `original` / `edited` |
| `base_unavailable_reason` | str，缺标定/缺Cmd的原因；正常为空字符串 |
| `targets` | tuple[Target,...]；无Tag是空元组 |
| `overlays` | list，原图像素坐标下的绘图数据，供 annotate 使用 |
| `age_s` | property，距 captured_at 的实时秒数 |
| `to_dict()` | 返回独立、可 json.dumps 的字典 |

### Target

| 字段 | 类型与单位 |
|---|---|
| `tag_id` | int |
| `tag_camera_mm` | tuple(x,y,z)，毫米；启用中值时是最近最多5次该ID观测的位置中值 |
| `target_camera_mm` | tuple(x,y,z)，毫米；Tag中心加旋转后的偏移 |
| `target_base_mm` | tuple(x,y,z)/None，毫米 |
| `offset_tag_mm` | tuple(dx,dy,dz)，毫米，Tag局部坐标 |
| `rotation_camera_tag` | 3×3 list；当前帧姿态，未做中值滤波 |
| `corners_px` | 4×2 list，原图亚像素角点 |
| `reprojection_error_px` | float，当前帧原始PnP四角点的像素RMS重投影误差 |

位置中值保留旧版逻辑：每ID最近5次成功识别，并非严格连续5个相机帧；旋转使用当前帧。丢失后重新出现可能短暂受旧观测影响，需要重新初始化时调用 `configure()`。

数据类字段不可重新赋值，但嵌套 list 是普通 Python 容器；把交出的批次当作只读。`to_dict()` 适合记录或跨模块传递。

## AprilTagSystem：托管设备

```python
from apriltag_sdk import AprilTagSystem
system = AprilTagSystem(locator=None, config_path=None, detection_hz=15)
```

| 方法/属性 | 契约 |
|---|---|
| `locator` | 上述 Locator；通过它设置识别/标定/偏移 |
| `gimbal` | GimbalController |
| `start_camera(width=None,height=None,fps=None,pixel_format="bgr8",serial=None)` | 全省略尺寸时发现设备并选择最高分辨率；否则三项都必填；异步启动 |
| `stop_camera(timeout_s=5.0)` | 请求停止并等待两个工作线程，超时抛 TimeoutError，保留仍运行对象供再次关闭 |
| `running` | 相机线程是否活跃且未被请求停止；不是首帧已到达或标定已就绪的证明 |
| `state_snapshot()` | `{running,status,error}` |
| `latest_result(max_age_s=1.0)` | 批次独立副本；不存在/过期返回None；传None关闭年龄限制 |
| `latest_frame(max_age_s=1.0)` | `(seq,captured_at,owned_bgr,K,D,meta)` 或None；数组为拷贝 |
| `request_camera_options(values)` | 返回请求generation，未运行/参数线程未就绪抛 RuntimeError |
| `camera_options_snapshot()` | 最近RGB快照的独立副本或None |
| `save_config(path=None)` | 显式路径或构造时config_path；保存JSON，原子替换目标配置 |
| `load_config(path)` | 停机时加载JSON；不存在则保持当前配置；坏JSON/失效标定路径抛异常 |
| `close()` | 停相机、断串口；构造时给了config_path则成功关闭后保存设置 |
| `with AprilTagSystem(...) as system` | 退出with调用close |

`detection_hz` 目标范围1–30 Hz，并不保证耗时过长时也能达到。生命周期与配置文件读写应由一个拥有者线程调用；只把文档中明确的快照/事件用于其他线程。不要在运行中直接给 `locator`、`gimbal` 等成员换对象。

配置保存包含：Locator参数、偏移、两个标定路径与编辑副本、计算来源、Home、TX频率、RGB成功写入值、识别频率。它不保存/自动打开串口，不自动启动相机或自动运动。分辨率、串口端口等由宿主自己的启动配置管理。完整上位机还通过INI保存窗口、预览、所选相机profile等UI状态。

### EventHook 事件

`status(str)`、`error(str)`、`result_ready(DetectionBatch)`、`rgb_options(dict)`。

```python
system.error.connect(handler)
system.error.disconnect(handler)
```

这些是普通 Python 回调，不是 Qt signal；由发出事件的后台线程同步调用。回调应短小，不操作 QWidget，不执行阻塞任务，不在其中调用生命周期方法。回调异常会写 Python logging，不拖垮其他订阅者。慢回调仍会拖慢发出它的线程。PyQt6 宿主优先用 `QtSystemBridge`。

RGB 快照：`serial`、`records`（各参数范围/当前值/只读/可选枚举）、`applied`（成功值）、`errors`、`generation`。请求槽会合并同名待写值；没有逐条排队保证。参数失败不会立即停止视频流。

### 时效与内存

System 内部只保留最新图像与最新结果，不追赶历史帧。`latest_frame()` 为用户拷贝一张图，安全但有带宽成本；无需显示原尺寸时可先在宿主预览链缩小。底层 `frames.snapshot()` 为借用只读约定，性能要求高才使用，禁止原地修改。

配置变更生效于之后处理的新批次，已经返回的批次保持其历史意义。`stop_camera()` 成功后清空图像和结果；设备连续5次取帧失败会报错并退出采集。USB驱动调用卡死时关闭仍可能超时，调用者应保留对象并处理错误。

## GimbalController

继承原 `PantiltSerial`，TX协议保持10字节：

```
FF FE pan tilt 00 00 00 00 00 (pan XOR tilt)
```

| API | 说明 |
|---|---|
| `list_ports()` | 串口名称列表 |
| `connect(port,baudrate=115200)` / `disconnect()` | 连接/释放设备与TX/RX线程 |
| `connected` | 是否保持打开的serial对象 |
| `command(pan,tilt)` | 更新最新目标并返回夹紧后的整数二元组；不等待实际到位 |
| `set_tx_rate_hz(hz)` | 限制到1–30 Hz，默认12 |
| `command_snapshot()` | `{pan_cmd_deg,tilt_cmd_deg}`；本次连接无命令或断开时两值None |
| `feedback_snapshot(stale_after_s=1.0)` | 编码器pan/tilt、各轴年龄、online、接收文本/十六进制与error |
| `tx_snapshot()` | target_pan/tilt、last_pan/tilt、pending、tx_rate_hz、last_tx_age_s、error |
| `start_jog(pan_direction=0,tilt_direction=0,step=1,speed=20)` | 方向-1/0/+1，立刻一步，延迟260ms后每50ms改变目标 |
| `stop_jog()` | 停止继续改变目标；已提交的最后目标仍由TX发送 |
| `set_home(pan,tilt)` | 保存Home，不运动 |
| `home()` | 停止jog后发送Home |

Pan范围0–255，Tilt范围9–171，沿用设备协议。`pan`/`tilt` 是当前请求目标；`last_pan`/`last_tilt` 是已发出目标；两者都不是 Encoder。Cmd快照是目标采样，不代表物理角度或精确曝光同步。

## Qt 组件

| 导入路径 | 接口 |
|---|---|
| `apriltag_sdk.qt.bridge.QtSystemBridge` | `(system,parent=None,interval_ms=33)`；start/stop；frame_ready/result_ready/rgb_options/status/error 五类 Qt 信号；前三类可能发None清空状态 |
| `apriltag_sdk.qt.image.bgr_to_qimage` | BGR→拥有像素内存的QImage |
| `apriltag_sdk.qt.views.FrameView3D` | QWidget；set_frames(dict或None,message="")、set_view(yaw,pitch,zoom)、view_state()、view_changed信号 |
| `apriltag_sdk.qt.calibration_editor.CalibrationPanel` | QGroupBox；传QSettings；restore()、bind_locator(locator)、chain_changed(CalibrationChain) |
| `apriltag_sdk.qt.camera_options.RGBOptionsPanel` | QGroupBox；传QSettings；update_snapshot(dict)、reset()、requested(dict)信号；minimum_generation可过滤视觉旧快照 |
| `apriltag_sdk.qt.app.MainWindow` | 完整上位机QMainWindow；settings_path可选；results_updated(dict/None)、result_snapshot(max_age_s=1.0)；用close()释放线程 |

QtSystemBridge 仅负责GUI轮询与信号，不拥有设备生命周期。它的 `stop()` 不会关闭 AprilTagSystem。全部QWidget/QTimer创建与使用都在Qt GUI线程。

## 底层组件

- `calibration.CalibrationDocument` / `CalibrationChain`：延续既有模型、数值校验与原值/编辑副本。
- `camera.discover_color_profiles(serial=None)`：返回设备名、序列号、profile列表；每项 `(width,height,fps,rs.format)`，按分辨率/帧率降序。
- `camera.CameraCaptureWorker`：Python线程相机采集；要求camera额外依赖。
- `workers.DetectionWorker`：Python线程，消费LatestFrameBuffer，原始结果字段用米；其结果不包含Base变换。通常用Locator/System更方便。
- `qt.workers`：上述工作线程的QThread适配器，供完整上位机使用。
- `camera_options.SensorOptions` / `RGBOptionsWorker`：硬件参数读取、串行异步写入与结果快照。
- `drawing.annotate(image_bgr,overlays,width=None)`：返回新BGR数组，不改输入。

底层原始结果里的 `x_m`、`target_x_m` 等字段是**米**，不要与高层 Target 的毫米字段混用。
