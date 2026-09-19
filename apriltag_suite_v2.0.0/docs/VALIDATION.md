# 验证记录 · v2.0.0

验证环境为 Linux x86_64。GUI使用 `QT_QPA_PLATFORM=offscreen`；没有连接真实D435、云台或机械臂，不宣称完成硬件精度/运动验收。

## 自动化检查

| 项目 | Python 3.8.20 | Python 3.14.6 |
|---|---|---|
| 回归和SDK检查 | 33 passed | 33 passed |
| 原上位机16项回归 | 通过 | 通过 |
| 新SDK、Qt绑定及宿主接口17项检查 | 通过 | 通过 |

已覆盖：

- 独立矩阵公式校验Legacy变换、每ID偏移、Camera/Base单位。
- 合成AprilTag真实检测、PnP、角点、重投影误差与空帧清空目标。
- 标定原值/编辑副本、导出不覆盖、失效选择停用Base、来源字段不限制组合。
- 用户提供的两份标定文件可计算Base/3D链，包内参考文件与原文件逐字节相同。
- RGB自动/手动参数依赖、范围、异步写入、按序列号恢复、旧快照成功值保存。
- 采集/慢识别解耦、最新帧消费、异步退出、重启、相机异常、启动前停止。
- 串口10字节包、GBK任意分割解析、Cmd与Encoder分离、短按/长按、Home。
- 核心导入无Qt/RealSense依赖；公共结果时效、快照内存隔离、JSON配置恢复。
- PyQt6完整窗口结果接口、标定面板绑定Locator、Qt桥重复帧抑制/停机清空、QImage像素所有权。

串口、RGB与连续采集相关自动化使用假设备或模拟帧；不能证明真实USB/串口硬件表现。

## 依赖版本

| 依赖 | Python 3.8.20 | Python 3.14.6 |
|---|---|---|
| PyQt6 | 6.7.1 | 6.11.0 |
| Qt运行库 | 6.7.3 | 6.11.2 |
| PyQt6-sip | 13.8.0 | 13.12.0 |
| NumPy | 1.24.4 | 2.5.3 |
| opencv-contrib-python-headless | 4.10.0.84 | 5.0.0.93 |
| pyrealsense2 | 2.55.1.6486 | 2.58.4.10922 |
| pyserial | 3.5 | 3.5 |
| pytest（仅测试） | 8.3.5 | 9.1.1 |

Python3.14.7也完成过30项阶段性检查；最终交付以与用户一致的3.14.6为主要验证环境。

两组固定依赖均已使用 `pip download --platform win_amd64 --python-version ... --only-binary=:all: --no-deps` 实际下载对应Windows wheel，验证这些版本存在预编译分发包。这不是在Windows上运行了应用，也不是离线安装包交付。

## 复现检查

按README安装对应依赖与SDK后：

```bash
python -m pip install pytest
python -m pytest -q
python examples/external_image.py --synthetic --output preview.png
python examples/pyqt6_host.py --simulate --smoke-test
```

桌面环境直接执行；无显示的Linux环境在命令前设置 `QT_QPA_PLATFORM=offscreen`。测试不需要真实相机，但回归界面模块需要安装Qt与camera依赖。

## 本机设备验收

请重点确认D435高分辨率profile、RGB调参、静止姿态定位与原版一致、真实Cmd/Encoder反馈、按键释放、Home及反复停止/重新打开设备。README与接入文档给出了操作顺序和坐标前提。

本次软件迁移不改变两份标定文件的几何正确性，也不要求文件名配套；结果精度仍依赖用户已确认的标定、实际Tag边长、安装姿态与设备状态。
