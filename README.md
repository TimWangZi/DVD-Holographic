# DVD-Holographic

DVD-R 正四面体全息实验。

默认介质现为普通单层 **DVD-R**，输出可直接烧录的 ISO9660 镜像。目标仍是三维正四面体的六条棱，跳过前置光栅阶段。已实现主机数据到 DVD 记录结构的参考模型，并保留旧 CD-R 配置和结果。

当前结果是软件仿真与可烧录数据；DVD-R 实盘是否重建成功仍需测试。记录态折射率、标记边缘和盘片几何使用可修改的先验。

## 先标定盘片几何

已加入 [标定程序与操作说明](docs/CALIBRATION.zh-CN.md)。运行 `calibration-build` 生成本地标定镜像 `output/calibration_dvd/dvd_calibration.iso`，用它烧录九个已知位置的圆环和两条连续辐条，拍摄完整盘面，再取点拟合。程序输出新的配置，不会把名义参数或合成数据伪装成实测值，也不会从盘面照片估计复折射率。

```powershell
python -m cdholo calibration-build
python -m cdholo calibration-measure --image photo.jpg --output measurements.csv --diameter-mm 120
python -m cdholo calibration-fit --measurements measurements.csv --mode refine --output output/calibration_fit
python -m cdholo build --config output/calibration_fit/calibrated_config.json --output output/tetrahedron_dvd_calibrated
```

圆环外缘从内向外为 R01–R09；S01 有一个侧点、S02 有两个侧点。外缘和中心孔各点至少 8 点，每环至少 6 点，两条辐条都沿内到外连续点选至少 8 点。默认通过这两条不对称辐条判断螺旋方向；照片必须是数据面正视且未镜像。实际盘径已测量时用 `--diameter-mm` 替换 120。若辐条扭曲得难以连续识别，先用 `--mode coarse` 拟合圆环、重刻标定盘，再精标定。完整键鼠操作、照片要求和误差判断见上述说明。

## 使用当前结果

生成镜像、仿真图片和校验记录不纳入 Git 仓库；克隆后按下方运行说明生成。以下数值对应默认配置的一次本地运行。

- 镜像：`output/tetrahedron_dvd/tetrahedron_dvd.iso`，553,058,304 字节，270,048 个逻辑扇区。
- [DVD-R 刻录与光学测试说明](docs/DVD-R.zh-CN.md)
- [完整参数配置](configs/tetrahedron_dvd.json)
- 仿真预览：`output/tetrahedron_dvd/overview.png`
- 配置与校验记录：`output/tetrahedron_dvd/manifest.json`

ImgBurn 选择 **Write image file to disc**，打开 ISO，用空白单层 DVD-R、DAO、封盘并启用 Verify。不要把 ISO 当普通文件写入另一张数据盘，也不要提取其中的文件再重建镜像。镜像包含 `README.TXT` 与固定位置的 `HOLOGRAM.BIN`；光学图样依赖原扇区位置。

## 运行

使用 Python 3.10 或更新版本，在项目目录执行：

```powershell
python -m pip install -r requirements.txt
python -m cdholo build
python -m unittest discover -s tests -v
```

默认使用 `configs/tetrahedron_dvd.json`，输出到 `output/tetrahedron_dvd`。也可以显式指定：

```powershell
python -m cdholo build --config configs/tetrahedron_dvd.json --output output/my_dvd
```

同一输出目录的同名文件会更新。改变参数时换一个输出目录，便于比较。生成只写本地文件，不操作光驱。

## DVD 写入模型

主机 ISO 的 2048 字节扇区经过 DVD-R ID/IED、EDC、主数据扰码、PO/PI 纠错、PO 行交织、EFMPlus、同步和 NRZI。每 16 扇区一个 ECC 块，每个记录扇区 38,688 通道位。软件反向抵消扰码，使指定数据字节到达目标空间位置；ECC 等物理编码仍由刻录机执行。

默认候选字节为 `0x93/0xA5`，按 EFMPlus 状态平均跳变密度投影二值目标。两者已检查所有起始状态的重复字节循环，没有持续直流漂移。这是二值投影和编码补偿，尚不是迭代求解全盘光场的全局优化器。

全窗口预测包含实际镜像的 ID/IED/EDC、扰码、PI/PO 和交织结果。光学包络使用状态平均 EFMPlus 跳变密度，以 `0.5 - trim_T × transition_density` 估计沿轨标记覆盖率，再乘标记宽度/轨距；这是粗尺度有效面积模型。另有一个完整 ECC 块的通道样本，采用参考双流 DSV 编码并检查 3T–11T 数据游程和 14T 同步游程。完整样本不等于全窗口逐位传播，也不等于捕获了设备的实际固件输出。

## 可调参数

局部坐标为切向 x、径向向外 y、反射侧实像距离 z；旋转依次为右手 X/Y/Z 旋转。

| 配置组 | 参数及单位 | 默认值 |
| --- | --- | --- |
| `disc` | 数据起点半径 mm、轨距 μm、通道位长 nm、起始角 °、通道时钟偏移、螺旋方向、容量及尾部余量扇区、背景字节 | 24 mm、0.74 μm、133.3 nm |
| `patch` | 中心半径、切向/径向尺寸 mm、方位角 ° | 28 mm、4 × 4 mm |
| `target` | 棱长、距离、偏移 mm、旋转 °、每棱点数、随机相位及种子 | 棱长 4 mm、距离 300 mm、偏移 `[0,5]` mm |
| `optics` | 波长 nm、照明方向角 °、光斑类型、半径和中心 mm | 632.8 nm、法向、高斯强度直径 2 mm |
| `media` | 各层 `[n,k]`、厚度 nm、偏振、薄膜内部角度 °、标记宽度和每端收缩 μm、直接复反射覆盖 | 未记录染料 `2.3+0.05i`、70 nm；收缩 0.025 μm |
| `optimization` | 候选字节、二值占比、反转、扰码补偿开关 | `[165,147]`、占比 0.5、开启补偿 |
| `simulation` | 网格 μm、重建图像像素及范围、焦面偏移 mm、批量大小、通道样本扇区数、初始状态/极性/DSV | `[2.5,0.74]` μm、320 像素/5 mm、16 扇区样本 |

`beam_radius_mm` 为 1/e² 强度半径。通道位长不随刻录倍速乘倍。外部照明方向用两个方向余弦的反正弦表示；薄膜内部角度独立配置，倾斜照明时应按折射关系一并调整。`n+ik` 使用被动材料约定；`reflection_mode=direct` 时改用 `[实部,虚部]` 形式的 `land_r`、`mark_r`。改变波长时也要检查材料色散。

格式常量如 2048 字节/扇区和数据起点 PSN `0x030000` 按 DVD-R 标准固定。当前仅支持单层 DVD-R，不套用于 DVD+R、DVD-RW 或双层盘。

## 验证与输出

27 项测试覆盖原 CD 链路和新 DVD 链路，包括独立 RS/EDC 校验、扰码分组、PI/PO 行列校验、完整 EFMPlus 解码、ISO9660、载荷位置和光学传播。默认生成时 8,583,926 个受控字节经完整扇区编码后为零错位。详细结果以本次 `manifest.json` 为准。

输出包含 ISO、参数快照、SHA-256、三维目标与重建对照、多焦面预览、`simulation.npz` 数组以及 `channel_sample.npz` 通道位/标记游程。`edge_trim_um=0` 的对照保留在预览中，显示输入相关的面积反差消失后剩余的孔径背景。仿真指标不是实盘效率。

## 保留的 CD-R 模式

```powershell
python -m cdholo build --config configs/tetrahedron.json --output output/tetrahedron
```

旧模式输出 WAV/CUE，使用 CIRC/EFM；不能用于本次 DVD-R 刻录。详见 [CD-R 原说明](docs/CD-R-README.md)和 [CD-R 刻录说明](docs/BURNING.zh-CN.md)。

资料与实现依据见 [DVD-R 说明](docs/DVD-R.zh-CN.md)、[原论文核对](docs/physics-notes.md)及[上游接口审查](docs/upstream-audit.md)。核心实现独立编写，未复制 DiscImageStudio 的 GPL 代码。
