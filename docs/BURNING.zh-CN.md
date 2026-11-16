# 正四面体 CD-R 全息刻录与测试

把生成的 `tetrahedron.cue` 和 `tetrahedron.wav` 放在同一个目录，按 **音频 CD 镜像**刻录到空白 CD-R。WAV 是为控制记录结构生成的数据载体，不是音乐；不要用播放器试听。

## Windows 用 ImgBurn 刻录

1. 插入一张空白 CD-R，确认光驱支持 CD-R 刻录。
2. 打开 ImgBurn，选择 **Write image file to disc**（写入镜像文件到光盘）。在 Source 选择 `tetrahedron.cue`。不要选择“把文件/文件夹写入光盘”；那会生成数据盘，改变编码链路。
3. 确认显示 **一条 AUDIO 音轨**，文件为 `tetrahedron.wav`。CUE 中应为 `FILE "tetrahedron.wav" WAVE`、`TRACK 01 AUDIO`、`INDEX 01 00:00:00`。不手动追加两秒静音，不修改索引。
4. 在 Tools → Settings → Write 中选择 **SAO/DAO** 写入方式，完成后关闭光盘。用该刻录机和盘片实际支持的固定速度，例如列表中有 8× 就可先用 8×；记录请求速度和日志里的实际速度。不强制盘片不支持的 1×。
5. 保持原始 PCM，不经音乐编辑器转换，不做音量归一化、均衡、淡入淡出或重采样；不要转 MP3。点击 Write 完成刻录，保存日志。

入口和 CUE 用法依据 [ImgBurn 官方镜像刻录指南](https://forum.imgburn.com/topic/61-how-to-write-an-image-file-to-a-disc-using-imgburn/)及[官方音频 CD 指南](https://forum.imgburn.com/topic/5555-how-to-write-an-audio-cd-from-music-files-using-imgburn/)。SAO/DAO 设置见[官方设置说明](https://forum.imgburn.com/topic/6232-the-imgburn-settings/)。本项目没有在当前光驱上执行写盘。

配置中的 `disc.linear_velocity_m_s = 1.2` 是计算物理位置的 **1× 基准线速度**，不是要求用 1× 刻录。更改刻录速度不能在配置里把它简单乘以 8 或 16。

## 已安装命令行刻录工具时

在 WAV 所在目录运行；把示例设备和速度换成实际值。以下命令会真正开始写盘。

```text
cdrecord -scanbus
cdrecord dev=0,0,0 speed=8 -v -dao -audio tetrahedron.wav
```

WAV 已包含 44.1 kHz、双声道、16 位 PCM 格式信息，cdrecord 会识别字节序，**不要添加 `-swab`**。单轨文件的长度已经按 CD 扇区对齐，不需要音频转换。参数依据 [cdrecord 官方手册](https://cdrtools.sourceforge.net/private/man/cdrecord/cdrecord.1.html)。

若用 cdrdao，在音轨目录另存文本文件 `tetrahedron.toc`：

```text
CD_DA
TRACK AUDIO
FILE "tetrahedron.wav" 0
```

然后运行：

```text
cdrdao scanbus
cdrdao show-toc tetrahedron.toc
cdrdao write --device 0,0,0 --speed 8 tetrahedron.toc
```

此处使用 TOC，避免不同 cdrdao 版本对 WAVE CUE 支持的差异。格式和命令依据 [cdrdao 项目手册](https://github.com/cdrdao/cdrdao/blob/master/dao/cdrdao.man)。

## 默认光学测试布置

下表对应 `configs/tetrahedron.json`；重新生成后以输出目录中的配置快照和 manifest 为准。

| 项目 | 默认值与做法 |
| --- | --- |
| 光源 | 632.8 nm 相干光，准直后照射 CD 数据面 |
| 入射 | 接近法向入射；照明方向在配置里可修改 |
| 写入窗口 | 中心半径 28 mm，切向 × 径向为 4 × 4 mm |
| 查找窗口 | 在半径约 26–30 mm 环带缓慢旋转光盘；实际起始方位角未知 |
| 光斑 | 高斯光束，中心对准写入窗口中心；默认 1/e² 强度直径约 2 mm（半径 1 mm），窗口边缘受光较弱 |
| 成像面 | 反射一侧，距盘面约 300 mm；先用白屏或相机找实像 |
| 离轴位置 | 图像中心相对镜面反射点沿窗口径向外侧偏约 5 mm，切向偏移 0 mm |
| 物体 | 棱长 4 mm 的正四面体，六条棱均参与计算 |

法向布置中入射与反射光共路，可用分光镜分离并将观察屏放在返回光路；不要为避开光源而随意大角度倾斜盘片。若使用倾斜照明，需要把对应入射角写入配置后重新生成。观察时在约 298–302 mm 前后移动屏幕。默认有效孔径较小，景深可能覆盖大部分四面体，前后移动 2 mm 的清晰度变化未必明显；软件计算的是三维点源，单张屏幕图不能单独证明立体视差。均匀硬边照明会引入较强孔径衍射，建议先使用配置对应的高斯光斑。

镜面反射和沟槽衍射可能比目标像亮。遮挡零级时应保留距其约 5 mm 的设计像区域。使用屏幕或相机定位，避免眼睛接收直接光及镜面反射光。

## 首张盘要记录什么

- 盘片品牌、型号、批次，刻录机型号与固件，刻录软件版本及实际速度。
- 光源波长、盘到屏幕距离、照明角度和光斑直径。
- 整个盘面的照片、半径 26–30 mm 区域照片，以及镜面反射点附近含设计像区域的屏幕照片。保留曝光信息和尺标。
- 使用的配置快照、manifest 和刻录日志。

预测图是在模型假设下计算的，不是实盘照片。首版把标记边缘收缩 `edge_trim_um = 0.06` 作为未标定先验；它影响两种输入字节的有效光学对比度。若真实盘片没有足够对比度，正确刻录也可能只看到背景。实际轨距、起始半径和帧时序与配置不符，则会使全息条纹错位、弯曲或散焦。测试后可调整这些参数重新生成同一个四面体。

## 可选的音轨读回检查

用能保留原始样本的音频抓轨软件将 Track 01 读回为 44.1 kHz、双声道、16 位 PCM WAV。比较 WAV 的 PCM `data` 部分，先处理读写采样偏移和轨道边界，再做字节比较；WAV 文件头或整文件哈希不同本身不能证明音轨不同。不要做归一化或静音裁切。

读回一致只能确认主机音频数据被保存，不能读出固件实际选择的 EFM 合并位、微观记录标记形状或复反射率。普通光驱的 `RAW`/DAO 也不代表可以跳过 CIRC/EFM 直接写任意 pits。

项目提供偏移搜索和逐样本比较工具。在项目目录执行：

```powershell
python tools/compare_readback.py output/tetrahedron/tetrahedron.wav your_readback.wav
```

默认搜索 ±3000 个立体声采样的读写偏移。结果同时报告未比较的边缘采样数，重叠部分一致不等于轨道首尾也经过验证。
