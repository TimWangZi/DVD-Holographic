# DiscImageStudio 接口与编码边界复核

核查日期：2026-11-16。此项目独立生成可供普通刻录软件使用的 PCM WAV/CUE，借鉴方案中的链路划分；生成过程不调用 DiscImageStudio，可重复结果不依赖其后续版本变化。

## 固定版本

- 仓库：[JiaFeiMiao-K-Cat/DiscImageStudio](https://github.com/JiaFeiMiao-K-Cat/DiscImageStudio)。
- 核查提交：[`28310c8700646808bffa434feb2db68ab2c26a50`](https://github.com/JiaFeiMiao-K-Cat/DiscImageStudio/tree/28310c8700646808bffa434feb2db68ab2c26a50)。从仓库页面 `currentOid` 取得，并下载该提交归档核对代码。
- 归档：[固定提交 ZIP](https://codeload.github.com/JiaFeiMiao-K-Cat/DiscImageStudio/zip/28310c8700646808bffa434feb2db68ab2c26a50)。本次取得的 ZIP SHA-256 为 `84dd4a91717682f633b573bd5529f9e094c8e2496be59ba57c6a7ae1f424c0f1`。GitHub 未来重新打包时归档压缩字节可能变化，源码身份以提交为准。

## 实际接口

[`CdTrackGenerator.cs`](https://github.com/JiaFeiMiao-K-Cat/DiscImageStudio/blob/28310c8700646808bffa434feb2db68ab2c26a50/src/DiscImageStudio.Cd/CdTrackGenerator.cs) 的 `Generate` / `GenerateSamplesToStream` 路径依次进行图像采样、影调字节映射、可选数据位置预补偿和 PCM 输出。WAV 路径写入格式头并选择小端样本，RAW 路径选择大端；WAV 可附带 WAVE 类型 CUE。此处没有生成最后的 588T 物理通道帧，也没有将通道位传入光驱。

[`CddaInterleaveTable.cs`](https://github.com/JiaFeiMiao-K-Cat/DiscImageStudio/blob/28310c8700646808bffa434feb2db68ab2c26a50/src/DiscImageStudio.Cd/CddaInterleaveTable.cs) 定义 24 个数据位置、深度 4、112 帧缓冲及输出位置偏移。[`CddaInterleaver.cs`](https://github.com/JiaFeiMiao-K-Cat/DiscImageStudio/blob/28310c8700646808bffa434feb2db68ab2c26a50/src/DiscImageStudio.Cd/CddaInterleaver.cs) 将这些字节提前排列以补偿后续编码延迟，不计算 C1/C2 校验字节。它的 `Flush` 只补齐最后的音频扇区，不排空交织缓冲；因此不能把其有限音轨末尾直接当作完整目标图案。

[`CdDiscParameters.cs`](https://github.com/JiaFeiMiao-K-Cat/DiscImageStudio/blob/28310c8700646808bffa434feb2db68ab2c26a50/src/DiscImageStudio.Cd/CdDiscParameters.cs) 用扇区数、1× 基准线速度和内外半径映射螺旋位置。其几何字节索引属于主机 PCM 平均位置，不代表每个数据字节就是一块独立物理像素。完整物理映射还须计入校验、子码、同步和合并位。

官方[音轨兼容说明](https://github.com/JiaFeiMiao-K-Cat/DiscImageStudio/blob/28310c8700646808bffa434feb2db68ab2c26a50/docs/CD_AUDIO_COMPATIBILITY.md)确认 WAVE CUE 和 ImgBurn 的音轨入口，并明确预览只做文件字节映射，不能代替完整 CIRC/EFM 与实盘判定。[物理模型审计](https://github.com/JiaFeiMiao-K-Cat/DiscImageStudio/blob/28310c8700646808bffa434feb2db68ab2c26a50/docs/PHYSICAL_MODEL_AUDIT.md)也区分了合并位的固件自由度与子码传输格式。

## 此实现采用的边界

输入链路为：目标三维点集 → 二值复振幅全息条纹 → 螺旋数据符号采样 → CIRC 数据位置预补偿 → WAV PCM。烧录软件把 PCM 交给刻录机，后者继续生成 CIRC、子码、EFM、合并位和记录脉冲。软件内的正向编码器用于预测及验证，预测通道文件不是刻录输入。

CIRC 和 EFM 的标准依据为 [ECMA-130 第二版附录 C、D、E](https://www.ecma-international.org/wp-content/uploads/ECMA-130_2nd_edition_june_1996.pdf)。ECMA-130 是 CD-ROM 标准，此处引用其共同的物理编码说明；音频 PCM 不经过 CD-ROM Mode 1 的扇区头、扰码、EDC 和外层 ECC。CD-R 的介质背景见 [ECMA-394](https://www.ecma-international.org/wp-content/uploads/ECMA-394_1st_edition_december_2010.pdf)。

`RAW`、`RAW96` 或 DAO 常指主通道扇区及子通道的传输格式，不意味着主机可以绕过 CIRC/EFM 或逐位指定记录脉冲。本模型选用的合法合并位策略仍可能不同于具体光驱；写入热响应、记录态复反射率和实盘微米位置亦需实测验证。

上游源码只用于核查接口与行为，未将上述 GPL 文件复制或接入此项目的运行链路。
