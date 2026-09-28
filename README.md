# ClusterForge

> 世界顶级聚类与表征学习系统 · 作者：晨星 (CJX0712)

ClusterForge 复用顶级开源（scikit-learn / scipy / numpy），在**零下载可跑**的前提下，提供从经典算法到共识集成旗舰的完整聚类流水线，并内置**纯 numpy 离线兜底**与**确定性可复现基准**。

## 核心特性

- **15+ 算法 / 4 档分层**：sklearn 系（KMeans / Agglomerative / Spectral / GMM / Birch / DBSCAN / OPTICS）、纯 numpy 兜底（KMeans / GMM-EM / Jacobi-PCA）、可选 SOTA 后端（HDBSCAN / UMAP / scikit-network，自动探测降级）、旗舰 **ConsensusForge**。
- **旗舰 ConsensusForge**：平衡基集成（KMeans + Spectral + DBSCAN）+ 证据累积聚类（EAC）共联矩阵，阈值 τ=0.5，连通分量切分；非凸数据显著优于单一 KMeans。
- **纯 numpy 离线兜底**：k-means++ 初始化 + Lloyd 迭代、GMM-EM、Jacobi 旋转特征分解 PCA — 不依赖任何第三方聚类库即可运行。
- **7 档难度梯度合成数据集**：blobs / varied / anisotropic（凸）→ moons / circles / noisy_circles（非凸）→ no_structure（无结构，证明系统不会"无中生有"）。
- **确定性可复现基准**：固定 `random_state=42` 与种子组 (42/123/7)，跨算法跨数据集评测，落盘 `benchmark.json`。
- **数值不变量自检**：31 项单测覆盖惯性单调不增、GMM 责任度行和=1、Jacobi `A·v≈λv`、ARI/轮廓 ∈ [-1,1] 等硬金标准。

## 安装

```bash
pip install -r requirements.txt   # numpy / scipy / scikit-learn
# 可选 SOTA 后端（缺失则自动跳过）
pip install hdbscan umap-learn scikit-network
```

## 快速开始

```python
from clusterforge.core.config import Config
from clusterforge.pipeline.pipeline import ClusterPipeline

cfg = Config.from_env()
pipe = ClusterPipeline(cfg)
rows = pipe.benchmark(seeds=cfg.benchmark_seeds)
summary = pipe.summarize(rows)
print(summary["best_algorithm"], summary["ranking"][:3])
```

命令行：

```bash
clusterforge benchmark            # 跨算法基准 -> benchmark.json
clusterforge run --dataset moons  # 单数据集跑全部算法
clusterforge datasets             # 列出 7 档合成数据集
clusterforge algorithms           # 列出全部可用算法
```

端到端演示（生成数据 → 跨模型基准 → 打印表 + 落盘）：

```bash
python -m clusterforge.examples.run_demo
```

## 基准结果（摘要，v0.1.0，21 评估点 / 273 条）

| 排名 | 算法 | mean ARI | min ARI | mean(s) |
|----:|------|---------:|--------:|--------:|
| 1 | spectral | 0.722 | -0.002 | 0.011 |
| 2 | **consensus-forge** | **0.671** | -0.002 | 0.206 |
| 3 | dbscan | 0.473 | -0.001 | 0.005 |
| 4 | gmm | 0.462 | -0.002 | 0.032 |
| 5 | agglomerative | 0.436 | -0.003 | 0.008 |
| … | kmeans / birch / numpy-* | 0.32–0.42 | — | — |
| 13 | numpy-gmm | 0.000 | 0.000 | 0.015 |

> 非凸结构（moons / circles）上 ConsensusForge 显著领先 KMeans 系：circles ARI≈+0.75、moons ARI=+1.00、blobs ARI≈+0.996。

## 架构

```
clusterforge/
├── core/          Config / Dataset / ClusterResult / EvalRow / 错误码
├── data/          synthetic.py  7 档难度梯度合成数据集
├── domain/
│   ├── clustering/  base(sklearn) · numpy_impl(离线兜底) · optional(SOTA) · ensemble(旗舰)
│   └── representation/  PCA / UMAP / t-SNE 降维
├── eval/          metrics.py  外部(ARI/NMI/同质性) + 内部(轮廓/CH/DB)
├── pipeline/      pipeline.py  基准编排 + 聚合排名
└── examples/      run_demo.py  端到端演示
```

详见 [docs/architecture.md](docs/architecture.md)。

## 作者

晨星 (CJX0712) — 随机创新世界顶级 AI 系统系列之 ClusterForge。

## 许可

MIT
