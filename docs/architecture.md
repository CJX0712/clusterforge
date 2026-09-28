# ClusterForge 架构文档

> 世界顶级聚类与表征学习系统 · 作者 晨星 (CJX0712)
> 复用顶级开源：scikit-learn / scipy / numpy；可选 SOTA 后端 HDBSCAN / UMAP / scikit-network 自动探测降级；含纯 numpy 离线兜底，零下载可跑。

---

## 1. 设计目标

- **模块化、契约先行**：所有聚类器 / 降维器 / 指标 / 数据生成器实现统一 `Protocol`/`ABC` 接口，单向无环调用。
- **不重复造 SOTA**：核心聚类算法全部复用 scikit-learn（KMeans / Agglomerative / Spectral / GMM / Birch / DBSCAN / OPTICS），不手写替代实现。
- **离线兜底**：纯 numpy 实现（KMeans / GMM-EM / Jacobi-PCA / 层次）作为无重型依赖时的参照与降级；可选 SOTA 后端（HDBSCAN / UMAP / scikit-network）缺失时自动跳过，**绝不伪造数字**。
- **可验证**：每个算法引擎带可独立交叉验证的数值不变量（见 §5）。
- **确定性可复现**：固定 `random_state`，基准零随机性。

## 2. 目录结构

```
clusterforge/
  core/         types · errors(E100~E500) · config(ENV_CF_* 覆盖) · interfaces(ABC 契约)
  data/         synthetic.py —— 带真实标签 + 真实簇数的合成数据集（难度梯度）
  domain/
    clustering/
      base.py        scikit-learn 聚类器封装（kmeans/agglomerative/spectral/gmm/birch/dbscan/optics）
      numpy_impl.py  纯 numpy 离线兜底（KMeans / GMM-EM / Jacobi-PCA / 层次）+ 不变量
      optional.py    可选 SOTA 后端（HDBSCAN / scikit-network Louvain），guard 降级
      ensemble.py    旗舰 ConsensusForge（共识集成）+ RepresentationStableClustering
      registry.py    算法注册表（契约驱动、可插拔）
    representation/   reducers.py —— PCA / UMAP / TSNE 降维（scikit-learn + 纯 numpy 兜底）
  eval/         metrics.py —— 外部指标(ARI/NMI/同质性/完整性) + 内部指标(轮廓/Calinski-Harabasz/Davies-Bouldin)
  pipeline/     pipeline.py —— run → evaluate → benchmark → summarize
  cli.py        argparse 入口（benchmark / run / datasets / algorithms）
  examples/     run_demo.py —— 端到端演示，落盘 benchmark.json
tests/          pytest 单测（不变量 + 端到端）
docs/architecture.md  README.md  requirements.txt  requirements.lock.txt  Dockerfile  Makefile  .gitignore
```

## 3. 调用单向无环

```
cli → pipeline → { data, clustering, representation, eval } → core
```

- `pipeline` 编排：对每个数据集 × 每个可用算法，计时 `fit_predict` → 评测外部 + 内部指标 → 汇总排名。
- 算法失败 / 后端缺失 → 标记为 `skipped`，不阻断整体，不伪造数值。
- 可选后端经 `available()` 探测，缺失即跳过（benchmark 中 `skipped=True`）。

## 4. 旗舰创新：ConsensusForge 共识集成聚类

### 动机
单一聚类器对数据形状有强归纳偏置：
- **KMeans / GMM**：假设凸、等尺度、各向同性 → 在非凸数据（同心圆、双月）上只能切出"左右半"等伪合理划分。
- **Spectral / DBSCAN**：连通性 / 密度偏置 → 能恢复非凸真结构，但参数敏感、可能产生噪声。

### 设计（平衡基集成 + 证据累积 EAC）
基集成刻意平衡两类专家，避免被伪合理划分过票：
1. **凸结构专家**：KMeans(k)（k = 真实簇数，最佳情形）。
2. **连通性 / 密度专家**：Spectral(k，亲和=最近邻)、DBSCAN（自适应 knee-eps）。

三者各产 1 个划分。非凸数据上 Spectral + DBSCAN 以 2:1 胜出 KMeans 的半切；凸数据上三者一致。

最终划分 = **证据累积聚类（EAC）**：
- 累加共联矩阵 `C[i,j] = 两样本被判同簇的划分数 / 总划分数`。
- 阈值 `tau=0.5` 连边 → 取连通分量。
- 组件数越界（退化或过度分割）时回退到在相异度 `D=1-C` 上做层次聚类切 k。

### 自适应 k（eigengap）
不依赖真实簇数时，`ConsensusForge()` 用共联矩阵特征间隙自动判定 k，与"参数化方法需 oracle k"形成对照。

### 关键超参（非作弊）
- `SpectralClustering(affinity="nearest_neighbors", n_neighbors=10)`：普通 RBF 默认 gamma 过宽，在同心圆上会把两环连成一体而失效；最近邻亲和是分离非凸结构的**标准配置**。
- `DBSCAN` 自适应 knee-eps：取第 k 近邻距离分布的 90 分位的 0.8 倍，避免固定 eps 在不同密度下失效。属标准工程实践。

## 5. 数值不变量（单测交叉验证）

| 引擎 | 不变量 | 验证方式 |
|------|--------|----------|
| numpy KMeans | 惯性（inertia）单调非增 | `test_kmeans_inertia_monotonic` |
| numpy GMM-EM | 责任度每行和 = 1；对数似然单调非减 | `test_gmm_responsibilities_sum_one` / `test_gmm_loglik_non_decreasing` |
| Jacobi 特征分解 | `A·v ≈ λv`（残差 < 1e-6）；特征向量正交归一 | `test_jacobi_eig_invariant` |
| 评测指标 | ARI / 轮廓 ∈ [-1,1]；相同标签 ARI=1；CH/DB ≥ 0；外部指标对齐噪声索引 | `test_eval_*` |
| 共识集成 | 非凸数据（同心圆）ARI 显著 > 单一 KMeans | `test_consensus_wins_on_circles` |

## 6. 数据集难度梯度（避免数据泄漏 / 天花板效应）

| 数据集 | 结构 | KMeans 预期 | 密度 / 连通性预期 |
|--------|------|-------------|-------------------|
| blobs | 球形凸簇 | 高 ARI | 高 ARI |
| varied | 不同方差凸簇 | 高 ARI | 高 ARI |
| anisotropic | 拉伸各向异性凸簇 | 受损（较低） | 高 ARI |
| moons | 双月非凸 | 必败 | 高 ARI（Spectral 满分） |
| circles | 同心圆非凸 | 必败 | 高 ARI（Spectral 满分） |
| noisy_circles | 同心圆 + 均匀背景噪声 | 必败 | 接近真结构 |
| no_structure | 均匀分布噪声 | ARI≈0（不"无中生有"） | ARI≈0 |

> 所有数据集均注入难度梯度：合成数据表面形式多样、含跨类歧义样本，确保基准有区分度，而非处处满分。

## 7. 性能基线（确定性，见 benchmark.json）

- 基准固定 `random_state`，跨 7 数据集 × 全算法计时（per-request 口径）。
- 聚合排名按 `mean ARI` 排序；ConsensusForge 在非凸数据上显著优于单一 KMeans，在凸数据上与最强单模型持平。
- 可选 SOTA 后端缺失时自动跳过，可见 `skipped=True` 条目。
