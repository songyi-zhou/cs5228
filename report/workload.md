# CS5228 项目并行工作流（4 人）

## 并行原则

四人从第一周同时开始，各自维护独立分支和独立输出；通过统一的数据接口、固定验证集和每周一次合并节点保持一致。

```text
                    ┌────────── A：数据治理与 EDA ──────────┐
原始 train/test ────┼────────── B：核心特征工程 ────────────┼──→ 统一特征接口
                    ├────────── C：基线/主模型/验证 ────────┤
                    └────────── D：外部数据/空间/第二模型 ──┘
                                                ↓
                                  固定时间验证 + 消融实验
                                                ↓
                                      最终模型 / 集成 / 报告
```

---

## 第 0 周：半天启动会

### 全员共同确定

- 主验证切分：

```text
训练：2021-01 ～ 2024-06
验证：2024-07 ～ 2025-03
```

- 统一指标：以 Kaggle 指标为主，同时记录 MAE 与 RMSE。
- 统一输入/输出约定：
  - 原始数据：`data/raw/`
  - 清洗数据：`data/processed/`
  - 特征表：`data/features/`
  - 所有实验记录：`outputs/metrics/experiment_log.csv`
- 每个实验必须写明：
  - 数据版本；
  - 特征版本；
  - 验证切分；
  - 模型参数；
  - 验证分数；
  - 结论。

### 共同接口

```python
# 所有成员均使用的基本接口
train_df = load_train_data()
test_df = load_test_data()

X_train, y_train = build_features(train_df, feature_set="v1")
X_test = build_features(test_df, feature_set="v1")

metrics = evaluate(model, X_train, y_train, validation_scheme="time_v1")
```

---

## 工作流 A：数据治理与 EDA

**负责人：A**
**预计投入：40–45 小时**
**可从第 1 天独立开始**

### 第 1 周

- 数据审计：字段、数据类型、缺失、重复、异常和 train/test 差异。
- 统一文本格式：
  - `3 room` → `3-room`
  - 统一大小写、空格、town/street/block 格式。
- 删除常数列：`FURNISHED`、`FEE`。
- 输出第一版清洗函数，即使后续还会迭代。
- 同时完成 EDA：
  - 租金分布；
  - 月度租金趋势；
  - town / 房型 / 面积 / 房龄与租金关系；
  - train/test 分布差异。

### 第 2～3 周

- 根据 B/C/D 的反馈补充诊断图。
- 为误差分析提供按时间、town、房型、面积的分组框架。
- 审核所有数据处理步骤的正确性。

### 持续交付

```text
src/data_cleaning.py
notebooks/01_eda.ipynb
outputs/figures/eda_*.png
docs/data_dictionary.md
```

### A 与其他人的接口

- 第 3 天前：提供 `clean_data()` 初版；
- 第 1 周末：提供 `processed_train.csv`、`processed_test.csv`；
- 后续修改字段时必须提前通知 B/C/D。

---

## 工作流 B：核心特征工程

**负责人：B**
**预计投入：40–45 小时**
**第 1 天即可开始，不等 A 完成**

### 第 1 周：先用原始数据开发

- 开发独立的特征函数，不依赖 A 最终清洗结果：
  - 时间：year、month、quarter、连续月份；
  - 房龄：`approval_year - lease_commence_year`；
  - 面积：分桶、面积与房型交互；
  - 地址：town、street、block 的频率编码或类别特征标记。
- 与 C 确认 CatBoost 对类别变量的输入格式。
- 先提交 feature schema，而不是等待最终数据。

### 第 2 周：接入 A 的清洗数据

- 将特征函数接到 A 的 `clean_data()` 输出。
- 实施不泄漏的地点统计特征：
  - 必须只基于训练期历史数据；
  - 验证期不得使用自己的真实租金；
  - target encoding 必须 out-of-fold。
- 形成 V1、V2、V3 特征组。

### 第 3～4 周

- 与 C 联合做核心特征消融。
- 将真正有稳定收益的特征写入最终流水线。

### 持续交付

```text
src/features.py
configs/feature_sets.yaml
outputs/metrics/core_feature_ablation.csv
```

### B 与其他人的接口

- 第 3 天前：给 C 一份基础特征 schema；
- 第 2 周末：给 C 可运行的 `feature_set=v1/v2/v3`；
- 第 3 周：将基础地址特征的边界交给 D，避免重复做空间特征。

---

## 工作流 C：基线、主模型与时间验证

**负责人：C**
**预计投入：45–50 小时**
**第 1 天即可开始**

### 第 1 周：建立独立的模型骨架

即使还没有清洗和高级特征，也应立刻开始。

- 编写统一评估函数和时间切分函数。
- 以原始字段训练：
  - 全局/分组中位数 baseline；
  - Ridge baseline；
  - CatBoost baseline。
- 固定随机种子、指标和实验日志格式。
- 生成第一份合法 submission。

### 第 2 周：接入 B 的特征版本

- 对 V1/V2/V3 特征集运行完全相同的验证。
- 调整 CatBoost：
  - depth；
  - learning rate；
  - iterations；
  - regularization；
  - early stopping。
- 只保留有证据支持的参数与特征变更。

### 第 3～4 周：模型收敛

- 完成滚动时间验证。
- 选择最佳 CatBoost 方案。
- 把结果交给 D，供第二模型与 ensemble 比较。

### 持续交付

```text
src/train.py
src/evaluate.py
src/predict.py
configs/models/catboost.yaml
outputs/metrics/experiment_log.csv
outputs/submissions/baseline_submission.csv
```

### C 与其他人的接口

- 第 2 天前：给全员统一 `evaluate()` 和时间验证定义；
- 每周更新最佳 benchmark；
- D 的模型必须使用 C 定义的相同验证集才能比较。

---

## 工作流 D：外部数据、空间特征与第二模型

**负责人：D**
**预计投入：45–52 小时**
**第 1 天即可开始，不依赖 B 完成**

### 第 1 周：外部数据可行性与空间管线

- 检查辅助数据中哪些表可直接使用。
- 建立 block 与经纬度的匹配策略。
- 建立距离计算函数：
  - block → 最近 MRT；
  - block → 最近商场；
  - block → 最近小学；
  - block → CBD。
- 记录每张辅助表的数据来源、时间范围、匹配率和潜在风险。

### 第 2 周：空间特征和辅助变量

- 生成 POI 距离与密度特征。
- 合并宏观月度数据，例如 COE。
- 构建空间簇或经纬度区域特征。
- 交付独立的 `spatial_feature_set=v1`，可由 C 直接载入。

### 第 3 周：第二模型

- 训练 LightGBM 或 XGBoost。
- 使用与 C 完全相同的时间验证。
- 输出与 CatBoost 的误差对比，以及预测残差相关性。

### 第 4 周：集成与空间误差

- 只在多折验证都有稳定收益时做集成。
- 分析误差在 town、空间簇、MRT 距离、租金区间上的变化。
- 输出空间特征的消融表。

### 持续交付

```text
src/external_data.py
src/geospatial_features.py
src/train_lgbm.py
src/ensemble.py
outputs/metrics/spatial_feature_ablation.csv
outputs/metrics/ensemble_results.csv
```

### D 与其他人的接口

- 第 1 周末：提供辅助数据可用性报告；
- 第 2 周末：提供空间特征表；
- 第 3 周末：提供第二模型结果；
- 第 4 周：与 C 确定是否使用 ensemble。

---

## 并行时间线

| 周次 | A：数据与 EDA | B：核心特征 | C：模型与验证 | D：空间与第二模型 | 合并节点 |
|---|---|---|---|---|---|
| 第 1 周 | 审计、清洗初版、EDA | 基础特征原型 | baseline、时间验证 | 外部数据可行性、距离管线 | 周末：统一数据接口与 benchmark |
| 第 2 周 | 清洗冻结、补充 EDA | 特征 V1–V3 | CatBoost 初步调参 | POI/宏观/空间特征 V1 | 周末：核心与空间特征首次对比 |
| 第 3 周 | 分布/异常诊断 | 地址统计特征、消融 | 滚动验证、主模型 | LightGBM/XGBoost | 周末：模型与特征筛选 |
| 第 4 周 | 最终图表初稿 | 特征固化 | 最佳 CatBoost | ensemble、空间误差 | Progress Report |
| 第 5 周 | 支持错误案例解释 | 最终特征复核 | 全量训练候选 | 鲁棒性和集成确认 | 最终方案冻结 |
| 第 6 周 | 数据与图表核查 | 方法段初稿 | 最终 submission | 空间结果与限制 | 报告、复现、提交 |

---

## 每周合并节点

### 周二：15 分钟接口同步

- A：清洗字段是否变化；
- B：新增特征名称与版本；
- C：当前最佳分数与验证规则；
- D：辅助数据匹配率与空间特征状态。

### 周五：45 分钟实验决策会

每人只报告四项：

1. 本周完成什么；
2. 最佳/最差实验结果；
3. 发现了什么；
4. 下周最值得投入的一个方向。

会议结束必须确定：

```text
下周保留的特征版本：
下周主模型：
下周停止的实验：
负责人：
截止日期：
```

---

## 并行时必须遵守的边界

| 风险 | 规则 |
|---|---|
| 数据版本不一致 | A 发布带版本号的清洗数据；其他人只使用已发布版本。 |
| 指标无法比较 | C 维护唯一的时间验证函数；所有模型统一使用它。 |
| 地址与空间特征重复 | B 负责文本地址编码；D 负责经纬度、POI、空间聚类。 |
| 目标泄漏 | B/D 的所有租金统计特征必须经过 C 或 D 的时间泄漏检查。 |
| 重复调参 | C 管理统一实验日志；每次新实验先登记。 |
| 报告最后堆积 | 每人每周同步提交自己的图、表和一句结论到 `report/sections/`。 |
```

最关键的并行设计是：C 不等待最终特征才开始建模，D 不等待 B 才开始空间工程，B 不等待 A 完成全部 EDA 才开发特征。第一周结束时，四条线都应有可运行的初版产出；随后只在统一验证框架中竞争和合并。
