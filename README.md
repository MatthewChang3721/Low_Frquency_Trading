# 美股低频量化系统 — 数据系统 MVP

对应《[系统设计与开发路线](美股低频量化系统设计与开发路线.md)》第 2 阶段（数据系统）的第一个可运行闭环。

**当前实现范围**：单标的（默认 `AAPL`）日频行情，从 **Yahoo Finance 下载 → 标准化 → 质量校验 → Hive 分区 Parquet → DuckDB 双重核验 → 原子发布 → 运行元数据**。

---

## 1. 快速开始

```powershell
# 1. 创建虚拟环境并安装运行时依赖
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .

# 2. 安装开发依赖（pytest / ruff）
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# 3. 运行完整管道
.\.venv\Scripts\python.exe -m data_sys.data_query --symbol AAPL --start 2024-01-02 --end 2024-03-29

# 4. 运行测试
.\.venv\Scripts\python.exe -m pytest
```

### CLI 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--symbol` / `-s` | `AAPL` | 股票代码（自动转大写并校验格式） |
| `--start` | `2020-01-01` | 起始日期（含），`YYYY-MM-DD` |
| `--end` | 今天 | 结束日期（含），`YYYY-MM-DD` |
| `--data-root` | `<项目根>/data` | 覆盖数据根目录（便于测试与多环境隔离） |

**退出码**：`0` = 成功；`1` = 失败（失败原因已写入元数据 JSON）。管道内部不抛未捕获异常。

---

## 2. 目录布局

```text
data/                                   # 已被 .gitignore 排除，不入库
  raw/market_bars/<SYMBOL>/
      <SYMBOL>_<start>_<end>_<run_id>.parquet     # 供应商原始数据，逐字节未改动
      <SYMBOL>_<start>_<end>_<run_id>.meta.json   # 抓取元数据 sidecar（来源/时间/行数/列名）
  standardized/market_bars/
      symbol=AAPL/year=2024/part-0.parquet        # 标准数据，Hive 分区（symbol + year）
  .staging/<run_id>/symbol=.../                   # 原子发布暂存区（每次运行后清空）
  .trash/<run_id>/symbol=.../                     # 发布时旧版本备份（成功后删除）
  metadata/run_<run_id>.json                      # 每次运行的完整可复现记录
```

初始化后即可直接用 DuckDB 查询，无需额外 ETL：

```sql
-- duckdb CLI 或 Python API
SELECT symbol, year, count(*) AS n, min(date) AS lo, max(date) AS hi
FROM read_parquet('data/standardized/market_bars/**/*.parquet', hive_partitioning=true)
GROUP BY 1, 2 ORDER BY 1, 2;
```

> 分区键 `symbol` / `year` **不写入** parquet 文件内部；读取时必须开启 Hive 分区（`hive_partitioning=true` 或 `pyarrow.dataset` 的 `partitioning="hive"`）。这与设计文档 §3.4「按数据类型及日期分区」一致，DuckDB 仅作查询层，`.duckdb` 文件可由 Parquet 随时重建（本实现只使用内存连接）。

---

## 3. 管道流程与核心保证

```text
Yahoo Finance 下载
  1. 原始数据落盘            data/raw/...           （未改动 + 元数据 sidecar）
  2. 标准化                  data_sys.standardize
  3. pandera 质量校验（内存）  data_sys.quality
  4. 写入暂存区（Hive 分区）  data/.staging/<run_id>/
  5. 回读 + 再校验 + DuckDB 核验
  6. 原子发布                data/standardized/...  （旧版本进 .trash，成功后删除）
  7. DuckDB 核验已发布数据
  8. 写运行元数据            data/metadata/run_<run_id>.json
```

### 三层防线

| 层次 | 校验对象 | 失败后果 |
| --- | --- | --- |
| 标准化层 | 必要列、代码格式、日期可解析性 | 抛 `StandardizationError`，不落盘 |
| pandera 质量层 | 值域、跨字段一致性、唯一性、时序单调性 | 抛 `QualityCheckError`，不进入发布 |
| DuckDB 核验层 | 磁盘上真实文件的物理类型、行数、日期范围、重复、空值 | 抛 `DuckDBVerificationError`，不发布 |

### 原子发布保证

发布是「**先备份、再替换、成功才删备份**」的交换过程，并记录每一次交换以便失败回滚：

1. 当前 `symbol=X` 分区移动到 `.trash/<run_id>/`；
2. 暂存分区移动到正式位置；
3. 全部成功后删除 `.trash/<run_id>/`。

若第 2 步失败，会从 `.trash` 恢复旧分区。因此**任何失败都不会让正式数据集缺失、半写或损坏**——这是「失败不影响已有数据」的硬约束，已有专门测试覆盖（`tests/test_storage.py::test_atomic_publish_restores_previous_version_on_failure`）。

`data/.staging/<run_id>` 无论成功失败都会在 `finally` 中清理；失败原因与状态始终写入 `metadata/run_<run_id>.json`。

---

## 4. 数据契约

逻辑 schema 见 `data_sys/schema.py::STANDARD_COLUMNS`（共 10 列，顺序固定）。其中 `symbol` 与 `year` 是 **Hive 分区键**，只体现在目录名上，**不写入 parquet 文件内部**；因此单个 parquet 文件实际只含 9 个物理列。

| 列 | 类型 | 存储位置 | 含义 |
| --- | --- | --- | --- |
| `date` | `DATE` | 文件内 | 交易日（朴素日期，无时区） |
| `symbol` | `VARCHAR` | `symbol=` 分区目录 | 大写代码 |
| `open` / `high` / `low` / `close` | `DOUBLE` | 文件内 | **未复权**真实价格（成交与账本使用，见设计文档 §3.2） |
| `adjusted_close` | `DOUBLE` | 文件内 | **复权**收盘价（研究用途，消除拆股/分红跳变） |
| `volume` | `BIGINT` | 文件内 | 成交量 |
| `daily_return` | `DOUBLE` | 文件内 | 复权收盘价的日收益率，首行为 `NULL`（无前值） |
| `dollar_volume` | `DOUBLE` | 文件内 | `close × volume`，用于流动性筛选（设计文档 §3.3 的 100 万美元门槛） |
| `year` | `BIGINT` | `year=` 分区目录 | 由 `date` 派生 |

原始层刻意保留 `--auto_adjust=False`：这样能同时拿到**未复权 OHLC** 与**独立的 `Adj Close`**，避免复权信息被烘焙进 OHLC 而无法还原。

### 质量规则

`high ≥ max(open, close, low)`、`low ≤ min(open, close, high)`；所有价格 `> 0`；`volume ≥ 0`；`(symbol, date)` 唯一；`date` 严格递增；首行 `daily_return` 必须为 `NULL`、其后不得缺失；`symbol` 必须匹配 `^[A-Z0-9.\-]+$`。

失败时会**收集全部违规项**（`lazy=True`）并逐条给出列名、行号与违规值，便于定位。

---

## 5. 运行元数据

每次运行生成 `data/metadata/run_<run_id>.json`：

```json
{
  "run_id": "20260927T223530Z_64b59f",
  "data_source": "yahoo_finance",
  "symbol": "AAPL",
  "requested_start": "2024-01-02",
  "requested_end": "2024-03-29",
  "output_path": "D:\\Python\\Low_Frequency Trading\\data\\standardized\\market_bars",
  "run_at_utc": "2026-09-27T22:35:30+00:00",
  "status": "success",
  "row_count": 61,
  "min_date": "2024-01-02",
  "max_date": "2024-03-28",
  "raw_path": "D:\\Python\\Low_Frequency Trading\\data\\raw\\market_bars\\AAPL\\AAPL_2024-01-02_2024-03-29_20260927T223530Z_64b59f.parquet",
  "years": [[2024, 61]],
  "failure_reasons": [],
  "package_versions": {
    "pandas": "3.0.6",
    "numpy": "2.5.3",
    "pyarrow": "25.0.1",
    "duckdb": "1.5.5",
    "pandera": "0.33.1",
    "yfinance": "1.7.0",
    "pydantic": "2.13.5",
    "tenacity": "9.1.4",
    "typer": "0.27.2",
    "exchange-calendars": "4.13.2"
  }
}
```

失败运行的 `status` 为 `failed`，`failure_reasons` 记录具体原因。元数据 + 原始 parquet + 依赖版本三者结合，可完整复现任意一次运行。

原始层 sidecar（`*.meta.json`）则记录**下载时**的事实：

```json
{
  "provider": "yahoo_finance",
  "symbol": "AAPL",
  "requested_start": "2024-01-02",
  "requested_end": "2024-03-29",
  "fetched_at_utc": "2026-09-27T22:35:31+00:00",
  "raw_rows": 61,
  "raw_columns": ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
}
```

> 两层元数据缺一不可：sidecar 证明「供应商当时给了什么」，`run_*.json` 说明「我们把什么发布到了哪里」。当 Yahoo 事后回溯调整历史价格时，二者的差异正是唯一可追溯的证据。

---

## 6. 模块地图

| 文件 | 职责 |
| --- | --- |
| `data_sys/errors.py` | 异常层次，全部派生自 `DataPipelineError`，编排层只需捕获基类 |
| `data_sys/config.py` | `PipelineConfig`（冻结 dataclass）+ 由它派生的全部路径，路径不硬编码 |
| `data_sys/utils.py` | `run_id` 生成、UTC 时间、日期归一化、JSON 序列化辅助 |
| `data_sys/providers/base.py` | `DataProvider` 抽象基类（稳定接口，后续接 SEC / 付费数据商只加实现） |
| `data_sys/providers/yahoo.py` | Yahoo Finance 实现，含 `tenacity` 重试 |
| `data_sys/standardize.py` | 原始 DataFrame → 标准 schema，列名映射、类型转换、派生列 |
| `data_sys/schema.py` | `STANDARD_COLUMNS` 与 pandera `DataFrameSchema`（唯一真源） |
| `data_sys/quality.py` | 质量检查与 `Failure` 收集，给出可定位的失败描述 |
| `data_sys/storage.py` | 原始层写入 + sidecar、Hive 分区写入/读取、`atomic_publish` 与回滚 |
| `data_sys/duckdb_verify.py` | 独立的磁盘级核验（内存连接，不产生 `.duckdb` 文件） |
| `data_sys/metadata.py` | 日志配置、依赖版本采集、`RunMetadata` 与落盘 |
| `data_sys/data_query.py` | 端到端编排 + Typer CLI（`python -m data_sys.data_query`） |

分层依赖是单向的：`data_query` → `storage`/`duckdb_verify`/`metadata` → `standardize`/`quality` → `schema`/`config`/`utils`。`storage` 与 `providers` 互不感知。

## 7. 测试

```powershell
.\.venv\Scripts\python.exe -m pytest            # 全部 49 项
.\.venv\Scripts\python.exe -m pytest -v         # 逐项
.\.venv\Scripts\python.exe -m ruff check .      # 静态检查
```

测试**完全不访问网络**：`yfinance.download` 始终被 monkeypatch，编排层测试使用 `FakeProvider`。覆盖重点：

- `test_storage.py` — Hive 布局、往返读写、空帧拒绝、sidecar、发布成功、**发布失败回滚到旧版本**、无暂存分区时报错；
- `test_duckdb_verify.py` — 逐项核验失败路径（类型/行数/日期范围/重复/空值/年份覆盖）；
- `test_pipeline.py` — 编排成功路径与失败路径的退出码、元数据落盘、暂存区清理；
- `test_standardize.py` / `test_quality.py` / `test_yahoo.py` — 字段映射、质量规则、provider 重试与错误包装。

## 8. 已知限制与后续

**本 MVP 明确不包含**（属于设计文档后续阶段）：

- 多标的批量下载与增量更新（当前每次运行会重新下载并完整替换该标的的规范化分区）；
- 交易日历、证券主数据、公司行为、SEC EDGAR 基本面；
- `UniverseSnapshot` 的完整筛选规则（`dollar_volume` 等字段已就绪，筛选逻辑待在股票池模块实现）；
- `scripts/download_data.py` 的 Google Drive 整包同步（设计文档 §7）。

**需要注意的行为**：

- 每次运行为单标的**全量替换**该 `symbol=` 分区，而非增量追加。这是 MVP 的有意取舍：换来「分区内容 == 本次请求的时间区间」这一强不变式，避免增量更新引入的重复/空洞状态。
- Yahoo Finance 的数据许可限制见设计文档 §7.4：**在把数据再分发给协作者之前必须复核条款**，必要时改用有明确再分发许可的数据源。
- 标准化层不做交易日补齐（缺失交易日不会生成空行），因此 `date` 的严格递增校验是「单调」而非「连续」。
