# 美股低频量化系统 — 数据系统 MVP

对应《[系统设计与开发路线](美股低频量化系统设计与开发路线.md)》第 2 阶段（数据系统）的第一个可运行闭环。

**当前实现范围**：**版本控制的初始股票池**（`config/universe_seed.csv`，20 只美股大型普通股 + 读取校验接口），以及日频行情的**增量更新闭环**——单标的入口 `python -m data_sys.data_query`（默认 `AAPL`）与整池批量入口 `python -m data_sys.batch_query`，两者共用同一条流程：**Yahoo Finance 下载 → 标准化 → 与已发布分区合并 → pandera 质量校验 → Hive 分区 Parquet 暂存 → 暂存区 DuckDB 核验 → 原子发布（PyArrow 复核）→ 运行元数据**。

---

## 1. 快速开始

```powershell
# 1. 创建虚拟环境并安装运行时依赖
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .

# 2. 安装开发依赖（pytest / ruff）
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# 3. 单标的增量更新（只下载缺失/需刷新的窗口，默认 AAPL）
.\.venv\Scripts\python.exe -m data_sys.data_query --symbol AAPL --start 2024-01-02 --end 2024-03-29

# 4. 整池批量增量更新（按股票池文件顺序逐个执行，单个标的失败不影响其他标的）
.\.venv\Scripts\python.exe -m data_sys.batch_query --universe config/universe_seed.csv --start 2024-01-02 --end 2024-03-29

# 5. 运行测试
.\.venv\Scripts\python.exe -m pytest
```

### CLI 参数（单标的：`python -m data_sys.data_query`）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--symbol` / `-s` | `AAPL` | 股票代码（自动转大写并校验格式） |
| `--start` | `2020-01-01` | 起始日期（含），`YYYY-MM-DD` |
| `--end` | 今天 | 结束日期（含），`YYYY-MM-DD` |
| `--refresh-overlap-sessions` | `5` | 每次运行重下最近 N 个 NYSE 交易日，用于吸收供应商对近期价格的回溯修正（`0` 关闭） |
| `--no-refresh` | — | 等价于 `--refresh-overlap-sessions 0` |
| `--calendar` | `XNYS` | 计划下载窗口所用的交易日历 |
| `--data-root` | `<项目根>/data` | 覆盖数据根目录（便于测试与多环境隔离） |

### CLI 参数（整池批量：`python -m data_sys.batch_query`）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--universe` / `-u` | `config/universe_seed.csv` | 经过校验的股票池 CSV |
| `--start` | `2020-01-01` | 起始日期（含），`YYYY-MM-DD` |
| `--end` | 今天 | 结束日期（含），`YYYY-MM-DD` |
| `--refresh-overlap-sessions` | `5` | 同单标的命令 |
| `--no-refresh` | — | 等价于 `--refresh-overlap-sessions 0` |
| `--include-inactive` | — | 连同 `is_active=false` 的条目一起更新 |
| `--symbols` | 股票池全部 active 条目 | 逗号分隔的子集；含股票池之外的代码会在下载前直接中止 |
| `--data-root` | `<项目根>/data` | 覆盖数据根目录 |
| `--calendar` | `XNYS` | 同单标的命令 |

**退出码**：单标的 `0` = `success` 或 `skipped`，`1` = 失败；批量 `0` = 全部标的都是 `success`/`skipped`，只要有一个 `failed` 就是 `1`（已经发布成功的标的不被回滚）。两个 CLI 内部都不抛未捕获异常，失败原因写入元数据 JSON。

---

## 2. 目录布局

```text
data/                                   # 已被 .gitignore 排除，不入库
  raw/market_bars/<SYMBOL>/
      <SYMBOL>_<window_start>_<window_end>_<run_id>.parquet    # 每个计划窗口一份，供应商原始数据逐字节未改动
      <SYMBOL>_<window_start>_<window_end>_<run_id>.meta.json  # 抓取元数据 sidecar（来源/时间/行数/列名/窗口原因）
  standardized/market_bars/
      symbol=AAPL/year=2024/part-0.parquet        # 标准数据，Hive 分区（symbol + year）
  .staging/<run_id>/symbol=.../                   # 原子发布暂存区（每次运行后清空）
  .trash/<run_id>/symbol=.../                     # 发布时旧版本备份（成功后删除）
  metadata/run_<run_id>.json                      # 每个标的每次运行的完整可复现记录
  metadata/batch_<batch_run_id>.json              # 一次批量运行的汇总索引（逐标的条目 + dataset_summary）
```

`run_id` 是「UTC 时间戳 + 随机后缀」；批量运行时每个标的的 `run_id` 派生为 `<batch_run_id>-<SYMBOL>`，因此批量里每个标的仍有自己独立的 `run_*.json`。

初始化后即可在**独立的研究进程**中用 DuckDB 查询正式数据，无需额外 ETL：

```sql
-- duckdb CLI 或 Python API（独立进程，不与更新进程混用）
SELECT symbol, year, count(*) AS n, min(date) AS lo, max(date) AS hi
FROM read_parquet('data/standardized/market_bars/**/*.parquet', hive_partitioning=true)
GROUP BY 1, 2 ORDER BY 1, 2;
```

> 分区键 `symbol` / `year` **不写入** parquet 文件内部；读取时必须开启 Hive 分区（`hive_partitioning=true` 或 `pyarrow.dataset` 的 `partitioning="hive"`）。这与设计文档 §3.4「按数据类型及日期分区」一致。

### 2.1 读取边界：DuckDB 只碰 staging，正式数据只走 PyArrow

DuckDB 在进程内按路径缓存文件句柄，而且**重建 view 并不会清空这份缓存**；在 Windows 上把「刚被替换过的同一路径」再交给 DuckDB 读取，会以原生 `access violation`（共享连接时表现为卡死）直接杀死解释器，而不是抛 Python 异常。

因此本实现的硬性边界是：

| 场景 | 读取方式 |
| --- | --- |
| 更新进程校验**唯一 staging 目录**（`.staging/<run_id>/`，只写一次、从不复用同一路径） | DuckDB（`data_sys/duckdb_verify.py`） |
| 更新进程读取**正式数据**（发布后复核、增量合并前回读旧分区、批量汇总） | PyArrow（`data_sys/storage.py`、`data_sys/summary.py`） |
| 独立研究进程只读查询正式 Parquet | 可以用 DuckDB；**不要与同进程的数据替换混用** |

`.duckdb` 文件不是必需品，可由 Parquet 随时重建（本实现只使用内存连接）。

---

## 3. 股票池与证券主数据

初始股票池是**版本控制的配置文件**，运行时**不抓取**任何指数成分或筛选器结果，因此任何协作者读到的都是同一份清单：

```text
config/universe_seed.csv                # 20 只美股大型高流动性普通股，入库 Git
```

### 3.1 字段契约

| 字段 | 规则 |
| --- | --- |
| `symbol` | 股票代码。必须非空、大写、**唯一**，且匹配 `data_sys.schema.SYMBOL_PATTERN`（`^[A-Z0-9.\-]+$`，因此 `BRK.B` 这类含点号的多类别代码合法），长度 ≤ 10。Yahoo 侧含点号的代码写作连字符（`BRK.B` 在供应商那里是 `BRK-B`），provider 会自动回退到该拼写，磁盘上仍是 `BRK.B`（见 §10） |
| `name` | 证券名称，非空 |
| `exchange` | 上市交易所，须属于 `ALLOWED_EXCHANGES`：`NASDAQ` / `NYSE` / `NYSE AMERICAN` / `NYSE ARCA` / `BATS` / `OTC` |
| `security_type` | 第一版**仅允许** `common_stock` |
| `is_active` | 布尔值，接受 `true/false`、`yes/no`、`y/n`、`1/0` |

额外列会被忽略（契约是**最小**列集合），便于后续加 `sector`、`cik` 等字段而不破坏现有校验。

### 3.2 读取接口

唯一入口是 `data_sys/universe.py`；下游模块只拿标准化后的结果，不接触 CSV 本身：

```python
from data_sys.universe import load_universe_symbols, read_universe

symbols = load_universe_symbols()  # 已标准化、去重、排序的 list[str]
symbols = load_universe_symbols(active_only=False)  # 含 is_active=false 的证券

universe = read_universe()  # 完整证券主数据
universe.by_symbol()["AAPL"].exchange  # -> 'NASDAQ'
universe.symbols  # 全部代码（排序、去重）
universe.active_symbols  # 仅 is_active=true
universe.to_frame()  # DataFrame，列顺序 == 字段契约顺序
```

`validate_universe(frame)` 是同一套规则的**不抛异常**版本，返回问题列表（空列表代表合法），可直接用于校验内存中构造的股票池。

### 3.3 归一化与校验

归一化（无损）：`symbol` / `exchange` 转大写，`security_type` 转小写，所有文本字段去除首尾空白。因此 `aapl`、` nasdaq `、`Common_Stock` 都会被接受并标准化；而 `aapl` 与 `AAPL` 同时出现会被判定为**重复代码**。

校验在归一化**之后**进行，失败时抛出 `UniverseError` 并**一次性列出全部问题**（行号按数据行计，不含表头），避免「改一处、报一次」：

| # | 校验项 |
| --- | --- |
| 1 | 文件存在、可读，且不是目录 |
| 2 | 五个必需字段齐全，且表头无重复列名 |
| 3 | 至少有 `MIN_UNIVERSE_SIZE`（=10）条记录 |
| 4 | `symbol` 非空、长度合法、匹配 `SYMBOL_PATTERN`，且归一化后唯一 |
| 5 | `name` 非空 |
| 6 | `exchange` 非空且属于允许集合 |
| 7 | `security_type` 非空且属于允许集合 |
| 8 | `is_active` 可解析为布尔值 |

真实输出示例（第 3 行代码格式错误、第 4 行交易所拼写错误）：

```text
invalid universe configuration in 'config/universe_seed.csv'
  - data row 3: symbol 'AA PL' does not match ^[A-Z0-9.\-]+$ (upper-case letters, digits, '.' and '-' only)
  - data row 4: unknown exchange 'NASDQ'; allowed: ['BATS', 'NASDAQ', 'NYSE', 'NYSE AMERICAN', 'NYSE ARCA', 'OTC']
```

文件以全文本方式读取（`dtype=str`, `keep_default_na=False`），避免 `NA`、`TRUE` 这类代码被 pandas 误判为缺失值。`allowed_exchanges` / `allowed_security_types` / `min_size` 均可按调用覆盖，因此新增 OTC 上市或放宽规模下限都不需要改模块常量。

> **本阶段明确不做**：CIK、退市历史、ticker 历史映射（留给 SEC / reference-data 阶段）；`UniverseSnapshot` 的逐决策日流动性筛选（`dollar_volume` 等字段已就绪，筛选规则待实现）。

---

## 4. 管道流程与核心保证

```text
单个标的的增量更新（data_sys.update.update_symbol —— 单标的与批量入口共用同一条流程）
  0. 计划窗口                data_sys.planner       （按 NYSE 交易日算出缺哪些窗口、以及为什么）
  1. 原始数据落盘            data/raw/...           （每个窗口一份，未改动 + 元数据 sidecar）
  2. 标准化                  data_sys.standardize
  3. 与已发布分区合并         data_sys.merge        （PyArrow 回读旧分区；同日新行胜出；派生列全量重算）
  4. pandera 质量校验（内存）  data_sys.quality
  5. 写入暂存区（Hive 分区）  data/.staging/<run_id>/
  6. 回读 + 再校验 + DuckDB 核验暂存区（DuckDB 只读 .staging/<run_id>/，永不碰正式目录）
  7. 原子发布事务            data/standardized/...  （见下「原子发布保证」）
  8. 写运行元数据            data/metadata/run_<run_id>.json
```

批量入口（`data_sys.batch_query`）不重复实现上述任何一步：它按股票池的**文件顺序**逐个调用同一个函数（`run_id` 派生为 `<batch_run_id>-<SYMBOL>`），最后写一份 `data/metadata/batch_<batch_run_id>.json` 作为整批的索引。

### 三层防线

| 层次 | 校验对象 | 失败后果 |
| --- | --- | --- |
| 标准化层 | 必要列、代码格式、日期可解析性 | 抛 `StandardizationError`，不落盘 |
| pandera 质量层 | 值域、跨字段一致性、唯一性、时序单调性 | 抛 `QualityCheckError`，不进入发布 |
| DuckDB 核验层（仅暂存区） | 磁盘上真实文件的物理类型、行数、日期范围、重复、空值 | 抛 `DuckDBVerificationError`，不发布 |

### 原子发布保证

`data_sys.storage.atomic_publish` 是一次**两阶段事务**：先把每一步交换记入日志（journal），**确认成功之后才删除备份**，任何失败都整体回滚。

1. **apply**：当前 `symbol=X` 分区移动到 `.trash/<run_id>/`（备份），暂存分区移动到正式位置；每一步交换都先记账再执行，所以第一步就失败也能还原。
2. **confirm**：事务调用调用方传入的 `confirm` 回调——即 `data_sys.update._confirm_publish`，用 **PyArrow** 回读新上线的分区（永不经过 DuckDB，见 §2.1「读取边界」）并报告问题。
3. **commit**：仅当第 1 步的每一步都成功**且**第 2 步没有报告任何问题时，才删除本次的 `.trash/<run_id>/`——备份的存活期覆盖整个确认过程。
4. **roll back**：第 1 步或第 2 步任一失败，则把所有备份按相反顺序放回原位，并删除本次事务自己创建的空备份目录（用 `rmdir` 而非 `rmtree`，因此**绝不会连带删掉尚未还原的备份**），随后抛 `PublishError`；未能还原的分区会出现在异常信息里，其备份也会被保留。若该 symbol **原本没有正式分区**（首次发布，此时没有备份可还原），回滚会删掉本次刚换入的分区，使正式目录回到「不存在该 symbol 分区」的状态——不会留下一个"已失败但数据在里面"的分区。

因此**任何失败都不会让正式数据集缺失、半写或损坏**：正式数据集要么是完整的新版本，要么是完整的旧版本；而 `result.published` / `meta.published` **只在第 3 步提交成功后**才置为 `True`，失败一律为 `False` 并把原因写入运行元数据（不吞异常、不伪造成功）。对应测试：

- `tests/test_storage.py::test_atomic_publish_restores_previous_version_on_failure`
- `tests/test_storage.py::test_atomic_publish_confirms_before_it_drops_the_backup`
- `tests/test_storage.py::test_atomic_publish_rolls_back_when_the_confirmation_reports_an_issue`
- `tests/test_storage.py::test_atomic_publish_rolls_back_when_the_confirmation_raises`
- `tests/test_storage.py::test_atomic_publish_rolls_back_a_first_publish_without_a_backup`
- `tests/test_update.py::test_a_publish_failure_keeps_the_published_dataset`
- `tests/test_update.py::test_a_failed_publish_confirmation_keeps_the_published_dataset`
- `tests/test_update.py::test_a_failed_confirmation_of_a_first_publish_leaves_no_partition`

`data/.staging/<run_id>` 无论成功失败都会在 `finally` 中清理；失败原因与状态始终写入 `metadata/run_<run_id>.json`。测试通过 `metadata/run_<run_id>.json` 里的 `run_id` 精确定位运行记录——`run_*.json` 的文件名以秒级时间戳 + **随机** run-id 后缀结尾，不能用来推断先后顺序。

### 批量运行的失败隔离

`python -m data_sys.batch_query` 逐标的调用同一个 `update_symbol`，并**严格顺序执行**（本阶段刻意不引入并发）：

- 每个标的都是一次**独立的**原子发布事务：一个标的失败**不会**中断整批，也**不会**回滚其它已经发布成功的标的；
- 批量退出码：全部 `success` / `skipped` 为 `0`，只要有一个 `failed` 就是 `1`，已发布的数据在两种情况下都保持在线；
- 结果落在 `metadata/batch_<batch_run_id>.json`：`requested_symbols`（股票池文件顺序）、`per_symbol_status`（逐标的状态、下载行数、发布日期范围）、`dataset_summary`（正式数据集的 `(symbol, year)` 行数与日期范围）与 `notes`；每个标的自己的细节仍在 `metadata/run_<batch_run_id>-<SYMBOL>.json`；
- `dataset_summary` **只走 PyArrow**（§2.1「读取边界」），并且是尽力而为的：即使汇总读不出来，批量记录也一定会写出，原因记入 `notes`；
- `--symbols` 里出现股票池之外的代码时，批量在**任何下载之前**就以 `UniverseError` 中止（退出码 `1`）。

---

## 5. 数据契约

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

## 6. 运行元数据

每次运行生成 `data/metadata/run_<run_id>.json`（下例只列固定字段，增量更新新增的字段见其后说明）：

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

增量更新还会在该文件里写入这些字段：`existing_row_count`、`downloaded_row_count`、`published_row_count`、`new_dates_count`、`replaced_dates_count`、`missing_ranges`、`refresh_range`、`raw_paths`（本次每个下载窗口一份原始文件）、`plan`（`data_sys.planner` 的完整计划，含每个窗口的下载原因）、`notes`、`batch_run_id`（批量成员才有）、`published`（**只有**原子发布提交成功才为 `true`）。批量运行的汇总（逐标的条目 + `dataset_summary`）另见 `data/metadata/batch_<batch_run_id>.json`。

原始层 sidecar（`*.meta.json`）则记录**下载时**的事实：

```json
{
  "provider": "yahoo_finance",
  "symbol": "AAPL",
  "vendor_symbol": "AAPL",
  "requested_start": "2024-01-02",
  "requested_end": "2024-03-29",
  "fetched_at_utc": "2026-09-27T22:35:31+00:00",
  "raw_rows": 61,
  "raw_columns": ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
}
```

> 两层元数据缺一不可：sidecar 证明「供应商当时给了什么」，`run_*.json` 说明「我们把什么发布到了哪里」。当 Yahoo 事后回溯调整历史价格时，二者的差异正是唯一可追溯的证据。sidecar 里的 `vendor_symbol` 是**供应商那边真正命中**的拼写（项目规范拼写与供应商拼写不同时，如 `BRK.B` ↔ `BRK-B`），`symbol` 则始终是项目规范拼写。

---

## 7. 模块地图

| 文件 | 职责 |
| --- | --- |
| `data_sys/errors.py` | 异常层次，全部派生自 `DataPipelineError`，编排层只需捕获基类 |
| `data_sys/config.py` | `PipelineConfig`（冻结 dataclass）+ 由它派生的全部路径，路径不硬编码 |
| `data_sys/utils.py` | `run_id` 生成、UTC 时间、日期归一化、JSON 序列化辅助 |
| `data_sys/universe.py` | 证券主数据：读取并校验 `config/universe_seed.csv`，返回标准化去重的代码列表；失败时抛出 `UniverseError` 并汇总全部问题 |
| `data_sys/market_calendar.py` | NYSE（`XNYS`）交易日历封装：会话序列、区间裁剪、最近 N 个会话——「一个交易日」的唯一真源 |
| `data_sys/planner.py` | 增量更新的下载计划：由「已发布会话 + 请求区间 + 重叠刷新宽度」决定**要下载哪些窗口、以及原因**；纯计算，无 I/O |
| `data_sys/merge.py` | 把新下载的行并入已发布历史：`(symbol, date)` 去重时新行胜出，之后整段重算 `daily_return` / `dollar_volume` |
| `data_sys/providers/base.py` | `DataProvider` 抽象基类（稳定接口，后续接 SEC / 付费数据商只加实现） |
| `data_sys/providers/yahoo.py` | Yahoo Finance 实现，含 `tenacity` 重试；**仅限请求**的代码拼写回退（`BRK.B` → `BRK-B`，先试规范拼写，落盘/元数据仍是 `BRK.B`） |
| `data_sys/standardize.py` | 原始 DataFrame → 标准 schema，列名映射、类型转换、派生列（`daily_return` / `dollar_volume` 的唯一实现） |
| `data_sys/schema.py` | `STANDARD_COLUMNS` 与 pandera `DataFrameSchema`（唯一真源） |
| `data_sys/quality.py` | 质量检查与 `Failure` 收集，给出可定位的失败描述 |
| `data_sys/storage.py` | 原始层写入 + sidecar、Hive 分区写入/读取、`atomic_publish` 两阶段事务（确认后提交 / 失败回滚） |
| `data_sys/summary.py` | 正式数据集的 `(symbol, year, n_rows, min_date, max_date)` 汇总，**只走 PyArrow**；`summary_records` 输出 JSON 安全记录 |
| `data_sys/duckdb_verify.py` | 独立的磁盘级核验，**只允许指向唯一 staging 路径**（内存连接，不产生 `.duckdb` 文件） |
| `data_sys/metadata.py` | 日志配置、依赖版本采集、`RunMetadata` 与落盘 |
| `data_sys/update.py` | 单标的增量更新的**唯一实现**：计划 → 下载 → 合并 → 校验 → 暂存核验 → 原子发布 → 运行元数据；两个 CLI 都只是它的薄壳 |
| `data_sys/data_query.py` | 单标的 Typer CLI（`python -m data_sys.data_query`） |
| `data_sys/batch_query.py` | 整池批量 Typer CLI（`python -m data_sys.batch_query`）：选择标的、顺序执行、失败隔离、写批量汇总 |
| `scripts/run_factor.py` | 因子研究层的最小真实入口：按名字建因子 → 决策日/日历校验 → 同键已存则读回（缓存命中）→ 否则读本地标准化行情、计算单日截面、质量检查、落盘 → 打印基础统计与 `cache_status`；`--force` 强制重算并原子覆盖（只组合 `factor_sys` 的公开接口） |
| `factor_sys/factor_schedule.py` | 多日期研究的**纯日期规则**：`decision_schedule()` / `select_decision_dates()` 按 weekly / biweekly / monthly + anchor 从交易日历里选出决策日（节假日回退、双周相位、月末）；不读行情、不算因子、不写文件，也不依赖 Store / Runner |
| `scripts/run_factor_batch.py` | 批量编排入口：日期规则 → 逐日调用 `run_factor()`（复用同一份 Panel 缓存）→ 单日失败隔离 → 每个日期一行状态 + 汇总 → 在 `factors/batch_runs/` 写一份 manifest（临时文件 + 原子替换） |
| `scripts/analyze_factor.py` | 多日期研究分析入口：调用上面的批量入口取得**本次日程**的 Panel → 逐期用 `ForwardReturnCalculator` 计算未来收益标签 → 复用 `QuickAnalyzer` / `ICAnalyzer` / `QuantileAnalyzer` 三种既有分析 → 在 `factors/research_runs/` 发布一份独立研究报告（metadata + 5 张 CSV，临时目录 + 一次原子重命名）；不重算公式、不扫 Panel 目录、不做策略或回测 |

分层依赖是单向的：`data_query` / `batch_query` → `update` → `planner` / `merge` / `storage` / `duckdb_verify` / `metadata` → `standardize` / `quality` / `market_calendar` → `schema` / `config` / `utils`；`summary` 只依赖 `storage`（因此这一层完全没有 DuckDB）。`storage` 与 `providers` 互不感知。`universe` 是独立于行情管道的一层（只依赖 `schema`/`config`/`errors`），由 `batch_query` 消费，行情管道本身不依赖它，因此两者可各自演进。

## 8. 运行一个因子截面

因子研究层的**最小真实入口**：给出「因子名 + 决策日」，读本地已发布的标准化行情、计算一个截面、做质量检查、落盘，并打印基础统计。同一个 `(因子名, 版本, 决策日)` 已经算过时，默认直接读回已存的 Panel（轻量缓存），`--force` 可强制重算。

```powershell
# 在项目根目录、已激活虚拟环境后：
python scripts/run_factor.py --factor momentum_12_1 --as-of 2024-12-31

# 忽略已存 Panel，重新读取行情、重新计算并原子覆盖：
python scripts/run_factor.py --factor momentum_12_1 --as-of 2024-12-31 --force
```

可选参数：

| 参数 | 含义 |
| --- | --- |
| `--factor NAME` | 必填。通过 `default_factor_registry()` 按名字查找因子（脚本不 import 任何公式模块） |
| `--as-of YYYY-MM-DD` | 必填。决策日，必须是交易日历（默认 `XNYS`）上的真实会话 |
| `--data-root PATH` | 可选。**同时**决定行情来源 `<PATH>/standardized/market_bars` 与因子落盘 `<PATH>/factors/panels`；默认 `PipelineConfig().data_root` |
| `--universe-path PATH` | 可选。只替换静态股票池 CSV；默认 `config/universe_seed.csv` |
| `--calendar NAME` | 可选。交易日历名；默认沿用数据系统与 Runner 的 `XNYS` |
| `--force` | 可选，无值开关。忽略缓存，重新读行情、重新计算，并原子覆盖同一个 Panel 文件 |

固定调用顺序（前四步命中与未命中共用，每一步都只走该层已有的公开接口，脚本本身不实现存储、质量或公式规则）：

```text
default_factor_registry() → registry.create(name) → 决策日 + 交易日历校验
→ ParquetFactorStore.exists(name, version, as_of)
    命中且未传 --force：ParquetFactorStore.read(panel)
                       → 默认质量检查器 inspect → QuickAnalyzer.summarize → 打印
    未命中或 --force：StaticUniverseProvider → ParquetMarketDataReader → FactorRunner.run()
                     → ParquetFactorStore.write(panel) → QuickAnalyzer.summarize → 打印
```

需要知道的几点：

1. **只读本地数据**：命令读取已发布的标准化 Parquet，**不下载、不联网、不更新 `data/`**；缺少行情目录或目录为空会直接报错，而不是尝试抓取。
2. **该日期需要有足够历史**：`momentum_12_1` 需要决策日之前至少 252 个交易日（连同决策日共 253 根 bar）；历史不足的标的会被标记为不合格，而不是猜测价格。
3. **输出是单日截面**：因子面板（每标的一行 `raw_value` / `is_eligible`）+ 质量发现 + 基础描述统计（`universe_size`、`eligible_count`、`eligible_coverage`、`valid_count`、`missing_count`、`missing_rate`、`mean`、`std`、`minimum`、`maximum`）+ `stored_panel_path`、`quality_status` 与 `cache_status: hit|miss|forced`。
4. **Panel 会写入**：

   ```text
   data/factors/panels/
     factor_name=<name>/
       factor_version=<version>/
         year=<year>/
           as_of_date=<date>.parquet
   ```

5. **同键默认读缓存**：`(factor_name, factor_version, as_of_date)` 相同且文件已存在时，直接读回该 Panel（详见下面的「缓存语义」）；`--force` 才会重新计算并原子覆盖同一个文件，任何情况下都不会产生第二个文件。
6. **质量报告是软发现**：有 issue 时输出 `quality_status: reported` 并逐条打印 `quality_issue_code` / `_severity` / `_message` / `_affected_symbols`，Panel 仍然写出；退出码 `0` 表示运行成功，`1` 表示领域错误（未知因子、非交易日、未知日历、股票池不可读、行情缺失、写入失败、缓存文件损坏等，stderr 一行 `error: ...`），`2` 由 argparse 给出（缺参数或未知参数）。
7. **这不是回测**：不产生真实或模拟订单，也没有交易成本、持仓或收益曲线。
8. **仍有幸存者偏差**：`StaticUniverseProvider` 使用静态股票池（今天的成分股），历史研究无法消除退市与成分变更带来的偏差。

### 缓存语义（`cache_status`）

| `cache_status` | 含义 |
| --- | --- |
| `miss` | 该键此前不存在：读取行情 → 计算 → 写入新 Panel |
| `hit` | 该键已存在：直接读回 Panel，**不读行情、不重新计算、不覆盖文件** |
| `forced` | 传了 `--force`：无论是否存在都重新计算并原子覆盖（不存在时就是一次首次计算） |

- 缓存默认开启，键只有 `(factor_name, factor_version, as_of_date)`，不需要额外参数。
- 命中时仍然重新做质量检查与统计：Factor Store 只保存 Panel，不保存旧的 `FactorQualityReport`，所以读回后按**当前**质量规则重新 `inspect`，再 `QuickAnalyzer.summarize`，输出的仍是同一套摘要格式。
- 命中时文件不被触碰：Panel 字节与修改时间都不变（`stored_panel_path` 就是那个已读文件）。
- 已存在但损坏、元数据不一致或无法恢复的缓存文件会让命令失败（退出码 `1`，stderr 一行 `error: ...`），**不会**悄悄重算并覆盖：坏文件是证据，需要手工处理或删除。
- 非交易日、非法 `--calendar`、未注册的因子在查询缓存之前就失败：某个同日期文件恰好存在也不会被当成命中。
- 需要 `--force` 的时机：更新了历史行情、替换了静态股票池、修改了因子实现或参数之后。
- 缓存**只**按因子名、版本与日期判断，不追踪行情版本、股票池版本或代码提交版本——它没有数据血缘，也不是严格的数据版本缓存。
- 因子公式或含义有实质变化时，优先**升级 `factor_version`**（得到新键、新目录，可与旧结果并存），而不是只依赖 `--force`。

这个缓存**不是**：严格数据版本缓存、数据血缘追踪、多日期调度、因子历史回测，也不会自动更新行情。

本入口刻意**不包含**：多日期批量、调仓日历、因子 transforms、未来收益标签、IC / ICIR、分位收益与 Top–Bottom spread、策略信号与回测。这些都需要**历史日期序列**，留给后续的独立研究命令或回测系统；缓存只是同键读回，不是多日期调度。批量生成多个决策日的 Panel 见下一节。

## 9. 批量生成因子 Panel

`scripts/run_factor.py` 只回答**一个决策日**（因子名 + 日期 → 一个截面、一次落盘、一次缓存判断）；`scripts/run_factor_batch.py` 只回答**一批决策日**：按频率规则选出日期，逐日调用上面那个入口（不复制任何单日计算、行情读取、缓存判断或落盘逻辑），并把这次真正用了哪些日期写成 manifest。

```powershell
# 每周四（默认频率与默认 anchor），2024 全年：
python scripts/run_factor_batch.py --factor momentum_12_1 --start 2024-01-01 --end 2024-12-31

# 双周（相位锚定在请求区间内的第一个候选周）：
python scripts/run_factor_batch.py --factor momentum_12_1 `
  --start 2024-01-01 --end 2024-12-31 --frequency biweekly

# 每月最后一个交易日（monthly 不接受 --anchor-weekday）：
python scripts/run_factor_batch.py --factor momentum_12_1 `
  --start 2024-01-01 --end 2024-12-31 --frequency monthly

# 强制重算每一个选中日期（原子覆盖各自的 Panel 文件）：
python scripts/run_factor_batch.py --factor momentum_12_1 `
  --start 2024-01-01 --end 2024-12-31 --force
```

| 参数 | 含义 |
| --- | --- |
| `--factor NAME` | 必填。与单日入口一样通过 `default_factor_registry()` 解析；未知因子在循环前就失败 |
| `--start` / `--end` | 必填。**闭区间**；`start > end` 明确报错 |
| `--frequency` | `weekly`（默认）/ `biweekly` / `monthly` |
| `--anchor-weekday` | `MON`..`FRI`，默认 `THU`；只对 weekly / biweekly 有意义，与 `monthly` 同时使用会被 argparse 以退出码 `2` 拒绝 |
| `--data-root` / `--universe-path` / `--calendar` / `--force` | 语义与单日入口完全一致，原样传给 `run_factor()` |

### 日期规则（`factor_sys/factor_schedule.py`）

| 频率 | 选择规则 |
| --- | --- |
| `weekly` | 每个自然周（ISO 周，周一至周日）选一天：anchor 星期几就是那一天；若交易所休市（感恩节周四、圣诞周三……），取该日期**之前最近**的交易日 |
| `biweekly` | 先按 `weekly` 得到候选，再**从请求区间内的第一个候选周开始每隔一个**取一次；相位写入 manifest 的 `biweekly_anchor: "first_selected_week_in_requested_range"`，因此同样的 `start/end/frequency/anchor/calendar` 永远得到同样的日期 |
| `monthly` | 每个自然月的**最后一个交易日**（由交易日历决定，不做 28/30/31 的算术）；`anchor_weekday` 记为 `null` |

- 闭区间，且选出的日期永远在区间内：落在区间外的候选（例如从周二开始请求 MON anchor）会被**丢弃**，不会被平移进来。
- 结果严格升序、无重复、全部是真实交易日；周末、节假日、月末与年末边界都交给 `data_sys.market_calendar`，本模块不维护任何节假日表。
- `DecisionSchedule` 是**自校验**的：手工构造时它会把每个日期再向它自己记录的那个日历确认一次，因此周末、节假日或日历未覆盖的日期都会被拒绝（错误信息带上星期，如 `2024-01-06 (SAT)`），不会出现「记录里写着交易日、实际却是休市日」的对象。
- 区间内没有任何合格日期**不是错误**：会写出 `selected_dates: []` 的空 manifest，退出码 `0`。
- 默认 `THU` 的含义是「周四收盘形成信号」，因此**理论上**订单可以在周五开盘执行；但本工具只计算因子截面，不产生订单、不建仓、不回测，也没有任何持仓期或成本假设。
- 日期规则是公共 API（`decision_schedule()` / `select_decision_dates()`），未来的 IC / 分位批量入口应调用同一个函数，而不是另写一份。

### 缓存行为

Panel 的唯一身份仍然只有 `(factor_name, factor_version, as_of_date)`；频率与 anchor 只决定**本次访问哪些日期**，不进入键、不进入 Parquet 路径、也不写进 `FactorPanel`。因此每周四与每周三的两次运行若落在同一日期，第二次一定是 `hit`，不会产生第二份文件。

| 结果 | 含义 |
| --- | --- |
| `miss` | 该日期此前没有 Panel：读行情 → 计算 → 落盘（`computed_count`） |
| `hit` | 该日期已有 Panel：直接读回，不读行情、不重算、不覆盖（`cached_count`） |
| `forced` | 传了 `--force`：每个选中日期都重算并原子覆盖各自文件（`forced_count`） |
| `failed` | 该日期失败（损坏缓存、行情缺失、单日因子失败……），记录后继续下一个日期（`failed_count`） |

- 一个日期失败**不会**中断其余日期，也不会回滚、删除或覆盖其他日期已成功写入的 Panel；质量报告是软提示，不算失败。
- 若任何日期失败，最终退出码为 `1`；全部成功为 `0`；参数错误由 argparse 给出 `2`；stderr 只有一行 `error: ...`，不打印 traceback。

### 输出

每个选中日期一行状态（日期、`hit`/`miss`/`forced`/`failed`、质量状态或错误类型），随后是失败明细，最后是固定的汇总块（最后一行永远是 `manifest_path`）：

```text
2024-01-04  miss    clean
2024-01-11  hit     clean
2024-01-18  failed  FactorStoreError
failure: 2024-01-18 FactorStoreError: ...
factor_name: momentum_12_1
factor_version: 1.0.0
start_date: 2024-01-01
end_date: 2024-01-31
frequency: weekly
anchor_weekday: THU
biweekly_anchor: n/a
calendar_name: XNYS
selected_dates: 3
computed_count: 1
cached_count: 1
forced_count: 0
failed_count: 1
manifest_path: data/factors/batch_runs/20240102T030405Z_9f8e7d.json
```

`monthly` 运行的 `anchor_weekday` 与 `biweekly_anchor` 都打印为 `n/a`；其余字段与单日入口的摘要同名同义。

### Manifest

每次运行都会在 `data_root` 下新增一个 JSON（路径由 `PipelineConfig(data_root=...)` 推导，不硬编码本机绝对路径）：

```text
data/factors/batch_runs/<batch_run_id>.json
```

```json
{
  "format_version": 1,
  "batch_run_id": "20240102T030405Z_9f8e7d",
  "created_at_utc": "2024-01-02T03:04:05+00:00",
  "factor_name": "momentum_12_1",
  "factor_version": "1.0.0",
  "requested_start_date": "2024-01-01",
  "requested_end_date": "2024-01-31",
  "frequency": "weekly",
  "anchor_weekday": "THU",
  "biweekly_anchor": null,
  "calendar_name": "XNYS",
  "selected_dates": ["2024-01-04", "2024-01-11", "2024-01-18", "2024-01-25"],
  "results": [
    {
      "as_of_date": "2024-01-04",
      "status": "hit",
      "quality_status": "clean",
      "panel_path": "factors/panels/factor_name=momentum_12_1/factor_version=1.0.0/year=2024/as_of_date=2024-01-04.parquet"
    }
  ],
  "failures": []
}
```

- `selected_dates` 记录**完整的**日期清单（不是只记 `start`/`end`）；每个选中日期必须**恰好**对应一条 `results` 或一条 `failures`：`FactorBatchOutcome` 在构造时就拒绝「选中了但没有记录」的缺口，因此摘要不会缺行，manifest 也不会漏日期。
- `panel_path` 相对 `data_root`、用 `/` 分隔，manifest 可以随数据目录一起搬迁。
- manifest 是**一次运行的记录**，不是缓存键：同样的参数运行两次会得到两份独立、可追溯的 manifest。
- 写入使用临时文件 + 原子替换；写失败不会损坏已存在的 manifest，并让命令以 `1` 退出（结果对象里 `manifest_error` 说明原因，已完成的 Panel 不受影响）。

后续的 IC / 分位分析（§10）应当**读取 manifest 或调用同一个 `decision_schedule()`**，不要把 `data/factors/panels` 下所有已存 Panel 混成一条 IC 序列：weekly / biweekly / monthly 的决策日互相重叠，混在一起算出的均值不对应任何策略。

本节**不含**：未来收益标签、IC / ICIR、分位收益与 Top–Bottom spread（这三项见 §10）、策略信号、回测与图表。

## 10. 多日期因子研究分析

`scripts/analyze_factor.py` 把「按日程算 Panel」（§9）接上「未来收益标签 + IC + 分位收益」：给定因子名、区间、频率与 horizon，它会**自动补齐或复用**所需 Panel，为每一期计算未来收益标签，复用既有的三种分析对象，并把结果写成一份独立的研究报告目录。

它仍然是**研究分析**：不产生订单、不建仓、不算净值、不算成本与滑点、不做横截面回归或显著性检验、不合成多因子、不定义新公式、不导出逐股票明细，也不新增任何永久性的未来收益存储层。

```powershell
python scripts/analyze_factor.py `
  --factor momentum_12_1 `
  --start 2024-01-01 `
  --end 2025-12-31 `
  --frequency weekly `
  --anchor-weekday THU `
  --horizon-sessions 20
```

| 参数 | 含义 |
| --- | --- |
| `--factor NAME` | 必填。通过 `default_factor_registry()` 解析；未知因子在读取任何行情前失败 |
| `--start` / `--end` | 必填。**闭区间**，`start > end` 明确报错；标签可以越过 `--end`（horizon 决定标签终点） |
| `--frequency` | `weekly`（默认）/ `biweekly` / `monthly`，规则完全沿用 §9 的 `decision_schedule()` |
| `--anchor-weekday` | `MON`..`FRI`；weekly / biweekly 默认 `THU`；与 `monthly` 同时使用会被 argparse 以退出码 `2` 拒绝 |
| `--horizon-sessions N` | 必填，正整数。**以交易日 session 计**：`5` ≈ 一周、`10` ≈ 两周、`20` ≈ 一个月；非正整数由 argparse 以 `2` 拒绝 |
| `--quantiles N` | 每期截面切成几组，默认 `5`，至少 `2` |
| `--min-pairs N` | 单期 IC 所需的最少可用配对，默认沿用 `ICAnalyzer` 的默认值（`2`） |
| `--data-root` / `--universe-path` / `--calendar` | 语义与 §8/§9 完全一致，原样传给批量入口 |
| `--force` | 原样传给批量入口：每个选中日期都重算并原子覆盖各自的 Panel |
| `--report-root PATH` | 可选。研究报告的父目录，默认 `<data_root>/factors/research_runs`（不硬编码本机路径） |

### 频率与 horizon 是两个独立概念

| 参数 | 决定什么 |
| --- | --- |
| `--frequency` / `--anchor-weekday` | **在哪些日期形成并分析因子横截面**（每周四收盘一期 / 双周 / 月末） |
| `--horizon-sessions` | **每个日期向后观察多少个交易日收益** |

因此 `weekly + THU + horizon_sessions=20` 的准确含义是：

> 每周四收盘形成一个因子 Panel，检验该 Panel 对未来 20 个 XNYS 交易日复权收益的预测能力。

`weekly` 不会被迫等于 5 日、`monthly` 也不会被迫等于 20 日；工具不会从频率推断步长，必须显式给出 `--horizon-sessions`。

### 自动复用或补齐 Panel

分析**不会**调用 `ParquetFactorStore.read_range()` 把 `data/factors/panels` 里所有已存 Panel 混进同一条 IC 序列（本地可能同时存着周三、周四、双周、月末和手工单日 Panel，混在一起的均值不对应任何采样口径）。它只调用 §9 的：

```python
scripts.run_factor_batch.run_factor_batch(...)
```

并使用返回的 `FactorBatchOutcome.outcomes` 里的 Panel，于是三种情形自动成立：

| 情形 | 行为 |
| --- | --- |
| 已有同一 `(factor, version, date)` Panel | 缓存命中，直接读回（`hit`） |
| 缺失 Panel | 读取本地标准化行情、计算、落盘（`miss`） |
| 传 `--force` | 每个选中日期都重算并原子覆盖（`forced`） |

单日计算、缓存判断、Store 路径、Panel 读取与 batch manifest 全部由既有模块负责，本入口不复制其中任何一段。

### 执行顺序与失败语义

```text
run_factor_batch()           本次日程的每一期 Panel（复用缓存 / 补齐缺失 / --force 重算）
  → 逐期 ForwardReturnCalculator.compute(panel, horizon_sessions)   每期一组标签
  → QuickAnalyzer().summarize_history(panels)                       基础统计
  → ICAnalyzer(min_pairs=...).analyze(panels, labels)               Spearman IC / 平均 IC / ICIR
  → QuantileAnalyzer(quantiles=...).analyze(panels, labels)         分位收益与 Top − Bottom spread
  → 发布一份独立报告目录
```

- 标签通过 `ForwardReturnCalculator(ParquetMarketDataReader(<data_root>/standardized/market_bars), calendar_name=...)` 计算，用的是与批量入口**同一个交易日历**（`adjusted_close(t+horizon) / adjusted_close(t) − 1`）。
- **日程为空**（窗口里没有任何合格决策日）明确拒绝，而不是让 IC 分析报「空输入」。
- **任一日期失败**（缓存损坏、行情缺失、单日因子报错）就**停止**：错误信息给出失败日期数量、具体日期与 batch manifest 路径，并且**不写**任何报告——不会静默丢掉失败日期只分析剩下的 Panel。已成功写入的 Panel 与 manifest 不回滚，它们是前一个阶段独立完成的成果。
- 质量报告仍然是**软提示**：不使分析失败，其影响体现在覆盖率、有效股票数与 IC 结果里。
- 退出码：`0` 全部成功且报告完整；`1` 有日期失败 / 标签或分析层硬拒绝 / 报告写入失败；`2` argparse 参数错误。stderr 只有一行 `error: ...`，不打印 traceback。

### 构造结果时的 batch 对账

`FactorResearchOutcome` 在构造时就做一次**对账**：不重算任何因子或收益，只核对「这次分析的样本」与「batch 记录」是否同一批数据：

```text
batch_outcome.selected_dates
  = batch_outcome.outcomes 里各期成功 Panel 的日期（按顺序、一期一个）
  = FactorResearchOutcome.factor_panels 的日期
  = forward_return_panels / panel_statistics / ic_observations / quantile_spreads 的日期
```

以下情形在构造时被拒绝（`FactorResearchError`）：

- batch 记录了**失败日期**——报告要么覆盖全部选中日期，要么一份都不写；
- batch 的 manifest 没有写成功——报告 metadata 指向它，发布出去就无法追溯；
- 选中日期数与成功 Panel 数不等，或某个选中日期在 batch 里没有对应结果；
- `factor_panels` 不是 batch 里那**同一批 Panel 对象**：同日期、同因子、同版本、数值也相同的「重建品」属于另一份样本，报告就无法追溯到这次 batch；
- batch 的因子名 / 版本与 Panel 的因子名 / 版本不一致（报告里的 `factor_name` 取自 batch，否则会被悄悄改名）。

因此 `metadata.json` 的 `selected_dates`、batch manifest、IC 序列与分位收益可以被证明指向**完全同一组数据**，而不只是「长度碰巧相同」。

### 输出布局

每次成功分析都会发布一个**新的独立目录**（同一请求运行两次得到两份报告；报告目录不是 Panel 缓存键）：

```text
<data_root>/factors/research_runs/<research_run_id>/
  metadata.json               # 本次请求、日程、horizon、两套参数、batch manifest 路径、IC 汇总
  panel_statistics.csv        # QuickAnalyzer.to_frame(...)
  forward_returns.csv         # 本次每期标签：每只股票、每个日期、收益可用性与 horizon
  ic_observations.csv         # ICReport.to_frame()
  quantile_observations.csv   # QuantileReport.to_frame()
  quantile_spreads.csv        # QuantileReport.spread_to_frame()
```

- `metadata.json` 里路径相对 `data_root`、用 `/` 分隔；领域对象的 `None` 写成 JSON `null`，**绝不**写 `NaN` / `Infinity`（写入时用 `allow_nan=False` 兜底）。
- 报告先写入 report root 下的唯一临时目录，全部文件写完并检查存在性后，再用**一次原子重命名**变成正式目录；任一环节失败会清理临时目录，不会留下半成品报告，也不影响已生成的 Panel 与 manifest。
- 终端只打印稳定摘要（因子、版本、区间、频率、anchor、horizon、quantiles、min_pairs、`computed/cached/forced` 计数、`valid_ic_count`、`mean_ic`、`ic_std`、`icir`、报告路径、manifest 路径），不打印整张表、逐股票标签或对象 repr：

```text
factor_name: momentum_12_1
factor_version: 1.0.0
start_date: 2024-01-01
end_date: 2024-01-31
frequency: weekly
anchor_weekday: THU
calendar_name: XNYS
horizon_sessions: 20
quantiles: 5
min_pairs: 2
selected_panel_count: 4
computed_count: 4
cached_count: 0
forced_count: 0
valid_ic_count: 4
mean_ic: -0.342857
ic_std: 0.328261
icir: -1.044466
report_path: data/factors/research_runs/20240102T030405Z_9f8e7d
batch_manifest_path: data/factors/batch_runs/20240102T030405Z_bf1eb3.json
```

### 研究口径与边界

1. **IC 是 Spearman 秩相关**：每期把因子值与未来收益各自按秩排序后求相关，`mean_ic` 是有效期内 `ic` 的均值，`ic_std` 是样本标准差（`ddof=1`），`icir = mean_ic / ic_std`（标准差未定义或为 0 时记 `None`）。IC 未定义（可用配对不足、或一侧是常数）的日期**保留**在表中，`ic` 为空。
2. **分位收益是研究分组统计**：每期把截面按因子值切成 `quantiles` 组，报告每组未来收益均值与 `top_minus_bottom`（最好一组减最差一组），空组保留（成员数为 0、均值为空），不是可交易组合收益。
3. **未来收益是复权收盘到收盘标签**：`adjusted_close(t+horizon)/adjusted_close(t)−1`，不代表未来回测里的开盘成交收益。
4. **重叠窗口**：`weekly` 信号配 20 日 horizon 时，相邻因子的未来窗口互相覆盖，IC 观测**不是独立样本**；因此 `ic_std` / `icir` 只能作为初步研究指标，不是严格显著性检验（本工具也不做 R²、横截面回归、t 检验或 Newey-West）。
5. **不能用存储目录代替本次日程**：把所有历史 Panel 自动混成一条研究序列会让采样口径失效；需要别的日期组合就再运行一次，日期由 `decision_schedule()` 决定。
6. **距离未来回测仍缺**：策略信号与组合构建、开盘成交、交易成本与滑点、换手与流动性约束、资金与仓位管理——本报告只是研究统计。
7. **尾部数据不足不是错误**：接近数据末端的日期会因为未来价格尚未发布而得到全部或部分 `is_available=False` 标签，这些日期照常保留（`pair_coverage` 为 0 或偏低、`ic` 为空），既不填零、不插值、也不用最近价格替代。

## 11. 测试

```powershell
.\.venv\Scripts\python.exe -m pytest            # 全部 1986 项（含 5 条既有 exchange_calendars 基线失败，见 §12）
.\.venv\Scripts\python.exe -m pytest -v         # 逐项
.\.venv\Scripts\python.exe -m ruff check .      # 静态检查
```

测试**完全不访问网络**：`yfinance.download` 始终被 monkeypatch，编排层测试使用 `FakeProvider`。覆盖重点：

- `test_universe.py` — 入库股票池自身的属性（必需字段、代码唯一性/格式、`common_stock` 限制、≥10 只、含 `BRK.B`），以及各类**异常配置**（文件缺失/目录/空文件/缺列/重复表头/重复代码/非法格式/未知交易所/非法证券类型/非法布尔值/规模不足）是否抛出带明确信息的 `UniverseError`；
- `test_storage.py` — Hive 布局、往返读写、空帧拒绝、sidecar、发布成功、**发布失败回滚到旧版本**、**确认报告问题 / 确认抛异常时回滚（含首次发布无备份可还原的情况）**、无暂存分区时报错；
- `test_update.py` — 增量更新端到端：首次回填、补缺口、重叠刷新、无操作运行、字段重算，以及下载 / 质量 / 暂存核验 / 发布确认四类失败都**不改变正式数据**（含首次发布失败后不留下任何分区）、运行元数据记录（用 `run_id` 精确定位）；另有**架构守卫**：注入的 `verify_dataset` 会断言目标路径里必须含 `.staging`（DuckDB 只核验暂存区）；
- `test_batch_query.py` — 批量端到端（内存 `SessionFakeProvider`，逐标的一次性注入）：按股票池**文件顺序**顺序执行且每个标的都落到正式 Parquet；**单标的下载失败与其余标的隔离**（batch 退出码为失败、失败标的保留旧分区、已成功标的的数据不被回滚）；batch 元数据字段（`batch_run_id`、请求标的、每标的状态/下载行数/发布日期范围、`dataset_summary`）；跨 `symbol=*` / `year=*` 多标的多年份的汇总（期望值用 PyArrow 直接按目录重建，不依赖 DuckDB）；
- `test_duckdb_verify.py` — 逐项核验失败路径（类型/行数/日期范围/重复/空值/年份覆盖）；
- `test_planner.py` / `test_market_calendar.py` / `test_merge.py` — 下载窗口的判定（首次回填 / 头部 / 内部缺口 / 尾部 / 重叠刷新 / 无会话）、NYSE 会话边界（跨周末与假日的「相邻」判定、区间裁剪、最近 N 个会话）、合并语义（同日新行胜出、派生列整段重算、边界行不残留旧 `daily_return`）；
- `test_summary.py` — 正式数据的 `(symbol, year)` 汇总（字段与排序、JSON 安全、就地替换 10 次后仍可读；与 DuckDB 汇总只在一个**新建**数据集上对照），以及**架构守卫**：`summary` 与 `batch_query` 的源码里绝不出现 `import duckdb`；
- `test_pipeline.py` — 编排成功路径与失败路径的退出码、元数据落盘、暂存区清理；
- `test_standardize.py` / `test_quality.py` / `test_yahoo.py` — 字段映射、质量规则、provider 重试与错误包装，以及**代码拼写回退**（`BRK.B` 先试规范拼写、失败后改试 `BRK-B`；交易所后缀 `RY.TO` 不会被改写；全部失败时报错会列出试过的每个拼写）；
- `test_run_factor.py` — 因子入口端到端（真实 Hive Parquet + 临时股票池）：面板与统计值用独立实现对照、`ParquetFactorStore.read` 读回一致、清洁与部分覆盖两种质量结论、未知因子 / 非交易日 / 行情缺失 / 字段缺失时**非零退出且不落盘**、`--data-root` 与 `--universe-path` 的实际作用范围、断网仍可运行；缓存三种状态（`miss` / `hit` / `forced`：命中时**不读行情、不跑 Runner、不调用 `write`**，文件字节与 mtime 不变；`--force` 时三者都真的被调用且只留一个文件、无 `.tmp` 残留；损坏的缓存文件失败且不被覆盖；周末即使有人为放置的同名文件也不会命中；未知 `--calendar` 在查询缓存前失败），以及**架构守卫**（不 import 公式模块、不重算公式或质量规则、不使用 Factor Store 私有成员也不复制其路径编码、没有日期区间参数）；
- `test_factor_schedule.py` — 日期规则（纯日历，无文件）：weekly 的 anchor 与「节假日回退到前一个交易日」、WED 与 THU 得到不同序列、候选落在区间外时被丢弃、biweekly 每隔一周且相位稳定、monthly 取每月最后一个交易日（3/6/8/11 月都提前一天）、跨年与月末边界；不变量（升序、无重复、全为会话、全在区间内）；各类非法输入（倒序区间、`datetime`、非严格 ISO、未知频率 / weekday / 日历、monthly 带 anchor）；手工构造的 `DecisionSchedule` 自校验（周末 `2024-01-06 (SAT)`、节假日 `2024-01-01 (MON)`、日历未覆盖的 1990 年、未知日历、超过 5 个非法日期的截断提示、空选择合法，其余字段矛盾仍被拒绝）；**架构守卫**（只允许 stdlib 与 `data_sys.market_calendar`，不 import pandas / PyArrow / DuckDB / Store / Runner / `factors` / `scripts`，不打开任何文件）；
- `test_run_factor_batch.py` — 批量入口：受控替身验证**每个选中日期只调用一次既有 `run_factor()`**、参数（因子 / 根目录 / 股票池 / 日历 / `force`）原样传递、`hit/miss/forced` 计数、单个日期失败后继续且顺序稳定、质量报告不算失败、未知因子 / 未知日历 / 空白 `--data-root` 在循环前失败；**每个选中日期恰好对应一条结果或一条失败**（缺口对象在构造时被拒绝、`format_summary()` 不会 `KeyError`、成功与失败混排仍能正常渲染）；manifest 字段与顺序、相对路径、空选择、两次运行两份独立 manifest、原子写失败不损坏旧文件且退出码为 `1`；真实 Parquet 端到端（`miss` → `hit` 文件字节不变、`--force` 全量重算、损坏的 Panel 只影响它自己那一天）；CLI 输出、退出码、argparse 的 `2`、从项目根目录真实进程运行；**架构守卫**（不 import 公式 / 网络 / 子进程，不重算 Panel 路径，只调用单日入口与日期规则）。
- `test_analyze_factor.py` — 多日期研究分析入口：受控替身验证**只调用既有 `run_factor_batch()`**（因子 / 区间 / 频率 / anchor / 根目录 / 股票池 / 日历 / `force` 原样传递）、只用本次 batch 的 Panel（目录里无关的 Panel 不会被读取、也不会被改写）、每个 Panel **恰好调用一次** `ForwardReturnCalculator.compute(panel, horizon)`、三种分析拿到同一组 panels / labels；真实 Parquet 端到端（`miss` → `hit` 文件字节不变、`--force` 全量重算、损坏的 Panel 使整次分析被拒绝且不写报告）；`weekly + THU + 20`（WED 与 THU 选到不同日期、biweekly 隔周、monthly 月末与无 anchor、monthly 带 anchor 退出码 `2`）、非法 horizon / quantiles / min-pairs（argparse 与函数两条路径）、空日程与失败日期（错误含失败数与 batch manifest 路径）、尾部行情不足的日期仍保留在报告中（`pair_coverage = 0`、`ic` 为空，不填零不删除）；报告的六份文件、列与既有 `to_frame()` 一致、metadata 的 `null`（不写 `NaN` / `Infinity`）与相对清单路径、两次运行两份独立报告、写入失败不留半成品目录且不影响 Panel 与 manifest；CLI 摘要顺序 / 退出码 / 单行 `error:` 风格、从项目根目录真实进程运行；**batch 对账**（结果对象在构造时核对 `selected_dates`、各期成功 outcome 的日期、`factor_panels` 的日期与**对象身份**：同日期同版本的重建 Panel、另一份 batch 的 Panel、缺一期的 Panel、被改名的 batch 都被拒绝，而真实运行的 Panel 与 batch 里的 Panel 是同一批对象）；**架构守卫**（不 import 公式 / 网络 / 子进程 / Store，不重算 Spearman 或分位分组，只构造报告路径）。

## 12. 已知限制与后续

**本 MVP 明确不包含**（属于设计文档后续阶段）：

- 并发与多进程写入：批量运行**严格顺序**执行，整个发布流程假设同一时刻只有一个写入进程（见下）；
- 公司行为（拆股 / 分红明细）与 SEC EDGAR 基本面；
- 证券主数据只覆盖股票池所需的 5 个字段：无 CIK、无退市历史、无 ticker 历史映射；
- `UniverseSnapshot` 的完整筛选规则（`dollar_volume` 等字段已就绪，筛选逻辑待在股票池模块实现）；
- `scripts/download_data.py` 的 Google Drive 整包同步（设计文档 §7）；
- 因子层的**策略与回测**：§10 只做研究统计（Panel + 复权收盘标签 + IC + 分位收益），策略信号、组合构建、开盘成交、成本与滑点、换手与流动性约束、资金与仓位管理都还没有；manifest 也仍然没有读取端；

**需要注意的行为**：

- 每个标的的分区是「**增量合并 + 整体替换**」：先把新下载的行并入已发布历史（`(symbol, date)` 同日时新行胜出），再整段重算 `daily_return` / `dollar_volume`，最后原子替换该 `symbol=` 分区。分区内容 = 已发布历史 ∪ 本次下载的窗口，因此增量更新既不丢历史，也不会引入重复行或空洞。
- **当前发布适合单写入进程**：不承诺支持并发读写（批量运行严格顺序执行，发布事务假设同一时刻只有一个写入进程）；同一个路径上的「发布替换」与「DuckDB 读取」也不要放进同一个进程。
- 正式数据在更新进程中**只用 PyArrow 读取**：DuckDB 会在进程内按路径缓存文件句柄，而重建 view 不会清空该缓存；读取刚被替换过的同一路径会原生崩溃（Windows 上表现为 `access violation`，共享连接时表现为卡死），因此 DuckDB 只用于唯一 `.staging/<run_id>/` 目录的核验。**正式 Parquet 的 DuckDB 查询应在独立、未执行发布替换的新进程中进行**（详见 §2.1）。
- Yahoo Finance 的数据许可限制见设计文档 §7.4：**在把数据再分发给协作者之前必须复核条款**，必要时改用有明确再分发许可的数据源。
- 标准化层不做交易日补齐（缺失交易日不会生成空行），因此 `date` 的严格递增校验是「单调」而非「连续」。
- 含点号的代码在 Yahoo 那边写作连字符（`BRK.B` → `BRK-B`）：provider **先按项目规范拼写请求**，只在返回空行或缺必需列时才回退到连字符拼写——规范拼写永远优先，`RY.TO` 这类交易所后缀不会被误改。代价是 `BRK.B` 每次更新会多出一次试探性请求（日志会写明 `resolved to the vendor ticker`）。文件名、标准化数据、Hive 分区与运行元数据里始终是 `BRK.B`，只有 sidecar 的 `vendor_symbol` 记录回退后的拼写。
- **测试基线**：当前环境安装的 `exchange_calendars` 把 `XNYS` 的首个会话定在 2006-10-26（项目文档与早期断言写作 2006-10-02），因此 `test_market_calendar.py` / `test_planner.py` / `test_update.py` 里有 5 条与此相关的既有失败。它们只涉及日历下界，与因子层无关，本次未修改。
- 因子层的缓存仍然**没有血缘**：`scripts/run_factor_batch.py` 只是逐日调用单日入口，所以批量运行同样看不到「行情 / 股票池 / 代码变过」，需要时用 `--force` 或升级 `factor_version`。
