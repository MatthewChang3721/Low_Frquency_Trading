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
| `symbol` | 股票代码。必须非空、大写、**唯一**，且匹配 `data_sys.schema.SYMBOL_PATTERN`（`^[A-Z0-9.\-]+$`，因此 `BRK.B` 这类含点号的多类别代码合法），长度 ≤ 10。Yahoo 侧含点号的代码写作连字符（`BRK.B` 在供应商那里是 `BRK-B`），provider 会自动回退到该拼写，磁盘上仍是 `BRK.B`（见 §9） |
| `name` | 证券名称，非空 |
| `exchange` | 上市交易所，须属于 `ALLOWED_EXCHANGES`：`NASDAQ` / `NYSE` / `NYSE AMERICAN` / `NYSE ARCA` / `BATS` / `OTC` |
| `security_type` | 第一版**仅允许** `common_stock` |
| `is_active` | 布尔值，接受 `true/false`、`yes/no`、`y/n`、`1/0` |

额外列会被忽略（契约是**最小**列集合），便于后续加 `sector`、`cik` 等字段而不破坏现有校验。

### 3.2 读取接口

唯一入口是 `data_sys/universe.py`；下游模块只拿标准化后的结果，不接触 CSV 本身：

```python
from data_sys.universe import load_universe_symbols, read_universe

symbols = load_universe_symbols()          # 已标准化、去重、排序的 list[str]
symbols = load_universe_symbols(active_only=False)   # 含 is_active=false 的证券

universe = read_universe()                 # 完整证券主数据
universe.by_symbol()["AAPL"].exchange      # -> 'NASDAQ'
universe.symbols                           # 全部代码（排序、去重）
universe.active_symbols                    # 仅 is_active=true
universe.to_frame()                        # DataFrame，列顺序 == 字段契约顺序
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

分层依赖是单向的：`data_query` / `batch_query` → `update` → `planner` / `merge` / `storage` / `duckdb_verify` / `metadata` → `standardize` / `quality` / `market_calendar` → `schema` / `config` / `utils`；`summary` 只依赖 `storage`（因此这一层完全没有 DuckDB）。`storage` 与 `providers` 互不感知。`universe` 是独立于行情管道的一层（只依赖 `schema`/`config`/`errors`），由 `batch_query` 消费，行情管道本身不依赖它，因此两者可各自演进。

## 8. 测试

```powershell
.\.venv\Scripts\python.exe -m pytest            # 全部 261 项
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
- `test_standardize.py` / `test_quality.py` / `test_yahoo.py` — 字段映射、质量规则、provider 重试与错误包装，以及**代码拼写回退**（`BRK.B` 先试规范拼写、失败后改试 `BRK-B`；交易所后缀 `RY.TO` 不会被改写；全部失败时报错会列出试过的每个拼写）。

## 9. 已知限制与后续

**本 MVP 明确不包含**（属于设计文档后续阶段）：

- 并发与多进程写入：批量运行**严格顺序**执行，整个发布流程假设同一时刻只有一个写入进程（见下）；
- 公司行为（拆股 / 分红明细）与 SEC EDGAR 基本面；
- 证券主数据只覆盖股票池所需的 5 个字段：无 CIK、无退市历史、无 ticker 历史映射；
- `UniverseSnapshot` 的完整筛选规则（`dollar_volume` 等字段已就绪，筛选逻辑待在股票池模块实现）；
- `scripts/download_data.py` 的 Google Drive 整包同步（设计文档 §7）。

**需要注意的行为**：

- 每个标的的分区是「**增量合并 + 整体替换**」：先把新下载的行并入已发布历史（`(symbol, date)` 同日时新行胜出），再整段重算 `daily_return` / `dollar_volume`，最后原子替换该 `symbol=` 分区。分区内容 = 已发布历史 ∪ 本次下载的窗口，因此增量更新既不丢历史，也不会引入重复行或空洞。
- **当前发布适合单写入进程**：不承诺支持并发读写（批量运行严格顺序执行，发布事务假设同一时刻只有一个写入进程）；同一个路径上的「发布替换」与「DuckDB 读取」也不要放进同一个进程。
- 正式数据在更新进程中**只用 PyArrow 读取**：DuckDB 会在进程内按路径缓存文件句柄，而重建 view 不会清空该缓存；读取刚被替换过的同一路径会原生崩溃（Windows 上表现为 `access violation`，共享连接时表现为卡死），因此 DuckDB 只用于唯一 `.staging/<run_id>/` 目录的核验。**正式 Parquet 的 DuckDB 查询应在独立、未执行发布替换的新进程中进行**（详见 §2.1）。
- Yahoo Finance 的数据许可限制见设计文档 §7.4：**在把数据再分发给协作者之前必须复核条款**，必要时改用有明确再分发许可的数据源。
- 标准化层不做交易日补齐（缺失交易日不会生成空行），因此 `date` 的严格递增校验是「单调」而非「连续」。
- 含点号的代码在 Yahoo 那边写作连字符（`BRK.B` → `BRK-B`）：provider **先按项目规范拼写请求**，只在返回空行或缺必需列时才回退到连字符拼写——规范拼写永远优先，`RY.TO` 这类交易所后缀不会被误改。代价是 `BRK.B` 每次更新会多出一次试探性请求（日志会写明 `resolved to the vendor ticker`）。文件名、标准化数据、Hive 分区与运行元数据里始终是 `BRK.B`，只有 sidecar 的 `vendor_symbol` 记录回退后的拼写。
