数据系统主要有两种操作：更新单只股票，或批量更新股票池。

先进入项目目录并激活环境：

```powershell
cd "D:\Python\Low_Frequency Trading"

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

看到命令行前出现 `(.venv)` 后，后续直接用 `python` 即可。若不激活环境，则把下方的 `python` 改为：

```powershell
.\.venv\Scripts\python.exe
```

## 1. 更新单只股票

例如下载或补齐 AAPL 在 2024 年的数据：

```powershell
python -m data_sys.data_query `
  --symbol AAPL `
  --start 2024-01-02 `
  --end 2024-12-31
```

默认会保留最近 5 个交易日的重叠下载，以修正 Yahoo 可能回溯调整过的数据。若明确不想重叠刷新：

```powershell
python -m data_sys.data_query `
  --symbol AAPL `
  --start 2024-01-02 `
  --end 2024-12-31 `
  --no-refresh
```

## 2. 批量更新指定股票

```powershell
python -m data_sys.batch_query `
  --symbols AAPL,MSFT,NVDA `
  --start 2024-01-02 `
  --end 2024-12-31
```

即使其中一只失败，其他股票仍会继续更新；批量命令最后会返回非零退出码，并在 metadata 中记录失败原因。

## 3. 更新整个默认股票池

```powershell
python -m data_sys.batch_query `
  --start 2020-01-01 `
  --end 2026-08-31
```

不传 `--symbols` 时，会读取：

```text
config/universe_seed.csv
```

并更新其中 `is_active=true` 的全部股票。

## 4. 日常增量更新

例如把当前股票池更新到指定最新日期：

```powershell
python -m data_sys.batch_query --end 2026-08-31
```

虽然默认起点写的是 `2020-01-01`，系统不会每次重新下载完整历史。它会检查本地已有数据，只下载：

```text
缺失的历史区间
+ 末尾新增日期
+ 最近 5 个交易日的重叠刷新
```

所以正常日常更新只会下载很小的尾部区间。

## 5. 查看生成结果

标准化后的正式行情：

```powershell
Get-ChildItem data\standardized\market_bars
```

原始下载数据：

```powershell
Get-ChildItem data\raw\market_bars
```

最新单股票运行元数据：

```powershell
Get-ChildItem data\metadata\run_*.json |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 |
  Get-Content
```

最新批量更新报告：

```powershell
Get-ChildItem data\metadata\batch_*.json |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 |
  Get-Content
```

数据最终以分区 Parquet 保存：

```text
data/standardized/market_bars/
  symbol=AAPL/year=2024/part-0.parquet
  symbol=MSFT/year=2024/part-0.parquet
```

这看似按股票、年份分开存，但读取层会自动把相关分区组合成统一长表。