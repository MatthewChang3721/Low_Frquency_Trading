是的。按当前约定的范围，Factor System 与快速有效性分析已完成：单日计算、批量生成、缓存、质量检查、基础统计、未来收益标签、IC、分位收益和报告输出都已具备。

未包含的仍是策略、组合构建、开盘成交、成本、回测，以及可选的因子变换和多因子合成。

## 启动环境

在项目根目录执行：

```powershell
cd "D:\Python\Low_Frequency Trading"

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

看到命令行前方出现 `(.venv)` 后，就可以直接使用 `python`。

查看所有参数：

```powershell
python scripts/run_factor.py --help
python scripts/run_factor_batch.py --help
python scripts/analyze_factor.py --help
```

## 1. 计算一个日期的单因子 Panel

```powershell
python scripts/run_factor.py `
  --factor momentum_12_1 `
  --as-of 2025-12-31
```

它会：

- 用当前股票池计算该日全部股票的动量值；
- 自动读取过去 253 个交易日所需行情；
- 运行质量检查；
- 输出该日 Quick Analysis 统计；
- 保存一个 Panel。

加 `--force` 会无视同日缓存、重新计算：

```powershell
python scripts/run_factor.py `
  --factor momentum_12_1 `
  --as-of 2025-12-31 `
  --force
```

Panel 保存位置：

```text
data/factors/panels/
  factor_name=momentum_12_1/
    factor_version=1.0.0/
      year=2025/
        as_of_date=2025-12-31.parquet
```

## 2. 批量生成 Panel

默认规则是“每周四；周四休市则前移至最近交易日”。

```powershell
python scripts/run_factor_batch.py `
  --factor momentum_12_1 `
  --start 2024-01-01 `
  --end 2025-12-31 `
  --frequency weekly `
  --anchor-weekday THU
```

其他频率示例：

```powershell
# 每两周一次；从请求区间内第一个有效周开始
python scripts/run_factor_batch.py `
  --factor momentum_12_1 `
  --start 2024-01-01 `
  --end 2025-12-31 `
  --frequency biweekly `
  --anchor-weekday THU

# 每月最后一个交易日
python scripts/run_factor_batch.py `
  --factor momentum_12_1 `
  --start 2024-01-01 `
  --end 2025-12-31 `
  --frequency monthly
```

它不会重复计算已存在的 Panel。每次运行还会写入日期清单：

```text
data/factors/batch_runs/<batch_run_id>.json
```

## 3. 一次完成 Quick Analysis、IC 与分位收益分析

例如：每周四形成动量信号，检验未来 20 个交易日收益。

```powershell
python scripts/analyze_factor.py `
  --factor momentum_12_1 `
  --start 2024-01-01 `
  --end 2025-12-31 `
  --frequency weekly `
  --anchor-weekday THU `
  --horizon-sessions 20
```

它会自动：

1. 按 schedule 补齐或读取所需 Panel；
2. 计算每期未来 20 个交易日复权收盘收益；
3. 输出每期基础统计；
4. 计算 Spearman IC、平均 IC、ICIR；
5. 计算五分位收益和 Q5 − Q1 spread；
6. 写出完整研究报告。

更改收益观察期只改 horizon：

```powershell
# 看未来一周
--horizon-sessions 5

# 看未来两周
--horizon-sessions 10

# 看未来约一个月
--horizon-sessions 20
```

报告目录：

```text
data/factors/research_runs/<research_run_id>/
  metadata.json
  panel_statistics.csv
  forward_returns.csv
  ic_observations.csv
  quantile_observations.csv
  quantile_spreads.csv
```

快速查看最近报告：

```powershell
Get-ChildItem data\factors\research_runs

Get-Content data\factors\research_runs\<research_run_id>\metadata.json

Import-Csv data\factors\research_runs\<research_run_id>\ic_observations.csv |
  Format-Table
```

注意：如果 `end` 接近本地行情的最后日期，最后若干 Panel 可能没有完整的未来 20 日收益。它们仍会保留在报告中，但 IC 会显示为不可用，而不会被伪造为零收益。

## 新增一个因子时的步骤

1. 先明确研究定义  
   写清楚公式、经济逻辑、所需字段、回看期与版本号。

2. 新建公式文件  
   例如：

```text
factors/value_earnings_yield.py
```

该文件只包含因子逻辑，不读取 Parquet、不访问网络、不写文件。

3. 定义 `FactorSpec`  
   需声明：

```text
name
version
required_fields
lookback_sessions
```

例如动量要求：

```text
required_fields = ("adjusted_close",)
lookback_sessions = 252
```

4. 实现 `compute(context)`  
   输入是已经准备好的 `FactorContext`；输出必须是 `FactorPanel`。

   每个股票都应有一行：

```text
raw_value      因子原始值
is_eligible    是否具备足够数据、可参与比较
```

数据不足时应使用：

```text
raw_value = NaN
is_eligible = False
```

不要填充价格、插值或读取未来数据。

5. 在注册表中注册  
   在 `factor_sys/registry.py` 的默认注册表中加一行注册，使 CLI 能通过名称找到它。

6. 添加测试  
   至少覆盖：

- 公式是否正确；
- 回看期是否正确；
- 缺失数据、短历史、异常值；
- 不读取未来数据；
- Runner 集成；
- 单日和多日期分析能运行。

7. 管理版本与缓存  
   如果公式含义、字段口径或参数发生实质变化，升级版本号，例如：

```text
1.0.0 → 1.1.0
```

如果只是想用最新数据重算既有版本，则加：

```powershell
--force
```

之后无需为新因子重写批量或分析工具：

```powershell
python scripts/run_factor_batch.py `
  --factor value_earnings_yield `
  --start 2024-01-01 `
  --end 2025-12-31 `
  --frequency weekly `
  --anchor-weekday THU

python scripts/analyze_factor.py `
  --factor value_earnings_yield `
  --start 2024-01-01 `
  --end 2025-12-31 `
  --frequency weekly `
  --anchor-weekday THU `
  --horizon-sessions 20
```