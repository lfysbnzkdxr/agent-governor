# Agent Governor

轻量级 Agent 治理与执行中间件 —— 成本控制、审计追踪、模型路由、DAG 编排、可观测性。

## 安装

```bash
# 核心功能
pip install agent-governor

# 全部可选功能（DAG + OTel + MCP + HTTP）
pip install agent-governor[all]

# 按需选择
pip install agent-governor[dag]        # DAG 编排（networkx）
pip install agent-governor[telemetry]  # OTel Tracing + Metrics
pip install agent-governor[mcp]        # MCP 工具执行器
pip install agent-governor[redis]      # Redis 状态后端
```

## 快速开始

### 基础用法：预算控制 + 模型路由

```python
from agent_governor import Governor

governor = Governor(
    budget_usd=10.0,
    models=["gpt-4o", "qwen-plus"],
    fallback_model="qwen-turbo",
)

response = await governor.execute(
    messages=[{"role": "user", "content": "Hello!"}],
    model="gpt-4o",
)
```

### DAG 编排：多步骤工作流

```python
from agent_governor import Governor
from agent_governor.control import TaskGraph, SchemaGate

governor = Governor(budget_usd=5.0)

graph = TaskGraph("research_report")
graph.add_node("search", task=Task(name="web_search", metadata={"query": "..."}))
graph.add_node("analyze", task=Task(name="analyze"), depends_on=["search"])
graph.add_node("validate", task=Task(name="validate"), depends_on=["analyze"],
               quality_gates=[SchemaGate(ReportModel)])

result = await governor.run_graph(graph)
```

### 可观测性：OTel Tracing + Metrics

```python
governor = Governor(
    budget_usd=10.0,
    tracing_enabled=True,
    otlp_endpoint="http://localhost:4317",
)

# 使用 async context manager 确保资源释放
async with governor:
    result = await governor.execute(messages=[...], model="gpt-4o")
```

## 功能特性

### 核心治理

- **预算控制** — USD 预算上限，调用前拦截
- **模型白名单** — 限制可调用的模型范围
- **降级路由** — 主模型失败自动切换备用模型
- **审计日志** — 每次 LLM 调用的结构化 JSON 记录
- **成本追踪** — 按模型累计 token 用量与费用

### DAG 编排与韧性

- **TaskGraph** — 基于 networkx 的 DAG 定义 + 环检测
- **分层并行执行** — Kahn 拓扑排序，同层节点并发运行
- **质量门** — SchemaGate / BudgetGate / LLMJudgeGate 管道式校验
- **异步熔断器** — 三态状态机 + 滑动窗口，防止级联故障
- **并发控制** — 按路由键隔离的 Semaphore + 令牌桶限流
- **检查点恢复** — SQLite WAL 模式，崩溃后断点续执行
- **Commander** — 高层任务分解 → DAG 构建 → 引擎执行

### 可观测性与扩展

- **OTel Tracing** — Governor / Dispatcher / DAG 三层 Span，OTLP 导出
- **OTel Metrics** — 调用计数、错误计数、预算超限、延迟直方图、token 直方图
- **MCP 执行器** — 包装 `mcp.ClientSession.call_tool()`
- **HTTP 执行器** — httpx 连接池，通用 REST API 调用
- **成本报告 CLI** — `agent-governor-report` 命令行工具

## CLI 工具

```bash
# 查看最近 7 天的成本报告
agent-governor-report --file audit.jsonl --last 7d --group-by model

# JSON 格式输出
agent-governor-report --file audit.jsonl --format json
```

## 项目结构

```
src/agent_governor/
├── governor.py          # 主入口
├── config.py            # GovernorConfig 配置模型
├── models.py            # Task / TaskResult 数据模型
├── protocols.py         # Executor / QualityGate 协议定义
├── exceptions.py        # 异常体系
├── control/             # 控制层：DAG、质量门、状态管理
├── data/                # 数据层：调度、熔断、并发、执行器
├── observability/       # 可观测层：Tracing、Metrics、审计、成本
└── cli/                 # 命令行工具
```

## 开发

```bash
# 创建环境
uv venv --python 3.12
uv pip install -e ".[dev]"

# 运行测试
pytest tests/ -q

# 静态检查
ruff check src/ tests/
mypy src/
```

## 文档

```bash
# 本地构建文档站
uv pip install -e ".[docs]"
mkdocs serve
```

## 许可证

MIT
