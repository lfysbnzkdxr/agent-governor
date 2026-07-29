# AGENTS.md

This file provides guidance to Lingma (lingma.aliyun.com) when working with code in this repository.

## Build & Development Commands

```bash
# 环境初始化（需要 uv 工具）
uv venv --python 3.12
uv pip install -e ".[dev]"

# 运行全部测试
pytest tests/ -q

# 运行单个测试文件
pytest tests/unit/test_governor.py -q

# 运行单个测试函数
pytest tests/unit/test_governor.py::test_function_name -q

# 静态检查
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/

# 自动修复 lint 问题
ruff check --fix src/ tests/

# 构建文档
uv pip install -e ".[docs]"
mkdocs serve
```

## Architecture

三层架构，以 `Governor` 为主入口，通过 `Dispatcher` 连接各层：

```
Governor (governor.py)
├── control/     — DAG 编排、质量门校验、状态管理
├── data/        — 调度执行、熔断器、并发控制、外部执行器
├── policy/      — 声明式策略引擎（budget / rate_limit / model_whitelist）
├── tenant/      — 多租户预算隔离、计费报告
└── observability/ — Tracing、Metrics、审计日志、成本追踪
```

### 核心调用链

`Governor.execute()` → `Dispatcher.dispatch()` → `litellm.acompletion()`

Dispatcher 在调用前执行治理逻辑：模型白名单检查 → 预算预检 → 并发信号量 → 重试+降级路由 → 成本记录 → 审计日志。

### DAG 执行

`Governor.run_graph()` → `DagExecutionEngine.execute()`：基于 Kahn 拓扑排序分层并行执行，每节点经过质量门（QualityGate）管道校验。TaskGraph 内部用 networkx 管理图结构。

### 协议驱动扩展

`protocols.py` 定义 `typing.Protocol` 接口（Executor、QualityGate、Validator、StateBackend），第三方可结构化适配无需继承。

### 可选依赖按功能分组

- `dag`: networkx（DAG 编排）
- `telemetry`: opentelemetry（Tracing + Metrics）
- `mcp`: mcp SDK（MCP 工具执行器，已废弃，v1.0 移除）
- `http`: httpx（HTTP 执行器，已废弃，v1.0 移除）
- `redis`: redis[hiredis]（状态后端）

observability 层的 OTel 组件通过 try/except ImportError 实现优雅降级。

## Key Conventions

- Python 3.11+ 语法，`from __future__ import annotations` 全局使用
- Pydantic v2 BaseModel 用于配置和数据模型（`config.py`, `models.py`）
- structlog 结构化日志（非标准 logging）
- 异常体系以 `GovernorError` 为基类，所有自定义异常继承它
- pytest-asyncio `asyncio_mode = "auto"`，异步测试无需标记 `@pytest.mark.asyncio`
- 测试中通过 `unittest.mock.patch("litellm.acompletion")` mock LLM 调用
- ruff line-length=100，mypy strict 模式 + pydantic 插件
- pre-commit hooks: ruff (lint+format) + mypy
