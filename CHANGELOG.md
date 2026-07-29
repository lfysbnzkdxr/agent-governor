# Changelog

## v0.4.0 (2026-07-29)

### 新增
- 多租户管理模块：租户级预算、限流、模型白名单隔离
- 声明式策略引擎：支持 YAML/JSON 规则加载，budget/rate_limit/model_whitelist/max_cost_per_call 规则类型
- 策略引擎集成到 Governor 核心执行链路
- CLI 报告支持 --group-by tenant 租户分组

### 变更
- 废弃 `agent_governor.data.executors` 模块（将在 v1.0 移除）
- 废弃 `mcp` 和 `http` 可选依赖（将在 v1.0 移除）
- 升级项目版本至 v0.4.0

### 文档
- README 更新为中文完整版
- 新增 AGENTS.md 项目指引

---

## v0.3.0 (2026-06-xx)

### 新增
- OTel Tracing：Governor / Dispatcher / DAG 三层 Span，OTLP 导出
- OTel Metrics：调用计数、错误计数、预算超限、延迟直方图、token 直方图
- MCP 工具执行器：包装 mcp.ClientSession.call_tool()
- HTTP 通用执行器：httpx 连接池，REST API 调用
- 成本报告 CLI：agent-governor-report 命令行工具，支持 table/json 输出
- `@traced` 装饰器：自动为异步函数创建 OTel Span
- TracedExecutor：包装任意 Executor 添加 tracing 能力

### 修复
- Security Audit 工作流 pip-audit 参数错误

---

## v0.2.0 (2026-05-xx)

### 新增
- TaskGraph DAG 定义 DSL（基于 networkx）+ 环检测
- Kahn 分层拓扑排序，同层节点并发执行
- 质量门框架：QualityGate 协议 + ValidationPipeline 管道式串联
- 内置验证器：SchemaGate（Pydantic 模型校验）、BudgetGate（成本检查）、LLMJudgeGate（LLM 评分）
- 异步熔断器：三态状态机（CLOSED/OPEN/HALF_OPEN）+ 滑动窗口 + asyncio.Lock
- 并发控制：KeyedSemaphore 按路由键隔离 + TokenBucket 令牌桶限流
- State Manager：SQLite WAL 模式检查点，崩溃恢复
- Commander：高层任务 → DAG 构建 → 引擎执行

---

## v0.1.0 (2026-04-xx)

### 新增
- 项目脚手架：pyproject.toml、src-layout、CI、pre-commit
- 核心 Protocol 定义：Executor、QualityGate、StateBackend、AuditLoggerProtocol、CostTrackerProtocol
- Pydantic 数据模型：Task、TaskResult、CallRecord、TokenUsage、ValidationResult
- GovernorConfig 配置模型
- 异常体系：GovernorError、BudgetExceededError、ModelNotAllowedError、CircuitOpenError 等
- Dispatcher：封装 litellm Router，模型白名单 + 预算拦截 + 降级路由
- CostTracker：token 累计 + 模型价格表 + 预算告警
- AuditLogger：structlog 结构化 JSON 审计日志
- Governor 主入口 + @governed 装饰器
- 单元测试 + quickstart 示例
