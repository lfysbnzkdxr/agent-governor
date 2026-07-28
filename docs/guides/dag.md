# DAG Workflows

Agent Governor supports multi-step task orchestration using Directed Acyclic
Graphs (DAGs). This allows you to define complex workflows where tasks depend
on each other, with automatic parallel execution of independent tasks.

## Basic DAG

```python
from agent_governor import Governor
from agent_governor.control.dag import TaskGraph
from agent_governor.models import Task

governor = Governor(budget_usd=5.0)

# Define tasks
task1 = Task(name="search", messages=[...], model="gpt-4o")
task2 = Task(name="analyze", messages=[...], model="gpt-4o")
task3 = Task(name="summarize", messages=[...], model="gpt-4o")

# Build graph
graph = TaskGraph("research_report")
graph.add_node("step1", task=task1)
graph.add_node("step2", task=task2, depends_on=["step1"])
graph.add_node("step3", task=task3, depends_on=["step2"])

# Execute
result = await governor.run_graph(graph)
print(result.status)
```

## With Quality Gates

```python
from agent_governor.control.validators.schema import SchemaValidator

# Add validation gates
schema = SchemaValidator()
schema.add_rule("content", "contains", "analysis")

graph.add_node("step2",
    task=task2,
    depends_on=["step1"],
    gate=schema,  # Validate output
)
```

## Parallel Execution

Tasks in the same layer (no dependencies between them) run in parallel:

```python
graph.add_node("search_a", task=search_a)
graph.add_node("search_b", task=search_b)  # Parallel with search_a
graph.add_node("merge", task=merge, depends_on=["search_a", "search_b"])
```

## Accessing Results

```python
for node_id, node_result in result.node_results.items():
    print(f"{node_id}: {node_result.status}")
    if node_result.result:
        print(f"  Cost: ${node_result.result.cost_usd:.6f}")
```
