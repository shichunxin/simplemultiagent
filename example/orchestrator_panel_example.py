import asyncio
import json

from multiagent.core.agent import Task
from multiagent.core.panel import OrchestratorPanel, WorkflowManager
from multiagent.expert.agent import DataCollectorAgent, DataProcessorAgent, AnalyzerAgent, DecisionAgent


async def example_data_pipeline():
    """示例：数据处理管道"""
    print("=" * 60)
    print("Example 1: Data Processing Pipeline")
    print("=" * 60)

    # 1. 创建协调器
    orchestrator = OrchestratorPanel()

    # 2. 注册Agents
    collector = DataCollectorAgent("collector_1")
    processor_a = DataProcessorAgent("processor_a", "type_a")
    processor_b = DataProcessorAgent("processor_b", "type_b")
    analyzer = AnalyzerAgent("analyzer_1")

    for agent in [collector, processor_a, processor_b, analyzer]:
        orchestrator.register_agent(agent)

    collection_task = Task(
        id="task_001",
        type="data_collection",
        data={"source": "database", "query": "SELECT * FROM users"}
    )

    processing_task = Task(
        id="task_002",
        type="data_processing_general",
        data={"action": "clean", "method": "normalize"}
    )

    analysis_task = Task(
        id="task_003",
        type="analysis",
        data={"metric": "user_engagement", "period": "7d"}
    )

    # 4. 处理任务
    print("\n--- Processing Collection Task ---")
    result1 = await orchestrator.process_task(collection_task, "capability_match")
    print(f"Result: {json.dumps(result1, indent=2, default=str)}")

    print("\n--- Processing Task ---")
    result2 = await orchestrator.process_task(processing_task, "broadcast")
    print(f"Result: {json.dumps(result2, indent=2, default=str)}")

    print("\n--- Processing Analysis Task ---")
    result3 = await orchestrator.process_task(analysis_task, "capability_match")
    print(f"Result: {json.dumps(result3, indent=2, default=str)}")


async def example_parallel_execution():
    """示例：并行执行多个Agent"""
    print("\n" + "=" * 60)
    print("Example 2: Parallel Multi-Agent Execution")
    print("=" * 60)

    orchestrator = OrchestratorPanel()

    # 注册多个相同类型的Agent
    for i in range(5):
        agent = DataProcessorAgent(f"processor_{i + 1}", f"type_{i % 2}")
        orchestrator.register_agent(agent)

    # 创建任务并广播给所有Agent
    task = Task(
        id="parallel_task",
        type="data_processing_type_0",
        data={"batch": "large_dataset", "size": 10000}
    )

    # 使用broadcast策略让所有能处理的Agent并行工作
    result = await orchestrator.process_task(task, "broadcast")
    print(f"Parallel execution result: {json.dumps(result, indent=2, default=str)}")


async def example_workflow():
    """示例：复杂工作流"""
    print("\n" + "=" * 60)
    print("Example 3: Complex Workflow with Dependencies")
    print("=" * 60)

    orchestrator = OrchestratorPanel()
    workflow_manager = WorkflowManager(orchestrator)

    # 注册Agents
    collector = DataCollectorAgent("collector_1")
    processor = DataProcessorAgent("processor_1")
    analyzer = AnalyzerAgent("analyzer_1")
    decider = DecisionAgent("decider_1")

    for agent in [collector, processor, analyzer, decider]:
        orchestrator.register_agent(agent)

    # 定义工作流任务
    tasks = [
        Task(id="collect", type="data_collection", data={"source": "api"}),
        Task(id="process", type="data_processing_general", data={"action": "clean"}),
        Task(id="analyze", type="analysis", data={"scope": "full"}),
        Task(id="decide", type="decision", data={"options": ["A", "B", "C"]})
    ]

    # 定义依赖关系
    dependencies = {
        "process": ["collect"],  # process依赖collect
        "analyze": ["process"],  # analyze依赖process
        "decide": ["analyze"]  # decide依赖analyze
    }

    # 定义工作流
    workflow_manager.define_workflow("data_pipeline", tasks, dependencies)

    # 执行工作流
    result = await workflow_manager.execute_workflow("data_pipeline")
    print(f"Workflow result: {json.dumps(result, indent=2, default=str)}")


async def main():
    print("=" * 60)
    print("MULTI-AGENT ORCHESTRATION SYSTEM")
    print("Pattern: Dispatch -> Parallel -> Aggregate")
    print("=" * 60)

    await example_data_pipeline()

    print("=" * 60)
    print("执行并发")
    await example_parallel_execution()
    print("=" * 60)
    print("执行工作流")
    await example_workflow()

if __name__ == "__main__":
    asyncio.run(main())