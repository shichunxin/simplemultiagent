import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Callable

from numpy.core.defchararray import capitalize

from multiagent.core.agent import Request, TaskDecompositionAgent, SubTask, AbilityTaskMatchingAgent, BaseAgent, \
    run_chat, Vote, VoteAgent, Task, TaskResult, TaskAgent
from multiagent.core.prompt import ReplanRromptBuilder


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class PlanStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Step:
    id:str
    description:str
    action:str
    parameters: Dict[str, Any] = field(default_factory=dict)
    status:StepStatus = StepStatus.PENDING
    result:Any = None
    error:Optional[str] = None
    retry_count: int = 0

@dataclass
class Plan:
    goal:str
    steps:list[Step] = field(default_factory=list)
    current_step_index:int = 0
    status:str = PlanStatus.ACTIVE
    context:dict[str,Any] = field(default_factory=dict)

class AgentStatus(Enum):
    """Agent执行状态枚举"""
    IDLE = "idle"          # 空闲，未开始执行
    RUNNING = "running"    # 正在执行
    PAUSED = "paused"      # 暂停（等待恢复）
    SUCCESS = "success"    # 执行成功
    FAILED = "failed"      # 执行失败
    RETRYING = "retrying"  # 正在重试

@dataclass
class AgentState:
    status: AgentStatus = AgentStatus.IDLE
    current_step: int = 0  # 当前执行到第几步
    total_steps: int = 0  # 总步骤数
    context: Dict[str, Any] = field(default_factory=dict)  # 执行上下文（中间结果）
    error_log: List[Dict[str, Any]] = field(default_factory=list)  # 错误历史
    retry_count: int = 0  # 当前步骤重试次数
    max_retries: int = 3
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        if "status" in data and isinstance(data["status"], str):
            data["status"] = AgentStatus(data["status"])
        return cls(**data)

    def save_checkpoint(self,filepath="agent_checkpoint.json"):
        with open(filepath, "w",encoding="utf-8") as f:
            json.dump(self.to_dict(), f,indent=2, ensure_ascii=False)

    @classmethod
    def load_checkpoint(cls,filepath="agent_checkpoint.json")->Optional['AgentState']:
        try:
            with open(filepath,"r",encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            print(f"加载检查点失败: {e}")
            return None


def get_agent_by_taskid(task_id:str,matchs:list,subagents:list):
    if not matchs or not subagents:
        return None
    agent_map = {sub.agent_id: sub for sub in subagents}

    for m in matchs:
        if m.get("task_id") == task_id:
            agent_id = m.get("agent_id")
            if agent_id is not None and agent_id in agent_map:
                return agent_map[agent_id]

    return None

#任务规划器
class SimpleTaskDecompositionPanel():
    def __init__(self, id:str,config:Dict[str,Any],request:Request,subagents:list[BaseAgent]) -> None:
        self.id = f"panel_{id}"
        self.config = config
        self.request = request
        self.subagents = subagents

    def run(self,request:Request):
        agent_id = f"agent_{uuid.uuid4().hex[:6]}"
        agent = TaskDecompositionAgent(agent_id=agent_id, config={})
        req_user_input = request.user_input
        rsp = asyncio.run(agent.process(request))
        data = rsp.data[0]
        print("[SimpleTaskDecompositionPanel]任务规划结果:")
        print(data)
        sub_tasks = data["subtasks"]
        subtasks = []
        for sub_task in sub_tasks:
            sub = SubTask()
            sub.id = sub_task["id"],
            sub.name = sub_task["name"],
            sub.required_capability = sub_task["description"]
            subtasks.append(sub)

        match_agent_id = f"agent_{uuid.uuid4().hex[:6]}"
        ability_task_matching = AbilityTaskMatchingAgent(agent_id=match_agent_id, config={}, tasks=subtasks,
                                                         agents=self.subagents)
        match_request = Request()
        match_request.session_id = request.session_id
        match_request.user_input = "给待处理任务匹配最佳Agent"
        rsp = asyncio.run(ability_task_matching.process(match_request))
        print("[SimpleTaskDecompositionPanel]任务Agent匹配结果:")
        print(rsp)
        history = [{"role": "user", "content": req_user_input}]
        last_task_result = req_user_input
        for sub_task in sub_tasks:
            history.append({"role": "system", "content": sub_task["description"]})
            sub_agent = get_agent_by_taskid(str(sub_task["id"]), rsp.data[0], self.subagents)
            req = Request()
            req.session_id = request.session_id
            req.context = {}
            history.append({"role": "user", "content": last_task_result})
            req.context["message"] = history
            writingExpertRsp = asyncio.run(sub_agent.process(req))
            last_task_result = writingExpertRsp.data[0]
            lastMessage = req.context["message"][-1]
            history.append({"role": f"{lastMessage['role']}", "content": f"{lastMessage['content']}"})

        print(f"[SimpleTaskDecompositionPanel]最终结果:{last_task_result}")

#含有重规划器的任务规划器
class SimpleReplanPanel:
    def __init__(self, id:str,config:Dict[str,Any],request:Request,subagents:list[BaseAgent],tools:list,max_retries=3,max_steps=30) -> None:
        self.id = f"panel_{id}"
        self.config = config
        self.request = request
        self.subagents = subagents
        self.tools = tools
        self.max_retries = max_retries
        self.max_steps = max_steps
        self.current_plan: Optional[Plan] = None
        self.execution_log: List[Dict] = []
        self._task_agent_matching = None

    def run(self,request:Request):
        task_decomposition_agent = TaskDecompositionAgent(agent_id=f"agent_{uuid.uuid4().hex[:6]}",config=self.config)
        initial_steps_rsp = asyncio.run(task_decomposition_agent.process(request))
        initial_steps_data = initial_steps_rsp.data[0]["subtasks"]
        subtasks = []
        for sub_task in initial_steps_data:
            sub = SubTask()
            sub.id = sub_task["id"],
            sub.name = sub_task["name"],
            sub.required_capability = sub_task["description"]
            subtasks.append(sub)
        match_agent_id = f"agent_{uuid.uuid4().hex[:6]}"
        ability_task_matching = AbilityTaskMatchingAgent(agent_id=match_agent_id, config={}, tasks=subtasks,
                                                         agents=self.subagents)
        match_request = Request()
        match_request.session_id = request.session_id
        match_request.user_input = "给待处理任务匹配最佳Agent"
        rsp = asyncio.run(ability_task_matching.process(match_request))
        self._task_agent_matching = rsp.data[0]
        self.current_plan = Plan(goal=request.user_input)
        self._load_steps(initial_steps_data)
        self.request.context["message"] = [{"role": "user", "content": request.user_input}]
        step_count = 0
        while self.current_plan.status == PlanStatus.ACTIVE:
            if step_count >= self.max_steps:
                self.current_plan.status = PlanStatus.FAILED
                print("[Agent] 达到最大步数限制，任务终止。")
                break
            current_step = self._get_next_pending_step()
            if not current_step:
                self.current_plan.status = PlanStatus.COMPLETED
                print("[Agent] 所有步骤执行完毕。")
                break
            success = self._execute_step(current_step)
            if not success:
                replan_success = self._handle_failure_and_replan(current_step)
                if not replan_success:
                    self.current_plan.status = PlanStatus.FAILED
                    print("[Agent] 重规划失败或达到重试上限，任务终止。")
                    break
            step_count += 1
            time.sleep(0.5)


    def _load_steps(self,steps_data: List[Dict]):
        if self.current_plan.steps:
            completed = [s for s in self.current_plan.steps if s.status == StepStatus.COMPLETED]
            self.current_plan.steps = completed
        for step in steps_data:
            step_id = step["id"]
            if any(s.id == step_id for s in self.current_plan.steps):
                step_id = f"{step_id}_new_{int(time.time())}"
            step_name = step["name"],
            step_required_capability = step["description"]

            new_step = Step(
                id=step_id,
                description=step_required_capability,
                action=step.get('action','plan'),
                parameters=step.get('parameters', {})
            )
            self.current_plan.steps.append(new_step)
        self._update_current_index()

    def _get_next_pending_step(self)->Optional[Step]:
        steps = self.current_plan.steps
        for step in steps:
            if step.status in (StepStatus.PENDING,StepStatus.FAILED):
                return step
        return None

    def _execute_step(self,step:Step) ->bool:
        print(f"\n[Executor] 执行步骤: {step.description} (Action: {step.action})")
        rst = _execute_step_once(step)
        return rst

    def _handle_failure_and_replan(self,step:Step)->bool:
        print(f"\n[Replanner] 处理步骤 '{step.id}' 的失败...")
        if step.retry_count >= self.max_steps:
            print(f"[Replanner] 步骤 '{step.id}' 已达到最大重试次数 ({self.max_retries})。")
            return False
        try:
            replan_prompt_builder = ReplanRromptBuilder()
            current_plan_str = _steps_to_json(self.current_plan.steps)
            execution_log_str = json.dumps(self.execution_log, ensure_ascii=False, indent=2)#json.dumps(self.execution_log)
            replan_prompt = replan_prompt_builder.build(self.current_plan.goal,current_plan_str,execution_log_str)
            print(f"replan_prompt:{replan_prompt}")
            replan_messages = [
                {
                    "role": "system","content": replan_prompt
                }
            ]
            replan_response = run_chat(replan_messages)
            raw = replan_response.choices[0].message.content
            raw = raw.replace("```json", "").replace("```", "")
            print(f"replan_run_chat:{raw}")
            replan_tasks = json.loads(raw)

            if not replan_tasks:
                print("[Replanner] LLM 未生成新的计划步骤。")
                return False
            subtasks = replan_tasks["subtasks"]
            self._load_steps(subtasks)
            print("[Replanner] 新计划已加载，将继续执行。")
            return True

        except Exception as e:
            print(f"[Replanner] 重规划过程出错: {e}")
            return False

    def _resolve_parameters(self,parameters:Dict[str, Any]):
        resolved = {}
        for k, v in parameters.items():
            if isinstance(v, str) and v.startswith("{{") and v.endswith("}}"):
                key = v[2:-2]
                val = self.current_plan.context.get(key, None)
                if val is None:
                    for s in self.current_plan.steps:
                        if s.status == StepStatus.COMPLETED and s.result:
                            if key in s.description.lower() or key == "result":
                                val = s.result
                                break
                resolved[k] = val if val is not None else v
            else:
                resolved[k] = v
        return resolved

    def _update_current_index(self):
        for i,s in enumerate(self.current_plan.steps):
            if s.status != StepStatus.COMPLETED:
                self.current_plan.current_step_index = i
                return

class RecoveryPanel:
    """"
    错误恢复
    """
    def __init__(self,config:Dict[str,Any],request:Request,subagents:list[BaseAgent],tools:list,plan:Plan,max_retries=3,max_steps=30):
        self.id = f"panel_{uuid.uuid4().hex[:4]}"
        self.config = config
        self.request = request
        self.subagents = subagents
        self.tools = tools
        self.max_retries = max_retries
        self.max_steps = max_steps
        self.current_plan: Optional[Plan] = plan
        self.execution_log: List[Dict] = []
        self.state = AgentState()

    def run(self,resume: bool = False)->bool:
        if resume:
            loaded_state = AgentState.load_checkpoint()
            if loaded_state:
                self.state = loaded_state
                if self.state.status == AgentStatus.FAILED:
                    self.state.status = AgentStatus.RETRYING
                    self.state.retry_count += 1
            else:
                self.state = AgentState()
        else:
            self.state = AgentState()
        self.state.total_steps = len(self.current_plan.steps)
        self.state.status = AgentStatus.RUNNING
        start_step = self.state.current_step
        for i in range(start_step,len(self.current_plan.steps)):
            step_info = self.current_plan.steps[i]
            self.state.current_step = i + 1
            try:
                result = self._execute_step(step_info)
                self.state.context[f"step_{self.state.current_step}_result"] = result
                self.state.retry_count = 0
                self.state.save_checkpoint()
            except Exception as e:
                error_record = {
                    "step": self.state.current_step,
                    "timestamp": time.time(),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "retry_count": self.state.retry_count
                }
                self.state.error_log.append(error_record)
                if self.state.retry_count >= self.state.max_retries:
                    self.state.status = AgentStatus.FAILED
                    self.state.save_checkpoint()
                    return {
                        "success": False,
                        "completed_steps": self.state.current_step - 1,
                        "failed_step": self.state.current_step,
                        "error": str(e),
                        "state": self.state.to_dict()
                    }
                self.state.status = AgentStatus.FAILED
                self.state.retry_count += 1
                self.state.save_checkpoint()
                return {
                    "success": False,
                    "completed_steps": self.state.current_step - 1,
                    "failed_step": self.state.current_step,
                    "error": str(e),
                    "state": self.state.to_dict(),
                    "can_resume": True
                }
        self.state.status = AgentStatus.SUCCESS
        import os
        if os.path.exists("agent_checkpoint.json"):
            os.remove("agent_checkpoint.json")

        return {
            "success": True,
            "completed_steps": self.state.total_steps,
            "context": self.state.context,
            "state": self.state.to_dict()
        }

    def _execute_step(self,step_info):
        rst = _execute_step_once(step_info)
        return rst

class MajorityVotePanel:
    """多方案投票"""
    def __init__(self,id:str,config:Dict[str,Any],request:Request):
        self.id = f"panel_{uuid.uuid4().hex[:4]}"
        self.config = config
        self.request = request

    def run(self,request:Request):
        voteAgentOne = VoteAgent(f"agent_{uuid.uuid4().hex[:4]}")
        requestOne = Request()
        requestOne.context["agentRole"] = "你是技术专家，擅长解决各种技术问题"
        requestOne.context["question"] = request.context["question"]
        requestOne.context["options"] = request.context["options"]
        rspOne = voteAgentOne.process(requestOne)
        voteOne = rspOne.data

        voteAgentTwo = VoteAgent(f"agent_{uuid.uuid4().hex[:4]}")
        requestTwo = Request()
        requestTwo.context["agentRole"] = "你是一个问题分析专家，擅长问题分析"
        requestTwo.context["question"] = request.context["question"]
        requestOne.context["options"] = request.context["options"]
        rspTwo = voteAgentTwo.process(requestTwo)
        voteTwo = rspTwo.data

        voteAgentThress = VoteAgent(f"agent_{uuid.uuid4().hex[:4]}")
        requestThree = Request()
        requestThree.context["agentRole"] = "你是解决方案专家，擅长各类解决方案制定"
        requestThree.context["question"] = request.context["question"]
        requestThree.context["options"] = request.context["options"]
        rspThree = voteAgentThress.process(requestThree)
        voteThree = rspThree.data;

        voteList = [voteOne, voteTwo, voteThree]
        weight = {}
        winner,wninerScore = self._weight_majority_vote(voteList,weight)
        return winner

    def _weight_majority_vote(self,votes:List[Vote],weights:Dict[str,float],threshold:float=0.6)->Tuple[str,float]:
        """
        加权投票聚合
        """
        scroes = {}
        for vote in votes:
            if vote.confidence >= threshold:
                weight = weights.get(vote.agent_id, 1.0)
                effective_weight = weight * vote.confidence
                scroes[vote.choice] = scroes.get(vote.choice, 0) + effective_weight

        winner = max(scroes, key=scroes.get)
        return winner,scroes[winner]

class OrchestratorPanel:
    """
    多智能体协调器
    实现分发(Dispatch) -> 并行(Parallel) -> 聚合(Aggregate)模式
    """
    def __init__(self):
        self.agents:dict[str,TaskAgent] ={}
        self.task_queue = asyncio.Queue()
        self.result_cache:Dict[str,list[TaskResult]] = {}
        self.execute_plan:Dict[str,list[str]] ={}
        self.aggregation_rules:Dict[str,Callable] ={}

    def register_agent(self,agent:TaskAgent):
        """
        注册Agent
        """
        self.agents[agent.agent_id] = agent
        print(f"已经注册Agent: {agent.agent_id} 能力是: {agent.capabilities}")

    def aggregation_rule(self,task_type:str,rule:Callable):
        """
        设置聚合规则
        """
        self.aggregation_rules[task_type] = rule
        print(f"已设置聚合规则，任务类型: {task_type}")

    async def dispatch_tasks(self,task: Task, strategy: str = "broadcast")->List[str]:
        """
        任务分配
        """
        assigned_agents = []
        if strategy == "broadcast":
            for agent_id,agent in self.agents.items():
                if agent.can_handle(task):
                    assigned_agents.append(agent_id)
        elif strategy == "round_robin":
            capability_agent = [agent_id for agent_id,agent in self.agents.items() if agent.can_handle(task)]
            selected_agent = capability_agent[hash(task.id)%len(capability_agent)]
            assigned_agents.append(selected_agent)
        elif strategy == "capability_match":
            for agent_id,agent in self.agents.items():
                if task.type in agent.capabilities:
                    assigned_agents.append(agent_id)
                    break
        elif strategy == "priority_based":
            best_agent = None
            min_load = float("inf")
            for agent_id,agent in self.agents.items():
                if agent.can_handle(task):
                    load = len(self.result_cache.get(agent_id,[]))
                    if load < min_load:
                        min_load = load
                        best_agent = agent_id
            if best_agent:
                assigned_agents.append(best_agent)
        self.execute_plan[task.id] = assigned_agents
        return assigned_agents

    async def execute_parallel(self,task:Task,agent_ids:List[str]) -> List[TaskResult]:
        """
        并行执行阶段
        多个Agent同时处理任务
        """
        async def execute_on_agent(agent_id:str)->TaskResult:
            agent = self.agents[agent_id]
            try:
                result = await agent.process(task)
                return result
            except Exception as e:
                return TaskResult(
                    task_id=task.id,
                    agent_id=agent_id,
                    data=None,
                    success=False,
                    error=str(e)
                )
        execute_tasks = [execute_on_agent(agent_id) for agent_id in agent_ids]
        results = await asyncio.gather(*execute_tasks,return_exceptions=True)
        processed_results = []
        for i,result in enumerate(results):
            if isinstance(result,Exception):
                processed_results.append(
                    TaskResult(
                        task_id=task.id,
                        agent_id=agent_ids[i],
                        data=None,
                        success=False,
                        error=str(result)
                    )
                )
            else:
                processed_results.append(result)
        self.result_cache[task.id] = processed_results
        return processed_results

    async def aggregate_results(self, task: Task, results: List[TaskResult]) -> Any:
        success_result = [result for result in results if result.success]
        failed_result = [result for result in results if not result.success]
        if task.type in self.aggregation_rules:
            aggregation_func = self.aggregation_rules[task.type]
            final_result = await aggregation_func(success_result, failed_result)
        else:
            final_result = await self._default_aggregation(success_result, failed_result,task)
        return final_result

    async def _default_aggregation(self,successful: List[TaskResult], failed: List[TaskResult],task: Task)->Dict[str, Any]:
        if not successful:
            return {
                "success": False,
                "error": "All agents failed",
                "failed_count": len(failed)
            }
        aggregated = {
            "success": True,
            "total_agents": len(successful) + len(failed),
            "successful_count": len(successful),
            "failed_count": len(failed),
            "results": {},
            "metadata": {},
            "timestamp": time.time()
        }
        for s in successful:
            aggregated["results"][s.agent_id] = s.data
            aggregated["metadata"].update(s.metadata)

        if len(successful) == 1:
            aggregated["final_output"] = successful[0].data
        else:
            # 多个结果需要合并
            aggregated["final_output"] = self._merge_results(
                [r.data for r in successful]
            )

        return aggregated

    def _merge_results(self, results: List[Any]) -> Any:
        if not results:
            return None
        if isinstance(results[0], dict):
            merged = {}
            for r in results:
                if isinstance(r, dict):
                    merged.update(r)
            return merged
        return results

    async def process_task(self,task: Task, dispatch_strategy: str = "capability_match")->Dict[str, Any]:
        assigned_agents = await self.dispatch_tasks(task,strategy=dispatch_strategy)
        if not assigned_agents:
            return {
                "success": False,
                "error": "No suitable agent found for task",
                "task_id": task.id
            }
        start_time = time.time()
        results = await self.execute_parallel(task, assigned_agents)
        execution_time = time.time() - start_time
        final_result = await self.aggregate_results(task, results)

        if isinstance(final_result, dict):
            final_result["execution_time"] = execution_time
            final_result["task_id"] = task.id

        return final_result


class WorkflowManager:
    """工作流管理器 - 支持复杂的有向无环图(DAG)工作流"""

    def __init__(self, orchestrator: OrchestratorPanel):
        self.orchestrator = orchestrator
        self.workflows: Dict[str, List[Task]] = {}
        self.dependencies: Dict[str, List[str]] = {}

    def define_workflow(self, name: str, tasks: List[Task], dependencies: Dict[str, List[str]] = None):
        """
        定义一个工作流
        dependencies: {task_id: [dependent_task_ids]}
        """
        self.workflows[name] = tasks
        if dependencies:
            self.dependencies.update(dependencies)

    async def execute_workflow(self, workflow_name: str) -> Dict[str, Any]:
        """执行工作流，支持任务依赖关系"""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow {workflow_name} not found")

        tasks = self.workflows[workflow_name]

        # 拓扑排序处理依赖
        execution_order = self._topological_sort(tasks)

        results = {}
        completed_tasks = set()

        for task in execution_order:
            # 检查依赖是否完成
            deps = self.dependencies.get(task.id, [])
            if deps:
                # 为有依赖的任务添加前置结果数据
                dep_results = {dep_id: results.get(dep_id) for dep_id in deps}
                task.metadata["dependencies_results"] = dep_results

            # 执行任务
            result = await self.orchestrator.process_task(task)
            results[task.id] = result
            completed_tasks.add(task.id)

        return {
            "workflow": workflow_name,
            "completed_tasks": list(completed_tasks),
            "results": results
        }

    def _topological_sort(self, tasks: List[Task]) -> List[Task]:
        """简单的拓扑排序"""
        # 这里实现简化版本，实际可以使用Kahn算法
        task_map = {t.id: t for t in tasks}
        in_degree = {t.id: 0 for t in tasks}

        for task in tasks:
            for dep_id in self.dependencies.get(task.id, []):
                if dep_id in in_degree:
                    in_degree[task.id] += 1

        # 简化处理：如果没有依赖，按原顺序返回
        if all(v == 0 for v in in_degree.values()):
            return tasks

        # BFS排序
        sorted_tasks = []
        queue = [t for t in tasks if in_degree[t.id] == 0]

        while queue:
            current = queue.pop(0)
            sorted_tasks.append(current)

            for task in tasks:
                if current.id in self.dependencies.get(task.id, []):
                    in_degree[task.id] -= 1
                    if in_degree[task.id] == 0:
                        queue.append(task)

        return sorted_tasks

class AdaptiveOrchestrator(OrchestratorPanel):
    """自适应协调器 - 能根据性能动态调整策略"""
    def __init__(self):
        super().__init__()
        self.performance_metrics: Dict[str, List[float]] = {}
        self.strategy_performance: Dict[str, Dict[str, float]] = {}

    async def process_task_adaptive(self, task: Task) -> Dict[str, Any]:
        """自适应任务处理"""
        # 根据历史性能选择最佳策略
        best_strategy = self._select_best_strategy(task)
        print(f"Selected strategy for task {task.id}: {best_strategy}")

        # 使用选定的策略处理
        result = await self.process_task(task, best_strategy)

        # 更新性能指标
        self._update_metrics(task, best_strategy, result.get("execution_time", 0))

        return result

    def _select_best_strategy(self, task: Task) -> str:
        """基于历史数据选择最佳策略"""
        if not self.strategy_performance:
            return "capability_match"

        # 为每种策略计算平均性能
        avg_performance = {}
        for strategy, metrics in self.strategy_performance.items():
            if metrics["count"] > 0:
                avg_performance[strategy] = metrics["total_time"] / metrics["count"]

        if not avg_performance:
            return "capability_match"

        # 返回性能最好的策略（最短执行时间）
        return min(avg_performance, key=avg_performance.get)

    def _update_metrics(self, task: Task, strategy: str, execution_time: float):
        """更新性能指标"""
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {"total_time": 0, "count": 0}

        self.strategy_performance[strategy]["total_time"] += execution_time
        self.strategy_performance[strategy]["count"] += 1


def _execute_step_once(self,step:Step):
    step.status = StepStatus.RUNNING
    try:
        params = self._resolve_parameters(step.parameters)
        if step.action == "final_answer":
            final_content = params.get('content', params.get('info', params.get('result', 'No data')))
            step.result = final_content
            step.status = StepStatus.COMPLETED
            self.current_plan.context['final_answer'] = final_content
            print(f"[Executor] 最终答案: {final_content}")
            return True
        if step.action == "tool":
            """
            result = self.tools.execute(step.action, **params)
            step.result = result
            step.status = StepStatus.COMPLETED

            self.current_plan.context[step.action] = result
            self.current_plan.context['last_result'] = result

            print(f"[Executor] 成功. 结果: {result}")
            self.execution_log.append({"step_id": step.id, "status": "success", "result": result})
            return True
            """
            return True
        if step.action == "plan":
            self.request.context["message"].append({"role": "system", "content": step.description})
            sub_agent = get_agent_by_taskid(str(step.id), self._task_agent_matching, self.subagents)
            req = Request()
            req.session_id = self.request.session_id
            req.context = {}
            req.context["message"] = self.request.context["message"]
            writingExpertRsp = asyncio.run(sub_agent.process(req))
            last_task_result = writingExpertRsp.data[0]
            step.result = last_task_result
            step.status = StepStatus.COMPLETED
            self.current_plan.context[step.action] = last_task_result
            self.current_plan.context['last_result'] = last_task_result
            self.request.context["message"].append({"role": "system", "content": last_task_result})
            self.execution_log.append({"step_id": step.id, "status": "success", "result": last_task_result})
            return True
        return False
    except Exception as e:
        step.error = str(e)
        step.status = StepStatus.FAILED
        step.retry_count += 1
        print(f"[Executor] 失败. 错误: {e}")
        self.execution_log.append({"step_id": step.id, "status": "failed", "error": str(e)})
        return False

def _step_encoder(obj):
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _steps_to_json(steps: List[Step]) -> str:
    step_dicts = [asdict(step) for step in steps]
    return json.dumps(step_dicts, default=_step_encoder, ensure_ascii=False, indent=2)



