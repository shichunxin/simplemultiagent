import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import field, dataclass
from enum import Enum
from typing import Any, Optional, Dict, List

from openai import OpenAI

from multiagent.config.appconfig import QIANWEN_URL, QIANWEN_KEY, QIANWEN_MODEL
from multiagent.core.prompt import TaskDecompositionPromptBuilder, AbilityTaskMatchingPromptBuilder, VotePromptBuilder


class Staus(Enum):
    """状态枚举"""
    SUCCESS = "success"
    FAILED = "failed"
    HANDOFF = "handoff"

class Request:
    """请求信息"""
    session_id:str
    user_input:str
    context : Optional[dict[str, Any]] = field(default_factory=dict)
    metadata: Optional[dict[str, Any]] = field(default_factory=dict)

class Response:
    """返回信息"""
    status:Staus = None
    message:str = None
    data:Optional[dict[str,Any]] = field(default_factory=dict)
    handoff_target:Optional[str] = None
    handoff_context:Optional[dict[str,Any]] = field(default_factory=dict)

class SubTask:
    """子任务"""
    id:str
    name:str
    required_capability:str

class Vote:
    """投票"""
    agent_id:str
    choice:str
    confidence:float
    reasoning:str

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    id:str
    type:str
    data:Any
    priority: int = 0
    status: TaskStatus = TaskStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskResult:
    """任务结果"""
    task_id:str
    agent_id:str
    data: Any
    success:bool = True
    error:Optional[str] = None
    metadata:Dict[str, Any] = field(default_factory=dict)
    timestamp:float = field(default_factory=time.time)

class BaseAgent(ABC):
    """Agent基类"""
    def __init__(self, agent_id:str,config:Dict[str,Any],capability=None) -> None:
        self.agent_id = agent_id
        self.config = config
        self.tools = config.get("tools",[])
        self.capability = capability
        self.current_load = 0

    @abstractmethod
    async def process(self, request:Request) -> Response:
        pass

    @abstractmethod
    def get_capabilities(self) -> list[str]:
        pass

    def should_handoff(self, request:Request) -> Optional[str]:
        return None

class TaskAgent(ABC):
    def __init__(self, agent_id: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self._running = False

    @abstractmethod
    async def process(self, task: Task) -> TaskResult:
        """处理任务的核心方法"""
        pass

    def can_handle(self, task: Task) -> bool:
        """判断是否能处理该任务"""
        return task.type in self.capabilities

    async def start(self):
        """启动Agent"""
        self._running = True
        print(f"Agent {self.agent_id} started")

    async def stop(self):
        """停止Agent"""
        self._running = False
        print(f"Agent {self.agent_id} stopped")

class TaskDecompositionAgent(BaseAgent):
    """任务分解Agent"""
    def __init__(self, agent_id:str,config:Dict[str,Any]) -> None:
        super().__init__(agent_id,config)

    async def process(self, request:Request) -> Response:
        taskPrompt = TaskDecompositionPromptBuilder()
        response = run_once(basePrompt=taskPrompt.build(),userPrompt=request.user_input)
        message = response.choices[0].message
        raw = message.content
        raw = raw.replace("```json", "").replace("```", "")
        tasks = json.loads(raw)
        print("[TaskDecompositionAgent]分解结果:")
        print(tasks)
        rsp = Response()
        rsp.status = Staus.SUCCESS,
        rsp.message = "",
        rsp.data = tasks,
        rsp.handoff_target = None,
        rsp.handoff_context = None
        return rsp

    def get_capabilities(self) -> list[str]:
        return ["任务规划与分解"]

    def should_handoff(self, request:Request) -> Optional[str]:
        return None

#能力任务匹配感知
class AbilityTaskMatchingAgent(BaseAgent):
    """能力任务匹配Agent"""
    def __init__(self, agent_id:str="",config:Dict[str,Any]=None,tasks:list[SubTask]=None,agents:list[BaseAgent]=None) -> None:
        self.agent_id = agent_id
        self.config = config
        self.tasks = tasks
        self.agents = agents

    async def process(self, request: Request) -> Response:
        tasks = json.dumps([{"task_id":task.id,"required":task.required_capability} for task in self.tasks],
                           ensure_ascii=False)
        agents = json.dumps(
        [{"agent_id":age.agent_id,"capabilities":age.get_capabilities(),"current_load":age.current_load} for age in self.agents],
            ensure_ascii=False)
        promptBuilder = AbilityTaskMatchingPromptBuilder(tasks,agents)
        basePrompt = promptBuilder.build()
        response = run_once(basePrompt=basePrompt, userPrompt=None)
        message = response.choices[0].message
        raw = message.content
        raw = raw.replace("```json", "").replace("```", "")
        matchJson = json.loads(raw)
        print("[AbilityTaskMatchingAgent]任务Agent匹配结果:")
        print(matchJson)
        rsp = Response()
        rsp.status = Staus.SUCCESS,
        rsp.message = "",
        rsp.data = matchJson,
        rsp.handoff_target = None,
        rsp.handoff_context = None
        return rsp

    def get_capabilities(self) -> list[str]:
        return ["给待处理任务匹配最佳Agent"]

class VoteAgent(BaseAgent):
    """投票Agent"""
    def __init__(self, agent_id:str,config:Dict[str,Any]=None,capability=None) -> None:
        self.agent_id = agent_id
        #self.agentRole = agentRole
        #self.question = quesstion
        #self.options = options
        self.config = config
        self.capability = capability

    async def process(self, request:Request) -> Response:
        promptBukilder = VotePromptBuilder()
        context = request.context
        prompt = promptBukilder.build(context["agentRole"],context["question"],context["options"])
        messages = [
            {
                "role":"system",
                "content":prompt,
            }
        ]
        response = run_once_messages(messages)
        message = response.choices[0].message
        raw = message.content
        raw = raw.replace("```json", "").replace("```", "")
        voteJson = json.loads(raw)
        print("[VoteAgent]投票返回:")
        print(voteJson)
        vote = Vote()
        vote.agent_id=self.agent_id
        vote.choice = voteJson["choice"]
        vote.confidence = voteJson["confidence"]
        vote.reasoning = voteJson["reasoning"]
        rsp = Response()
        rsp.status = Staus.SUCCESS,
        rsp.message = "",
        rsp.data = vote,
        rsp.handoff_target = None,
        rsp.handoff_context = None
        return rsp

class CommonAgent:
    def __init__(self,name:str,role:str,model:str,system_promtp:str) -> None:
        self.name = name
        self.role = role
        self.model = model
        self.system_promtp = system_promtp
        self.client = None
        self.history:list[dict] = []

    def process(self, message:str)->str:
        self.history.append({"role":"user","content":message})
        if self.client is None:
            self.client = OpenAI(base_url=QIANWEN_URL, api_key=QIANWEN_KEY)
        messages = [{"role":"system","content":self.system_promtp}]+self.history
        response = self.client.chat.completions.create(messages=messages, model=self.model)
        raw = response.choices[0].message.content
        self.history.append({"role":"assistant","content":raw})
        return raw

class WorkerAgent(ABC):
    def __init__(self,name:str,role:str,capability:str) -> None:
        self.name = name
        self.role = role
        self.capability = capability

    @abstractmethod
    def process(self,task:dict[str,Any])->str:
        pass

class ManageAgent:
    def __init__(self,workers:Dict[str,WorkerAgent]) -> None:
        self.workers = workers
        self.system_promtp = """
        ## 角色
        你是一个多Agent协同系统的管理者Manage Agent，负责协调和调度多个专业Worker Agent 完成复杂问题
        ## 职责
        1. 任务分析：理解用户需求，识别任务类型和复杂度
        2. 任务分解：将复杂问题拆分为可独立执行的字任务
        3. 智能分配：根据Worker能力描述匹配最合适的执行者
        4. 进度监控: 根据各子任务执行状态，处理异常情况
        5. 结果汇总：汇总Worker输出，生成完整的最终结果
        ## 可用Worker列表
        可用Worker：{workerList}
        ## 任务分配原则：
        - 每个子任务只分配给一个worker
        - 优先选择能力匹配度最高的Worker
        - 考虑Worker当前负载，实现负载均衡
        - 存在依赖关系的子任务需标明执行顺序
        ## 输出格式
        任务分解结果以JSON格式输出，包含subtask_id,description,assigned_worker,dependencies字段
        """
    async def decompose_task(self,task:str)->List[Dict[str,Any]]:
        worker_desc = ",".join([f"{k}:{v.capability}" for k,v in self.workers.items()])
        response =run_once_messages([
            {
                "role":"system",
                "content":self.system_promtp.format(workerList=worker_desc),
            },
            {
                "role":"user",
                "content":f"请分解任务：{task}",
            }
        ])
        return response.choices[0].message.content

    async def execute(self,task:str)->str:
        subtask = await self.decompose_task(task)
        result = await asyncio.gather(*[self.workers[st.get("assigned_worker")].process(st) for st in subtask])
        return await self._integrate_result(result)

    async def _integrate_result(self,results:list[str])->str:
        worker_desc = ",".join([f"{k}:{v.capability}" for k, v in self.workers.items()])
        response = run_once_messages([
            {
                "role": "system",
                "content": self.system_promtp.format(workerList=worker_desc),
            },
            {
                "role": "user",
                "content": f"Worker完成各自任务：{results},请结果汇总：生成完整的最终结果",
            }
        ])
        return response.choices[0].message.content


def run_once(basePrompt: str,userPrompt:str):
    messages = []
    if basePrompt:
        messages.append(
            {
            "role": "system",
             "content": basePrompt
            }
        )
    if userPrompt:
        messages.append(
            {
                "role": "user",
                "content": userPrompt,
            }
        )
    client = OpenAI(base_url=QIANWEN_URL, api_key=QIANWEN_KEY)
    response = client.chat.completions.create(messages=messages, model=QIANWEN_MODEL)
    return response

def run_once_messages(messages:list,retries=3,delay=5):
    client = OpenAI(base_url=QIANWEN_URL, api_key=QIANWEN_KEY)
    for i in range(retries):
        try:
            response = client.chat.completions.create(messages=messages, model=QIANWEN_MODEL)
            return response
        except Exception as e:
            print(f"API call failed on attempt {i + 1}/{retries}. Error: {e}")
            if i < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("All retries failed.")
                return None

def run_chat(messages:list):
    return run_once_messages(messages)