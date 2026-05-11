import asyncio
import random
import time
from typing import Optional, Any, Dict

from multiagent.core.agent import BaseAgent, Request, Response, run_once_messages, Staus, run_once, TaskAgent, \
    TaskResult, Task
from multiagent.core.mcp import create_mcp_message


#小说专家
class FictionWritingExpertAgent(BaseAgent):
    def __init__(self, agent_id:str,config:Dict[str,Any]) -> None:
        self.agent_id = agent_id
        self.config = config
        self.tools = config.get("tools",[])
        self.current_load = 0

    async def process(self, request:Request) -> Response:
        history = request.context["message"]
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print(history)
        response = run_once_messages(history)
        message = response.choices[0].message
        request.context["message"].append({"role": f"{message.role}", "content": f"{message.content}"})
        print(f"#################agent_id:{self.agent_id}####################")
        raw = message.content
        print(raw)
        rsp = Response()
        rsp.status = Staus.SUCCESS,
        rsp.message = "",
        rsp.data = raw,
        rsp.handoff_target = None,
        rsp.handoff_context = None
        return rsp

    def get_capabilities(self) -> list[str]:
        return ["你是一个小说写作专家，善于写各种类型的小说"]

    def should_handoff(self, request:Request) -> Optional[str]:
        return None

#编程专家
class CodeWritingExpertAgent(BaseAgent):
    def __init__(self, agent_id:str,config:Dict[str,Any]) -> None:
        self.agent_id = agent_id
        self.config = config
        self.tools = config.get("tools",[])
        self.current_load = 0

    async def process(self, request:Request) -> Response:
        response = run_once(basePrompt=None,userPrompt=request.user_input)
        print(response)
        message = response.choices[0].message
        print(f"#################agent_id:{self.agent_id}####################")
        raw = message.content
        print(raw)
        rsp = Response()
        rsp.status = Staus.SUCCESS,
        rsp.message = "",
        rsp.data = message,
        rsp.handoff_target = None,
        rsp.handoff_context = None
        return rsp

    def get_capabilities(self) -> list[str]:
        return ["你是一个编程专家，擅长Java、Python、Go、JavaScript、TypeScript等语言开发"]

    def should_handoff(self, request:Request) -> Optional[str]:
        return None

#词作家
class SongWritingExpertAgent(BaseAgent):
    def __init__(self, agent_id:str,config:Dict[str,Any]) -> None:
        self.agent_id = agent_id
        self.config = config
        self.tools = config.get("tools",[])
        self.current_load = 0

    async def process(self, request:Request) -> Response:
        history = request.context["message"]
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print(history)
        response = run_once_messages(history)
        message = response.choices[0].message
        request.context["message"].append({"role":f"{message.role}","content":f"{message.content}"})
        print(f"#################agent_id:{self.agent_id}####################")
        raw = message.content
        print(raw)
        rsp = Response()
        rsp.status = Staus.SUCCESS,
        rsp.message = "",
        rsp.data = raw,
        #rsp.data = raw[0],
        rsp.handoff_target = None,
        rsp.handoff_context = None
        return rsp

    def get_capabilities(self) -> list[str]:
        return ["你是一个歌词写作专家，善于写各种类型的歌词"]

    def should_handoff(self, request:Request) -> Optional[str]:
        return None


class TeamAgent:
    """
    团队协调器
    """
    def __init__(self,request:Request,agents:dict[str,Any]):
        self.request = request
        self.agents = agents or {}

    def run(self,request:Request):
        analystsAgent = self.agents["analysts"]
        print("-----------分析师分析----------")
        framework = analystsAgent.process(f"请分析需求并制定调研框架:{request.user_input}")
        researcherAgent = self.agents["researcher"]
        print(framework)
        print("-----------研究员研究----------")
        researche = researcherAgent.process(f"调研框架:{framework} \n请执行调研框架")
        print(researche)
        reportWritingExpertAgent = self.agents["reportWritingExpert"]
        print("-----------撰写员总结----------")
        resport = reportWritingExpertAgent.process(f"研究素材:{researche} \n请生成报告")
        print(resport)
        return resport


class DataCollectorAgent(TaskAgent):

    def __init__(self, agent_id: str):
        super().__init__(agent_id, capabilities=["data_collection"])

    async def process(self, task: Task) -> TaskResult:
        await asyncio.sleep(random.uniform(0.5, 1.5))
        sources = ["database", "api", "file", "stream"]
        collected_data = {
            "source": random.choice(sources),
            "records": random.randint(100, 1000),
            "timestamp": time.time(),
            "raw_data": f"raw_data_from_{task.id}"
        }

        return TaskResult(
            task_id=task.id,
            agent_id=self.agent_id,
            data=collected_data,
            metadata={"collection_time": time.time()}
        )


class DataProcessorAgent(TaskAgent):
    def __init__(self, agent_id: str, processor_type: str = "general"):
        super().__init__(agent_id, capabilities=[f"data_processing_{processor_type}"])
        self.processor_type = processor_type

    async def process(self, task: Task) -> TaskResult:
        await asyncio.sleep(random.uniform(1, 2))

        input_data = task.data
        processed_data = {
            "original": input_data,
            "transformed": f"processed_by_{self.agent_id}",
            "quality_score": random.uniform(0.8, 1.0),
            "processor": self.processor_type
        }

        return TaskResult(
            task_id=task.id,
            agent_id=self.agent_id,
            data=processed_data,
            metadata={"processing_time": time.time()}
        )


class AnalyzerAgent(TaskAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id, capabilities=["analysis"])

    async def process(self, task: Task) -> TaskResult:
        await asyncio.sleep(random.uniform(1.5, 3))

        analysis_result = {
            "summary": f"Analysis of {task.data}",
            "insights": ["trend_detected", "anomaly_found"],
            "confidence": random.uniform(0.7, 0.99),
            "recommendations": ["action_1", "action_2"]
        }

        return TaskResult(
            task_id=task.id,
            agent_id=self.agent_id,
            data=analysis_result,
            metadata={"analysis_duration": time.time()}
        )


class DecisionAgent(TaskAgent):

    def __init__(self, agent_id: str):
        super().__init__(agent_id, capabilities=["decision"])

    async def process(self, task: Task) -> TaskResult:
        await asyncio.sleep(random.uniform(0.5, 1))

        decision = {
            "action": random.choice(["approve", "reject", "escalate"]),
            "confidence": random.uniform(0.6, 0.95),
            "reasoning": f"Based on analysis of {task.data}",
            "impact": "medium"
        }

        return TaskResult(
            task_id=task.id,
            agent_id=self.agent_id,
            data=decision,
            metadata={"decision_time": time.time()}
        )

class ResearcherAgent():
    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    async def process(self, mcp_input: Dict[str,Any]) ->Dict:
        simulated_database = {
            "mediterranean diet": "地中海饮食富含水果、蔬菜、全谷物、橄榄油和鱼类。研究表明，它与较低的心脏病风险、改善的大脑健康以及更长的寿命相关。"
        }
        research_topic = mcp_input['content']
        research_result = simulated_database.get(research_topic.lower(), "No information found.")
        system_prompt = "你是一名研究分析员。将提供的信息综合成三到四个简明要点。"
        messages = [
            {
                "role":"system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": research_result,
            }
        ]
        summary = run_once_messages(messages).choices[0].message.content
        return create_mcp_message(
            sender="ResearcherAgent",
            content=summary,
            metadata={"source": "Simulated Internal DB"}
        )

class WriterAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    async def process(self, mcp_input: Dict[str,Any]) ->Dict:
        research_summary = mcp_input['content']
        system_prompt = "你是一名内容撰稿人。请根据以下研究要点撰写一篇简短、有吸引力的博客文章（约150字），并为其起一个吸引眼球的标题。"
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": research_summary,
            }
        ]
        blog_post = run_once_messages(messages).choices[0].message.content
        return create_mcp_message(
            sender="WriterAgent",
            content=blog_post,
            metadata={"word_count": len(blog_post.split())}
        )

class ValidatorAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    async def process(self, mcp_input: Dict[str,Any]) ->Dict:
        source_summary = mcp_input['content']['summary']
        draft_post = mcp_input['content']['draft']
        system_prompt = """
        你是一个细致的事实核查员。判断“DRAFT”是否与“SOURCE SUMMARY”在事实上一致。
        - 如果DRAFT中的所有声明都得到SOURCE的支持，则仅回复“通过”。
        - 如果DRAFT包含SOURCE中没有的任何信息，则回复“未通过”并提供一句话的解释。
        """
        validation_context = f"SOURCE SUMMARY:\n{source_summary}\n\nDRAFT:\n{draft_post}"
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": validation_context,
            }
        ]
        validation_result = run_once_messages(messages).choices[0].message.content
        return create_mcp_message(
            sender="ValidatorAgent",
            content=validation_result
        )


