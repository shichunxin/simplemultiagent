import copy
import json
import textwrap
import time
from dataclasses import field
from typing import Any, Dict

from markdown import Markdown
from rich.jupyter import display

from multiagent.core.agent import run_once_messages
from multiagent.core.mcp import create_mcp_message


class AgentRegistry:
    def __init__(self):
        self.registry:Dict[str,Any] = field(default_factory=dict)

    def register(self, name,agent):
        self.registry[name] = agent

    def get_handler(self, agent_name):
        handler = self.registry.get(agent_name)
        if not handler:
            raise ValueError(f"Agent '{agent_name}' not found in registry.")
        return handler

    def get_capabilities_description(self):
        return """
        Available Agents and their required inputs:

        1. AGENT: Librarian
           ROLE: Retrieves Semantic Blueprints (style/structure instructions).
           INPUTS:
             - "intent_query": (String) A descriptive phrase of the desired style or format.
           OUTPUT: The blueprint structure (JSON string).

        2. AGENT: Researcher
           ROLE: Retrieves and synthesizes factual information on a topic.
           INPUTS:
             - "topic_query": (String) The subject matter to research.
           OUTPUT: Synthesized facts (String).

        3. AGENT: Writer
           ROLE: Generates or rewrites content by applying a Blueprint to source material.
           INPUTS:
             - "blueprint": (String/Reference) The style instructions (usually from Librarian).
             - "facts": (String/Reference) Factual information (usually from Researcher). Use this for new content generation.
             - "previous_content": (String/Reference) Existing text (usually from a prior Writer step). Use this for rewriting/adapting content.
           OUTPUT: The final generated text (String).
        """

class ExecutionTrace:
    def __init__(self,goal):
        self.goal = goal
        self.plan = None
        self.steps = []
        self.status = "Initialized"
        self.final_output = None
        self.start_time = time.time()

    def log_plan(self, plan):
        self.plan = plan

    def log_step(self, step_num, agent, planned_input, mcp_output, resolved_input):
        self.steps.append({
            "step": step_num,
            "agent": agent,
            "planned_input": planned_input,
            "resolved_context": resolved_input,
            "output": mcp_output['content']
        })

    def finalize(self, status, final_output=None):
        self.status = status
        self.final_output = final_output
        self.duration = time.time() - self.start_time

    def display_trace(self):
        display(Markdown(
            f"### Execution Trace\n**Goal:** {self.goal}\n**Status:** {self.status} (Duration: {self.duration:.2f}s)"))
        if self.plan:
            display(Markdown(f"#### Plan:\n```json\n{json.dumps(self.plan, indent=2)}\n```"))

        display(Markdown("#### Execution Steps:"))
        for step in self.steps:
            print(f"--- Step {step['step']}: {step['agent']} ---")
            print("  [Planned Input]:", step['planned_input'])
            print("  [Output Snippet]:", textwrap.shorten(str(step['output']), width=150))
            print("-" * 20)

def planner(goal, capabilities):
    print("[Engine: Planner] 分析目标并生成执行计划...")
    system_prompt = f"""
    你是上下文引擎的战略核心。分析用户的高级目标，并使用可用的代理创建结构化执行计划。

    --- AVAILABLE CAPABILITIES ---
    {capabilities}
    --- END CAPABILITIES ---

    说明：
    1. 该计划必须是一个 JSON 对象列表，其中每个对象都是一个“步骤”。
    2. 您必须使用上下文链。如果某个步骤需要前一步的输入，请使用语法 $$STEP_X_OUTPUT$$ 引用它。
    3. 要有策略性。将复杂目标（例如顺序重写）分解为不同的步骤。对于 Writer 代理，请使用正确的输入键（'facts' 与 'previous_content'）。
    
    示例目标：“写一个关于阿波罗 11 号的悬疑故事。”
    示例计划（JSON 列表）:
    [
        {{"step": 1, "agent": "Librarian", "input": {{"intent_query": "suspenseful narrative blueprint"}}}},
        {{"step": 2, "agent": "Researcher", "input": {{"topic_query": "Apollo 11 landing details"}}}},
        {{"step": 3, "agent": "Writer", "input": {{"blueprint": "$$STEP_1_OUTPUT$$", "facts": "$$STEP_2_OUTPUT$$"}}}}
    ]

    示例目标：“撰写关于朱诺的技术报告，然后将其以随意的方式重写。”
    示例计划（JSON 列表）:
    [
        {{"step": 1, "agent": "Librarian", "input": {{"intent_query": "technical report structure"}}}},
        {{"step": 2, "agent": "Researcher", "input": {{"topic_query": "Juno mission technology"}}}},
        {{"step": 3, "agent": "Writer", "input": {{"blueprint": "$$STEP_1_OUTPUT$$", "facts": "$$STEP_2_OUTPUT$$"}}}},
        {{"step": 4, "agent": "Librarian", "input": {{"intent_query": "casual summary style"}}}},
        {{"step": 5, "agent": "Writer", "input": {{"blueprint": "$$STEP_4_OUTPUT$$", "previous_content": "$$STEP_3_OUTPUT$$"}}}}
    ]

    仅以 JSON 列表回复。
    """
    plan_json = ""
    try:
        messages = [
            {"role":"system","content":system_prompt},
            {"role":"user","content":goal},
        ]
        plan_json = run_once_messages(messages).choices[0].message.content
        plan = json.loads(plan_json)

        if not isinstance(plan, list):
             if isinstance(plan, dict):
                 if "plan" in plan and isinstance(plan["plan"], list):
                     plan = plan["plan"]
                 elif "steps" in plan and isinstance(plan["steps"], list): # <--- ADD THIS CHECK
                     plan = plan["steps"]
                 else:
                    raise ValueError("Planner returned a dict, but missing 'plan' or 'steps' key.")
             else:
                raise ValueError("Planner did not return a valid JSON list structure.")

        print("[Engine: Planner] Plan generated successfully.")
        return plan
    except Exception as e:
        print(f"[Engine: Planner] Failed to generate a valid plan. Error: {e}. Raw LLM Output: {plan_json}")
        raise e

def resolve_dependencies(input_params, state):
    resolved_input = copy.deepcopy(input_params)

    def resolve(value):
        if isinstance(value, str) and value.startswith("$$") and value.endswith("$$"):
            ref_key = value[2:-2]
            if ref_key in state:
                print(f"[Engine: Executor] Resolved dependency {ref_key}.")
                return state[ref_key]
            else:
                raise ValueError(f"Dependency Error: Reference {ref_key} not found in execution state.")
        elif isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve(v) for v in value]
        return value

    return resolve(resolved_input)

def context_engine(goal:str,registry)->None:
    print(f"\n=== [Context Engine] 开始新任务===\n目标: {goal}\n")
    trace = ExecutionTrace(goal)
    registry = registry
    try:
        capabilities = registry.get_capabilities_description()
        plan = planner(goal, capabilities)
        trace.log_plan(plan)
    except Exception as e:
        print(e)
    state = {}
    for step in plan:
        step_num = step.get("step")
        agent_name = step.get("agent")
        planned_input = step.get("input")

        print(f"\n[Engine: Executor] Starting Step {step_num}: {agent_name}")

        try:
            handler = registry.get_handler(agent_name)

            resolved_input = resolve_dependencies(planned_input, state)

            mcp_resolved_input = create_mcp_message("Engine", resolved_input)
            mcp_output = handler(mcp_resolved_input)

            output_data = mcp_output["content"]

            state[f"STEP_{step_num}_OUTPUT"] = output_data
            trace.log_step(step_num, agent_name, planned_input, mcp_output, resolved_input)
            print(f"[Engine: Executor] Step {step_num} completed.")

        except Exception as e:
            error_message = f"Execution failed at step {step_num} ({agent_name}): {e}"
            print(f"[Engine: Executor] ERROR: {error_message}")
            trace.finalize(f"Failed at Step {step_num}")
            return None, trace

        final_output = state.get(f"STEP_{len(plan)}_OUTPUT")
        trace.finalize("Success", final_output)
        print("\n=== [Context Engine] Task Complete ===")

        return final_output, trace

