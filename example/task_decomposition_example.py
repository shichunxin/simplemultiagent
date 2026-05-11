import asyncio
import uuid

from multiagent.core.agent import TaskDecompositionAgent, Request


def main():
    agent = TaskDecompositionAgent(agent_id="1001",config={})
    request = Request()
    request.session_id = uuid.uuid4().hex[:8]
    request.user_input = "我要写一篇关于军事题材的小说"
    asyncio.run(agent.process(request))

if __name__ == "__main__":
    main()