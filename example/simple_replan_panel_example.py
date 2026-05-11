import uuid

from multiagent.core.agent import Request
from multiagent.core.panel import SimpleReplanPanel
from multiagent.expert.agent import FictionWritingExpertAgent, CodeWritingExpertAgent, SongWritingExpertAgent


def main():
    request = Request()
    session_id = uuid.uuid4().hex[:8]
    request.session_id = session_id
    request.context = {}
    request.user_input = "我要写青春歌词"
    subagents = [
        FictionWritingExpertAgent(agent_id=f"agent_{uuid.uuid4().hex[:6]}", config={}),
        CodeWritingExpertAgent(agent_id=f"agent_{uuid.uuid4().hex[:6]}", config={}),
        SongWritingExpertAgent(agent_id=f"agent_{uuid.uuid4().hex[:6]}", config={})
    ]
    tools = []
    replan_panel = SimpleReplanPanel(id=uuid.uuid4().hex[:4],config={},request=request,subagents=subagents,tools=tools)
    replan_panel.run(request=request)

if __name__ == "__main__":
    main()