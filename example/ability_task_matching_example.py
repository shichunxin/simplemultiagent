import uuid

from multiagent.core.agent import Request, Response, AbilityTaskMatchingAgent, SubTask, BaseAgent
from multiagent.core.panel import SimpleTaskDecompositionPanel
from multiagent.expert.agent import FictionWritingExpertAgent, CodeWritingExpertAgent, SongWritingExpertAgent


def main():
    req_info = "我要写青春歌词"
    request = Request()
    session_id = uuid.uuid4().hex[:8]
    request.session_id = session_id
    request.user_input = req_info
    subagents = [
        FictionWritingExpertAgent(agent_id=f"agent_{uuid.uuid4().hex[:6]}",config={}),
        CodeWritingExpertAgent(agent_id=f"agent_{uuid.uuid4().hex[:6]}",config={}),
        SongWritingExpertAgent(agent_id=f"agent_{uuid.uuid4().hex[:6]}",config={})
    ]
    id = uuid.uuid4().hex[:4]
    decomposition_panel = SimpleTaskDecompositionPanel(id,{},request,subagents)
    decomposition_panel.run(request)

if __name__ == "__main__":
    main()