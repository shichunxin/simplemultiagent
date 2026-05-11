import uuid

from multiagent.config.appconfig import QIANWEN_MODEL
from multiagent.core.agent import Request, CommonAgent
from multiagent.core.prompt import build_analysts_prompt, build_researcher_prompt, build_report_writing_expert_prompt
from multiagent.expert.agent import TeamAgent


def main():
    request = Request()
    session_id = uuid.uuid4().hex[:8]
    request.session_id = session_id
    request.user_input = "对比claude与Trae两个工具的差异"
    agents = {}
    analystsAgent = CommonAgent(name="analysts_agent", role="analysts", model=QIANWEN_MODEL,
                                system_promtp=build_analysts_prompt())
    agents["analysts"] = analystsAgent
    researcherAgent = CommonAgent(name="researcher_agent", role="researcher", model=QIANWEN_MODEL,
                                  system_promtp=build_researcher_prompt())
    agents["researcher"] = researcherAgent
    reportWritingExpertAgent = CommonAgent(name="reportWritingExpert_agent", role="reportWritingExpert",
                                           model=QIANWEN_MODEL, system_promtp=build_report_writing_expert_prompt())
    agents["reportWritingExpert"] = reportWritingExpertAgent
    teamAgent = TeamAgent(request=request,agents=agents)
    result = teamAgent.run(request=request)
    print("-------------最终结果----------------")
    print(result)

if __name__ == "__main__":
    main()