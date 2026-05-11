import asyncio
import uuid

from multiagent.core.mcp import create_mcp_message, validate_mcp_message
from multiagent.expert.agent import ResearcherAgent, WriterAgent, ValidatorAgent


def main () -> None:
    research_topic = "Mediterranean Diet"
    mcp_to_researcher = create_mcp_message(sender="Orchestrator", content=research_topic)
    researcher = ResearcherAgent(agent_id=f"agent_P{uuid.uuid4().hex[:6]}")
    mcp_from_researcher = asyncio.run(researcher.process(mcp_to_researcher))
    if not validate_mcp_message(mcp_from_researcher) or not mcp_from_researcher['content']:
        return
    research_summary = mcp_from_researcher['content']

    final_output = "无法生成经过验证的文章。"
    max_revisions = 2
    for i in range(max_revisions):
        print(f"\n[Orchestrator] 写作尝试 {i + 1}/{max_revisions}")

        writer_context = research_summary
        if i > 0:
            writer_context += f"\n\n请根据这些反馈修改之前的草稿: {validation_result}"

        mcp_to_writer = create_mcp_message(sender="Orchestrator", content=writer_context)
        writer = WriterAgent(agent_id=f"agent_P{uuid.uuid4().hex[:6]}")
        mcp_from_writer = asyncio.run(writer.process(mcp_to_writer))

        if not validate_mcp_message(mcp_from_writer) or not mcp_from_writer['content']:
            break
        draft_post = mcp_from_writer['content']

        validation_content = {"summary": research_summary, "draft": draft_post}
        mcp_to_validator = create_mcp_message(sender="Orchestrator", content=validation_content)
        validator = ValidatorAgent(agent_id=f"agent_P{uuid.uuid4().hex[:6]}")
        mcp_from_validator = asyncio.run(validator.process(mcp_to_validator))

        if not validate_mcp_message(mcp_from_validator) or not mcp_from_validator['content']:
            break
        validation_result = mcp_from_validator['content']

        if "通过" in validation_result.lower():
            final_output = draft_post
            break
        else:
            if i < max_revisions - 1:
                print("请求修改。")
            else:
                print("已达到最大修订次数。工作流程失败。")

    print("\n" + "=" * 50)
    print("[Orchestrator] 工作流程完成。最终输出:")
    print("=" * 50)
    print(final_output)


if __name__ == "__main__":
    main()