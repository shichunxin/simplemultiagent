import json
from dataclasses import dataclass, field, asdict
from typing import List

from multiagent.core.agent import run_once, run_once_messages


@dataclass
class ConversationMessage:
    role:str
    content:str
    token_count:int

@dataclass
class SummaryBufferMemory:
    """记忆摘要"""
    max_buffer_tokens:int = 2000
    summary:str = ""
    buffer:List[ConversationMessage] = field(default_factory=list)
    current_tokens:int = 0

    def add_message(self,message:ConversationMessage)->None:
        self.buffer.append(message)
        self.current_tokens += message.token_count
        if self.current_tokens > self.max_buffer_tokens:
            self._compress_buffer()

    def _compress_buffer(self):
        keep_count = min(6, len(self.buffer))
        to_summary = self.buffer[:-keep_count]
        if to_summary:
            new_summary = self._gen_summary(to_summary)
            self.add_message(new_summary)
        self.buffer = self.buffer[-keep_count:]
        self.current_tokens = sum(s.token_count for s in self.buffer)

    def _gen_summary(self,message:List[ConversationMessage]) -> ConversationMessage :
        messagestr = self._format_message(message)
        prompt = f"""
        将以下对话压缩为简洁摘要，保留关键信息:
        {{messagestr}}
        要求: 1 保留用户意图 2 保留重要决策 3 不超过100字
        """
        ms = [
            {
                "role": "systme",
                "content": prompt
            }
        ]
        response = run_once_messages(ms)
        message = response.choices[0].message
        raw = message.content
        message = ConversationMessage()
        message.token_count = len(message)
        message.summary = raw
        message.role = "assistant"
        message.content = raw
        return message


    def _format_message(self,messages:List[ConversationMessage]) -> str:
        if not messages:
            return ""
        list = json.dumps([asdict(m) for m in messages])
        return list


