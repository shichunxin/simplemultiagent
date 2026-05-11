import os

from openai import OpenAI


class MultiModelClient:
    """
    多客户端
    """
    PROVIDERS = {
        "deepseek":{"base_url":"","env":"DEEPSEEK_API_KEY"},
        "qianwen":{"base_url":"","env":"QIANWEN_API_KEY"},
        "gpt":{"base_url":"","env":"GPT_API_KEY"},
    }
    def __init__(self):
        self.clients = {}
        self._init_clients()

    def _init_clients(self):
        for providers,config in self.PROVIDERS.items():
            api_key = os.getenv(config["env"])
            if api_key:
                self.clients[providers] = OpenAI(base_url=config["base_url"],api_key=api_key)

    def chat(self,model:str,messages:list)->str:
        client = self.clients[model]
        if not client:
            raise ValueError(f"模型{model}不存在")
        response = client.chat.completions.create(model=model,messages=messages,max_tokens=4096)
        return response.choices[0].message.content



