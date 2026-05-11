

def create_mcp_message(sender:str,content:str,metadata:dict=None):
    return {
        "protocol_version": "1.0",
        "sender": sender,
        "content": content,
        "metadata": metadata or {}
    }

def validate_mcp_message(message) -> bool:
    keys = ["protocol_version", "sender", "content", "metadata"]
    if not isinstance(message,dict):
        return False
    for key in keys:
        if key not in message:
            return False
    return True
