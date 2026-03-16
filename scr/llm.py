import os
from langchain_openai import ChatOpenAI
from config import DEEPSEEK_API_KEY  # 从config导入

def get_llm(model_name="deepseek-chat", temperature=0.7, max_tokens=500):
    """
    返回配置好的ChatOpenAI实例（兼容DeepSeek API）
    """
    return ChatOpenAI(
        model=model_name,
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base="https://api.deepseek.com/v1",
        temperature=temperature,
        max_tokens=max_tokens
    )


