import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import BaseTool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from llm import get_llm

# 从你的模块导入
from llm import get_llm                      
from qa_chain import create_qa_chain          

# --------------------------------------------------------------
# 工具1：计算器（自定义工具）
# --------------------------------------------------------------
class CalculatorTool(BaseTool):
    """执行数学计算，输入应为表达式如 '2 + 2'"""
    name: str = "Calculator"
    description: str = "用于执行数学计算，输入应为数学表达式，如 '2 + 2'"

    def _run(self, query: str) -> str:
        try:
            # 注意：eval 有安全风险，仅用于演示
            return str(eval(query))
        except Exception as e:
            return f"计算错误：{e}"

    async def _arun(self, query: str):
        raise NotImplementedError("异步未实现")

# --------------------------------------------------------------
# 工具2：网络搜索（使用 DuckDuckGo）
# --------------------------------------------------------------
def create_search_tool():
    search = DuckDuckGoSearchRun()
    return Tool(
        name="WebSearch",
        func=search.run,
        description="当需要实时信息（如天气、新闻、最新数据）时使用。输入搜索关键词。"
    )

# --------------------------------------------------------------
# 工具3：RAG 知识库查询（将你的 QA 链包装为工具）
# --------------------------------------------------------------
# 初始化 RAG 链（全局单例，避免重复加载）
_qa_chain = None

def get_rag_chain():
    global _qa_chain
    if _qa_chain is None:
        _qa_chain = create_qa_chain()  # 返回一个 RetrievalQA 链
    return _qa_chain

def rag_tool_func(query: str) -> str:
    """从内部知识库检索信息并回答问题"""
    chain = get_rag_chain()
    # RetrievalQA 的 invoke 返回字典，包含 'result' 和 'source_documents'
    result = chain.invoke({"query": query})
    return result['result']  # 只返回答案文本

def create_rag_tool():
    return Tool(
        name="KnowledgeBase",
        func=rag_tool_func,
        description="当你需要查询公司内部知识库、产品文档或政策时使用。输入应为具体问题。"
    )

# --------------------------------------------------------------
# 创建 Agent 执行器
# --------------------------------------------------------------
def create_agent(verbose=True):
    """
    创建 ReAct Agent，集成 RAG、搜索和计算器工具。
    使用 langchain-classic 的 initialize_agent，稳定可靠。
    """
    # 1. 初始化 LLM
    llm = get_llm(temperature=0)

    # 2. 初始化所有工具
    tools = [
        create_rag_tool(),
        create_search_tool(),
        CalculatorTool()
    ]

    # 3. 构建正确的提示模板
    #    对于工具调用 Agent，模板中不能包含 {tools} 或 {tool_names}
    #    只需系统消息、人类消息（含 {input}）和 agent_scratchpad 占位符
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个智能助手，可以使用工具来回答问题。请根据用户的问题，决定是否需要使用工具，如果需要就调用相应工具。"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    # 4. 创建 Agent（返回一个 Runnable 对象）
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 5. 创建执行器（包装 agent，管理迭代、工具调用、错误等）
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=5,          # 防止无限循环
        early_stopping_method="generate"
    )
    return agent_executor

# --------------------------------------------------------------
# 测试入口
# --------------------------------------------------------------
if __name__ == "__main__":
    agent = create_agent()
    
    # 测试问题1：纯知识库查询
    response1 = agent.invoke({"input": "PZ-DSP28335-L 开发板常用的一些模块"})
    print("Q1:", response1['output'])
    