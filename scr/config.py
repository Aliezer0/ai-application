from dotenv import load_dotenv
import os

# 加载.env文件（默认从项目根目录查找）
load_dotenv()

# 读取环境变量
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")