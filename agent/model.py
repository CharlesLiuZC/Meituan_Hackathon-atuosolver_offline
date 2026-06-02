import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    model="deepseek-v4-pro",
    model_provider="deepseek",
    api_key=os.getenv("sk-bd51480f13f04a8b868ee652bcdfc148"),
    base_url="https://api.deepseek.com/v4",
    model_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},
)
