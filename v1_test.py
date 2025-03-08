import os
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import gpt_4o_mini_complete
from dotenv import load_dotenv

load_dotenv()

WORKING_DIR = "data"

api_key = os.environ["OPENAI_API_KEY"]
base_url = os.environ["OPENAI_BASE_URL"]


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = PathRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,
)

data_file = "data/output.txt"
question = "Limitation of eixsting Reinfocement Learning approaches"
with open(data_file) as f:
    rag.insert(f.read())

print(rag.query(question, param=QueryParam(mode="hybrid")))
