import yaml
from app.utils.path_tool_util import get_abs_path

def load_rag_yaml(config_path: str = get_abs_path("config/rag.yml"), encoding="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_milvus_yaml(config_path: str = get_abs_path("config/milvus.yml"), encoding="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_app_yaml(config_path: str = get_abs_path("config/app.yml"), encoding="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_yaml(config_path: str = get_abs_path("config/agent.yml"), encoding="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


rag_conf = load_rag_yaml()
milvus_conf = load_milvus_yaml()
app_conf = load_app_yaml()