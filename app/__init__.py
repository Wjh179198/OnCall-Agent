"""SuperBizAgent Python 版本

基于 LangChain 的智能业务代理系统
"""
from app.utils.config_handler_util import app_conf

__version__ = app_conf["app_version"]

from app.utils import logger  # noqa: F401