from __future__ import annotations

"""
定义基础 LLM 配置类型。
"""

import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .errors import ConfigError, ErrorCode

# 是否已尝试加载过 .env，避免重复 IO。
_DOTENV_LOADED = False


def load_dotenv(start: str | Path | None = None) -> None:
    """
    从 `start`（默认当前工作目录）向上查找最近的 `.env` 并载入到 os.environ。

    零依赖实现：解析 `KEY=VALUE` 行，忽略空行与 `#` 注释，
    已存在的环境变量不被覆盖（环境变量优先级高于 .env）。
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return

    directory = Path(start or os.getcwd()).resolve()
    for folder in (directory, *directory.parents):
        candidate = folder / ".env"
        if candidate.exists():
            for raw in candidate.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
            _DOTENV_LOADED = True
            return
    # 没找到也标记已扫描，避免每次请求重复遍历文件系统。
    _DOTENV_LOADED = True


class LLMConfig(BaseModel):
    """
    描述一个语言模型客户端所需的基础连接配置。
    """

    # extra="forbid"：从 YAML 等外部来源加载时，未知字段直接报错，能当场抓出拼写错误。
    # （不开 strict：保持 Python 宽松，允许常规类型强转。）
    model_config = ConfigDict(extra="forbid")

    # API 基础地址，用于拼接模型请求。
    api_base_url: str
    # 模型名称，用于指定实际调用的模型。
    model_name: str
    # 环境变量名称，对应 .env 或运行环境中的真实密钥。
    api_key_env_name: str
    # 请求超时时间，单位为秒。
    timeout: float = Field(default=60.0, gt=0)

    @field_validator("api_base_url")
    @classmethod
    def validate_api_base_url(cls, value: str) -> str:
        """
        校验 API 地址格式。
        """
        url = value.strip()
        if not url:
            raise ValueError("api_base_url 不能为空。")
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError("api_base_url 必须以 http:// 或 https:// 开头。")
        return url.rstrip("/")

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        """
        校验模型名称。
        """
        model_name = value.strip()
        if not model_name:
            raise ValueError("model_name 不能为空。")
        return model_name

    @field_validator("api_key_env_name")
    @classmethod
    def validate_api_key_env_name(cls, value: str) -> str:
        """
        校验环境变量名称。
        """
        env_name = value.strip()
        if not env_name:
            raise ValueError("api_key_env_name 不能为空。")
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", env_name):
            raise ValueError("api_key_env_name 必须是合法的环境变量名称。")
        return env_name

    @classmethod
    def deepseek(
        cls,
        model_name: str = "deepseek-v4-pro",
        *,
        api_base_url: str = "https://api.deepseek.com",
        api_key_env_name: str = "DEEPSEEK_API_KEY",
        timeout: float = 60.0,
    ) -> "LLMConfig":
        """
        构造一个对接 DeepSeek（OpenAI 兼容）的连接配置。

        常用模型：deepseek-v4-pro、deepseek-v4-flash（两者均支持 reasoning）。
        """
        return cls(
            api_base_url=api_base_url,
            model_name=model_name,
            api_key_env_name=api_key_env_name,
            timeout=timeout,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "LLMConfig":
        """
        从 YAML 文件加载配置。
        """
        path = Path(yaml_path).expanduser().resolve()
        if not path.exists():
            raise ConfigError(code=ErrorCode.CONFIG_FILE_NOT_FOUND, path=str(path))

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ConfigError("YAML 顶层必须是对象映射。", code=ErrorCode.INVALID_CONFIG)

        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ConfigError(code=ErrorCode.INVALID_CONFIG, cause=exc, path=str(path)) from exc

    def resolve_api_key(self, env: Mapping[str, str] | None = None) -> str:
        """
        从环境变量中解析真实 API 密钥。

        若未显式传入 env，会先尝试加载项目 `.env` 文件再读取。
        """
        if env is None:
            load_dotenv()
        source = env if env is not None else os.environ
        api_key = source.get(self.api_key_env_name)

        if api_key is None or not api_key.strip():
            raise ConfigError(code=ErrorCode.MISSING_API_KEY, env=self.api_key_env_name)

        return api_key.strip()

    def to_request_options(self, env: Mapping[str, str] | None = None) -> dict[str, Any]:
        """
        解析为实际请求所需的最小参数集合。
        """
        return {
            "api_base_url": self.api_base_url,
            "model_name": self.model_name,
            "api_key": self.resolve_api_key(env),
            "timeout": self.timeout,
        }
