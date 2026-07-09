"""
Ollama 原生 API 模型类 - 支持禁用 thinking 模式
使用 Ollama 原生 /api/chat 接口，而非 OpenAI 兼容接口
"""
import json
import time
import requests
from typing import Any, Dict, List, Optional

from evalscope.api.messages import ChatMessage, ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
from evalscope.api.model import ModelAPI, ModelOutput
from evalscope.api.model.generate_config import GenerateConfig
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.utils import get_logger

logger = get_logger()


class OllamaNativeAPI(ModelAPI):
    """
    使用 Ollama 原生 API（/api/chat）的模型接口
    支持 think=false 参数来禁用 thinking 模式
    """

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
        )
        # 移除末尾的斜杠和 /v1 后缀，并构建 API 端点
        self.base_url = (base_url or 'http://localhost:11434').rstrip('/')
        # 如果 base_url 包含 /v1，移除它（因为我们要使用原生 API）
        self.base_url = self.base_url.removesuffix('/v1')
        self.chat_url = f'{self.base_url}/api/chat'
        self.model_name = model_name
        self.config = config
        self.kwargs = kwargs

    def _convert_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """将 ChatMessage 列表转换为 Ollama API 格式"""
        ollama_messages = []
        for msg in messages:
            if isinstance(msg, ChatMessageSystem):
                ollama_messages.append({
                    'role': 'system',
                    'content': msg.content
                })
            elif isinstance(msg, ChatMessageUser):
                ollama_messages.append({
                    'role': 'user',
                    'content': msg.content
                })
            elif isinstance(msg, ChatMessageAssistant):
                ollama_messages.append({
                    'role': 'assistant',
                    'content': msg.content
                })
            else:
                # 其他类型（如工具消息）暂不支持
                logger.warning(f'Unsupported message type: {type(msg).__name__}')
        return ollama_messages

    def _build_request(self, messages: List[Dict[str, Any]], config: GenerateConfig,
                       tools: List[ToolInfo], tool_choice: ToolChoice) -> Dict[str, Any]:
        """构建 Ollama API 请求体"""
        request = {
            'model': self.model_name,
            'messages': messages,
            'think': False,  # 关键：禁用 thinking 模式
            'stream': False,
        }

        # 添加生成参数
        if config.max_tokens is not None:
            request['options'] = request.get('options', {})
            request['options']['num_predict'] = config.max_tokens

        if config.temperature is not None:
            request['options'] = request.get('options', {})
            request['options']['temperature'] = config.temperature

        if config.top_p is not None:
            request['options'] = request.get('options', {})
            request['options']['top_p'] = config.top_p

        if config.top_k is not None:
            request['options'] = request.get('options', {})
            request['options']['top_k'] = config.top_k

        # 注意：Ollama 原生 API 的工具调用支持可能有限
        if tools and len(tools) > 0:
            logger.warning('Tool calling may not be fully supported in Ollama native API')

        return request

    def _parse_response(self, response_data: Dict[str, Any]) -> ModelOutput:
        """解析 Ollama API 响应为 ModelOutput"""
        message_data = response_data.get('message', {})
        content = message_data.get('content', '')

        # 构建 ChatMessageAssistant
        chat_message = ChatMessageAssistant(content=content)

        # 构建 ChatCompletionChoice
        from evalscope.api.model.model_output import ChatCompletionChoice
        choice = ChatCompletionChoice(
            message=chat_message,
            stop_reason='stop',
        )

        # 构建 ModelOutput（必须通过 choices 字段）
        output = ModelOutput(
            model=self.model_name,
            choices=[choice],
        )

        # 添加 usage 信息（如果可用）
        if 'eval_count' in response_data or 'prompt_eval_count' in response_data:
            from evalscope.api.model.model_output import ModelUsage
            usage = ModelUsage(
                input_tokens=response_data.get('prompt_eval_count', 0),
                output_tokens=response_data.get('eval_count', 0),
            )
            output.usage = usage

        return output

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """生成模型输出（同步）"""
        # 转换消息格式
        ollama_messages = self._convert_messages(input)

        # 构建请求
        request_data = self._build_request(ollama_messages, config, tools, tool_choice)

        # 发送请求
        try:
            t_start = time.monotonic()
            response = requests.post(self.chat_url, json=request_data, timeout=120)
            response.raise_for_status()
            total_time = time.monotonic() - t_start

            response_data = response.json()

            # 解析响应
            output = self._parse_response(response_data)
            output.time = total_time

            return output

        except requests.exceptions.RequestException as e:
            logger.error(f'Ollama API request failed: {e}')
            raise

    async def generate_async(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """生成模型输出（异步）"""
        # 简单实现：在线程池中运行同步方法
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.generate,
            input,
            tools,
            tool_choice,
            config,
        )


# 注册到 EvalScope
from evalscope.api.registry import register_model_api
register_model_api('ollama_native')(OllamaNativeAPI)

print("[OK] OllamaNativeAPI 已注册（eval_type='ollama_native'）")
