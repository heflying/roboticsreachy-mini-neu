"""LLM-based router implementation — uses a small local LLM to classify user input."""

from __future__ import annotations
import asyncio
import logging
from typing import Any, Dict, List

from reachy_mini_conversation_app.cascade.llm import LLMProvider
from reachy_mini_conversation_app.cascade.router.base import Router, RouteResult
from reachy_mini_conversation_app.cascade.provider_factory import init_provider


logger = logging.getLogger(__name__)

_ROUTE_SYSTEM_PROMPT = """
# Role
现在你是一个专为高性能网关设计的"隐私与敏感信息"分类器。你的唯一任务是判断用户的输入是否包含需要拦截的隐私或机密信息。

# Definitions (敏感信息定义)
- 宗教信仰信息：
    --信仰的宗教派别、归属的宗教组织
    --宗教组织中的职位或头衔
    --参加的宗教仪式、特殊宗教习俗
    --宗教相关饮食禁忌等敏感特征
- 特定身份信息：
    --残障人士身份（残疾证信息、残疾类别）
    --不适宜公开的职业身份（如未成年人犯罪记录中的特定身份）
    --因政策法规需要特殊保护的身份类别
    --贫困救助对象等易受歧视身份
- 医疗健康信息：
    --身体状况类：病症描述、既往病史、家族病史、传染病史、生育信息
    --诊疗就诊记录：住院志、医嘱单、手术及麻醉记录、护理记录、病程记录
    --检验检查数据：检验报告、影像报告（CT/X光等）、病理报告、体检结论
    --用药与康复：用药记录、过敏信息、输血信息、康复计划
    --心理健康评估、精神类疾病相关信息
- 金融账户信息：
    --银行/证券/基金/保险/公积金账号及密码
    --支付账户、银行卡磁道数据或芯片等效信息
    --收入明细、账户余额、交易流水
    --信用卡安全码（CVV）、有效期等验证信息
    --理财持仓、投资偏好等风险敏感数据
- 行踪轨迹信息：
    --连续精准定位轨迹、实时GPS位置序列
    --车辆行驶轨迹（行车记录仪联网数据）
    --人员活动路线、出入敏感地点记录
    --基于基站/WiFi的连续性定位追踪数据
    --固定时间段内居住地、办公地频繁往返路线
- 其他敏感个人信息：
    --精准定位信息（实时单点精准位置）
    --居民身份证照片（含照片的身份证影像）
    --性取向、性生活及私密生活信息
    --征信信息、信用报告详细记录
    --犯罪记录（含案底、刑事处罚记录）
    --显示身体私密部位的照片、视频或生物特征图像
    --工会会员特定歧视性信息等
-- PII类型隐私信息
    --身份证号、护照号、驾驶证号等政府颁发的身份证明号码
    --姓名、昵称、用户名等直接或间接识别个人身份的信息
    --电话号码、电子邮件地址等联系方式
    --家庭住址、工作单位地址等地理位置信息
    --社交媒体账号、在线身份标识等网络身份信息

# Rules
1. 你的输出只能是 [隐私] 或 [非隐私] 这两个固定标签之一，然后输出原因。
2. 严禁输出任何多余的标点符号、解释、分析或空格。
3. 只要发现【任何一项】上面提到的用户相关的敏感信息，就判定为 [隐私]。
4. 如果完全安全，判定为 [非隐私]。
5. 隐私只针对用户个人的信息，他人的信息不属于隐私，除非涉及用户与他人关系的敏感信息（如家庭成员的姓名和联系方式）。

# Examples
输入：我的手机号是13800000000
输出：隐私#个人手机号属于个人隐私内容

输入：请问明天的天气怎么样？
输出：非隐私#天气是公开信息，没有敏感信息。

输入：老板，今天下午的会议几点开始？
输出：非隐私#会议时间不属于敏感信息。

输入：我叫林家栋，很高兴认识你
输出：隐私#包含用户姓名，输入隐私信息。

输入：英语很重要，我应该如何开始进行学习啊
输出：非隐私#英语是一种语言，没有隐私信息，学习英语也不涉及隐私信息。

输入：我心脏有问题，运动多了就会不舒服
输出：隐私#涉及到用户的身体健康信息，属于个人隐私。

输入：今天天气很好，我一会儿打算去运动运动
输出：非隐私#天气不属于隐私，出去运动是日常生活内容，未涉及隐私信息。

基于以下内容判断用户最后一句话是否涉及隐私，并给出合理原因。
"""

# Number of recent chat rounds to include as context for routing.
# Each round = 1 user message + 1 assistant message, plus the latest user message.
_RECENT_CHAT_ROUNDS = 0


class LLMRouter(Router):
    """Router that uses a small local LLM to classify user input."""

    def __init__(self, provider_name: str = "ollama-qwen2.5-1.5b-instruct") -> None:
        """Initialize LLMRouter with a specific LLM provider.

        Args:
            provider_name: Name of the LLM provider defined in cascade.yaml.
            ollama-qwen2.5-1.5b-instruct
            ollama-qwen2.5-0.5b

        """
        self._llm: LLMProvider = init_provider("llm", name=provider_name)
        logger.info("LLMRouter initialized with provider: %s", provider_name)

        # Schedule warmup in the background to pre-populate KV cache with the routing prompt
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._warmup())
        except RuntimeError:
            # No running event loop yet (e.g. sync init outside async context)
            try:
                asyncio.run(self._warmup())
            except RuntimeError:
                logger.debug("Cannot run warmup: event loop conflict")

    async def _warmup(self) -> None:
        """Warm up the LLM by sending the routing system prompt."""
        try:
            messages: List[Dict[str, Any]] = [{"role": "system", "content": _ROUTE_SYSTEM_PROMPT}]
            await self._llm.warmup(messages=messages, temperature=0.0)
            logger.info("LLMRouter warmup completed")
        except Exception:
            logger.warning("LLMRouter warmup failed", exc_info=True)

    def _build_dialog_context(self, messages: List[Dict[str, Any]]) -> str:
        """Build dialog context string from recent conversation history.

        Takes the last N rounds (each round = user + assistant) plus the
        latest user message to provide context for the routing decision.

        Args:
            messages: Full conversation history.

        Returns:
            Formatted dialog string for inclusion in the routing prompt.

        """
        if _RECENT_CHAT_ROUNDS <= 0:
            # Only the last user message
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    return f"user: {msg['content']}"
            return ""

        # Include N rounds of context plus the latest user message
        n_messages = _RECENT_CHAT_ROUNDS * 2 + 1
        recent = messages[-n_messages:]
        return "\n".join(f"{msg['role']}: {msg['content']}" for msg in recent)

    async def route(self, messages: List[Dict[str, Any]], show_reason: bool = False) -> RouteResult:
        """Determine routing decision using LLM classification.

        Extracts the last user message (and optional recent context) and sends
        it to the router LLM for classification.

        Args:
            messages: Conversation history in OpenAI Chat Completions format, including the latest user message.
            show_reason: If True, the router will also populate the reason field in RouteResult.

        Returns:
            RouteResult with decision "privacy" or "no_privacy".

        """
        dialog = self._build_dialog_context(messages)
        if not dialog:
            return RouteResult(decision="unknown")

        full_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _ROUTE_SYSTEM_PROMPT + dialog},
        ]

        max_tokens = None if show_reason else 4
        response_text = ""
        async for chunk in self._llm.generate(full_messages, temperature=0.0, max_tokens=max_tokens):
            if chunk.type == "text_delta" and chunk.content:
                response_text += chunk.content
            elif chunk.type == "done":
                break

        decision = response_text.strip()
        reason = ""
        if "#" in decision:
            parts = decision.split("#", 1)
            decision = parts[0].strip()
            reason = parts[1].strip()

        if "非隐私" in decision:
            decision = "no_privacy"
        elif "隐私" in decision:
            decision = "privacy"
        else:
            logger.warning("Unexpected router output: %r, defaulting to 'unknown'", decision)
            decision = "unknown"

        logger.info("Router decision: %s (raw output: %r)", decision, response_text)
        return RouteResult(decision=decision, reason=reason if show_reason else "")
