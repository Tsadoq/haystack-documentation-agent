from haystack.agents.memory import ConversationSummaryMemory
from haystack.nodes import PromptNode


def return_memory_node(openai_key: str) -> ConversationSummaryMemory:
    """
    Returns the memory node.
    :param openai_key: the OpenAI key
    :return: the memory node
    """
    memory_prompt_node = PromptNode('gpt-3.5-turbo-16k', api_key=openai_key, max_length=1024)
    return ConversationSummaryMemory(memory_prompt_node)
