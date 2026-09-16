import pytest

from app.agent import LearnMateAgent, _safe_calculate


def test_safe_calculate_basic_and_rejects_code():
    assert _safe_calculate("2 + 10")["result"] == 12
    assert _safe_calculate("3 * (4 + 2)")["result"] == 18
    with pytest.raises(ValueError):
        _safe_calculate("__import__('os').getcwd()")


@pytest.mark.asyncio
async def test_agent_executes_function_tool_and_keeps_current_message_primary():
    calls = []
    agent = LearnMateAgent(lambda name, args: pytest.fail(f"unexpected tool: {name}"))

    responses = [
        {
            "output": [{
                "type": "function_call",
                "name": "calculate",
                "arguments": '{"expression":"2+3"}',
                "call_id": "call-1",
            }]
        },
        {"output_text": "2 + 3 = 5."},
    ]

    async def fake_request(instructions, input_data):
        calls.append(input_data)
        return responses.pop(0)

    agent._request = fake_request
    reply, actions = await agent.run(
        "2+3",
        {
            "topic": "Binary Search",
            "conversation": [{"role": "assistant", "content": "Let's discuss binary search."}],
        },
    )

    assert reply == "2 + 3 = 5."
    assert actions == []
    assert "CURRENT LEARNER MESSAGE (highest priority):\n2+3" in str(calls[0])


@pytest.mark.asyncio
async def test_agent_calls_custom_tool_and_returns_follow_up():
    tool_calls = []

    async def executor(name, args):
        tool_calls.append((name, args))
        return {"learner_name": "Test", "weak_topics": ["Java"]}

    agent = LearnMateAgent(executor)
    responses = [
        {
            "output": [{
                "type": "function_call",
                "name": "get_learning_profile",
                "arguments": "{}",
                "call_id": "profile-1",
            }]
        },
        {"output_text": "Your current weak topic is Java."},
    ]

    async def fake_request(instructions, input_data):
        return responses.pop(0)

    agent._request = fake_request
    reply, _ = await agent.run("What is my weak topic?", {})

    assert tool_calls == [("get_learning_profile", {})]
    assert "Java" in reply
