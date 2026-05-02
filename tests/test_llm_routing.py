from app.services.llm import should_escalate


def test_should_escalate_for_complex_query():
    q = "Compare cross-department tradeoffs and run a what if scenario for deficit sensitivity."
    assert should_escalate(q)


def test_should_not_escalate_for_simple_query():
    q = "What is the budget timeline?"
    assert not should_escalate(q)
