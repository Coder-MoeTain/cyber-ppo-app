from autocd.coordination import RiskAwareCoordinator

def test_high_risk_gets_response_task():
    tasks = RiskAwareCoordinator().assign({"host-a": .9, "host-b": .2})
    assert any(t.agent == "response" and t.target == "host-a" for t in tasks)

def test_empty_agents():
    assert RiskAwareCoordinator().assign({"host-a": .9}, ()) == []
