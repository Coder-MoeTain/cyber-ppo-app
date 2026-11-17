import numpy as np
from autocd.envs.cyborg_adapter import CybORGAdapter
class FakeBase:
    def __init__(self,scenario_generator):self.scenario_generator=scenario_generator
class Space:n=3
class FakeWrapper:
    def __init__(self,env,pad_spaces=True):self.names=[f"blue_agent_{i}" for i in range(5)]
    def reset(self):return ({a:np.zeros(4,dtype=np.float32) for a in self.names},{a:{} for a in self.names})
    def action_space(self,a):return Space()
    def action_mask(self,a):return [True,False,True]
    def action_labels(self,a):return ["Sleep","Invalid","Analyse"]
    def step(self,actions):
        assert all(v in (0,2) for v in actions.values())
        return ({a:np.ones(4,dtype=np.float32) for a in self.names},{a:1. for a in self.names},{a:False for a in self.names},{a:False for a in self.names},{a:{} for a in self.names})
    def close(self):pass
def test_adapter_contract_fake_backend():
    e=CybORGAdapter(scenario_generator=object(),cyborg_factory=FakeBase,wrapper_factory=FakeWrapper)
    obs,info=e.reset();assert len(obs)==5 and e.observation_dims["blue_agent_0"]==4
    assert e.action_dims["blue_agent_0"]==3 and info["blue_agent_0"]["action_mask"].tolist()==[True,False,True]
    out=e.step({a:0 for a in e.agents});assert out[1]["blue_agent_0"]==1.;e.close()
