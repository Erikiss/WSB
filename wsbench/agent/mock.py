"""A scripted stand-in for the Managed Agents API, used when WSB_LLM=mock (offline demo and tests)."""
from types import SimpleNamespace as NS

SCRIPT = "model neox-tiny\nP1: gauge permute(L[1].mlp, randperm)\nP2: gauge rotate(L[1].mlp, orth)\n"


class MockPlatform:
    def __init__(self):
        self.turn, self.count = -1, 0
        self.beta = NS(
            environments=NS(create=lambda **kw: NS(id="env_mock")),
            agents=NS(create=lambda **kw: NS(id="agent_mock", version=1),
                      update=lambda agent_id, **kw: NS(id=agent_id, version=2)),
            sessions=NS(create=lambda **kw: NS(id="sesn_mock"),
                        retrieve=lambda session_id: NS(usage=NS(list_cost=NS(amount=str(12 * (self.turn + 1)), currency="USD"))),
                        events=NS(stream=self._stream, send=self._send, list=lambda session_id, **kw: [])),
        )

    def _event(self, kind, **fields):
        self.count += 1
        return NS(type=kind, id=f"sevt_mock_{self.count}", **fields)

    def _send(self, session_id, events):
        if events[0]["type"] == "user.message":
            self.turn += 1

    def _stream(self, session_id):
        text = lambda t: [NS(type="text", text=t)]
        end = self._event("session.status_idle", stop_reason=NS(type="end_turn"))
        if self.turn + 1 == 0:
            events = [
                self._event("agent.message", content=text("I will compare a neuron permutation with a neuron rotation.")),
                self._event("agent.custom_tool_use", name="wsl_check", input={"script": SCRIPT}),
                self._event("agent.custom_tool_use", name="wsl_run", input={"script": SCRIPT}),
                self._event("agent.message", content=text("The permutation is an exact symmetry; the rotation is not.")),
                end,
            ]
        else:
            events = [self._event("agent.message", content=text("Understood.")), end]

        class Stream:
            def __enter__(inner): return iter(events)
            def __exit__(inner, *a): return False
        return Stream()
