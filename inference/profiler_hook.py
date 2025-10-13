from typing import Callable
import sys

from react_agent import MultiTurnReactAgent
from simple_profiler import SimpleProfiler

class AgentHookForProfiler:
    """
    Manage a SimpleProfiler instance across multiple runs.

    Usage:
        p = AgentHookForProfiler(agent: MultiTurnReactAgent)
        p.hook()
        # run agent multiple times (profiler stays active and records)
        p.unhook()
    """
    agent: MultiTurnReactAgent
    _hooked_agents: dict[str, Callable] = {}
    
    def __init__(self, agent: MultiTurnReactAgent, profile_dir: str = "./profiler_traces"):
        self.agent = agent
        self.profiler = SimpleProfiler(profile_dir=profile_dir)

    def hook(self):
        """Install wrappers on the agent that lazy-start the profiler in the call thread."""
        orig_call_server = getattr(self.agent, 'call_server')
        orig_custom_call_tool = getattr(self.agent, 'custom_call_tool')

        def wrapped_call_server(msgs, planning_port, *a, **kw):
            assert orig_call_server is not None, "Agent has no call_server to wrap"
            with self.profiler.record_function(f"llm:{getattr(self.agent, 'model', 'unknown')}"):
                return orig_call_server(msgs, planning_port, *a, **kw)

        def wrapped_custom_call_tool(tool_name, tool_args, *a, **kw):
            assert orig_custom_call_tool is not None, "Agent has no custom_call_tool to wrap"
            with self.profiler.record_function(f"tool:{tool_name}"):
                return orig_custom_call_tool(tool_name, tool_args, *a, **kw)

        setattr(self.agent, 'call_server', wrapped_call_server)
        setattr(self.agent, 'custom_call_tool', wrapped_custom_call_tool)
        self._hooked_agents = {}
        self._hooked_agents['call_server'] = orig_call_server
        self._hooked_agents['custom_call_tool'] = orig_custom_call_tool
        
        print(f"[AgentProfiler] Installing hooks on agent {self.agent}", file=sys.stderr)
    
    def unhook(self):
        """Remove any installed wrappers and restore original methods."""
        for method_name, orig_method in list(self._hooked_agents.items()):
            setattr(self.agent, method_name, orig_method)
        self._hooked_agents = {}
        self.export()
    
    def export(self):
        self.profiler.export_trace()