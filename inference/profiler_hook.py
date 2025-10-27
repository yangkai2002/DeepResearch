from typing import Callable
import asyncio
import sys
import random
import time
from react_agent import MultiTurnReactAgent, TOOL_MAP
from simple_profiler import SimpleProfiler
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError

def call_tool(self, profiler: SimpleProfiler, tool_name: str, tool_args: dict, **kwargs):
    if tool_name in TOOL_MAP:
        profile_context = profiler.record_function(f"tool: {tool_name}")
        profile_evt = profile_context.__enter__()
        tool_args["params"] = tool_args
        if "python" in tool_name.lower():
            result = TOOL_MAP['PythonInterpreter'].call(tool_args)
        elif tool_name == "parse_file":
            params = {"files": tool_args["files"]}
            
            raw_result = asyncio.run(TOOL_MAP[tool_name].call(params, file_root_path="./eval_data/file_corpus"))
            result = raw_result

            if not isinstance(raw_result, str):
                result = str(raw_result)
        else:
            raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
            result = raw_result
        profile_evt.set(tokens=self.count_tokens([{"role": "user", "content": result}]))
        profile_context.__exit__(None, None, None)
        return result
    else:
        return f"Error: Tool {tool_name} not found"
    

def call_server_stream(self, profiler: SimpleProfiler, msgs, planning_port, max_tries=10):
    openai_api_key = "EMPTY"
    openai_api_base = f"http://127.0.0.1:{planning_port}/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        timeout=600.0,
    )

    base_sleep_time = 1
    
    for attempt in range(max_tries):
        try:
            print(f"--- Attempting to call the service (stream), try {attempt + 1}/{max_tries} ---")
            
            chat_response = client.chat.completions.create(
                model=self.model,
                messages=msgs,
                stop=["\n<tool_response>", "<tool_response>"],
                temperature=self.llm_generate_cfg.get('temperature', 0.6),
                top_p=self.llm_generate_cfg.get('top_p', 0.95),
                max_tokens=10000,
                presence_penalty=self.llm_generate_cfg.get('presence_penalty', 1.1),
                stream=True  # Enable streaming
            )
            
            content_parts = []
            token_count = 0
            
            prefill_context = profiler.record_function("llm: prefill")
            prefill_evt = prefill_context.__enter__()
            
            decode_context = None
            decode_evt = None

            for chunk in chat_response:                    
                if (chunk.choices and 
                    len(chunk.choices) > 0 and 
                    chunk.choices[0].delta and 
                    hasattr(chunk.choices[0].delta, 'content') and
                    chunk.choices[0].delta.content is not None):
                    
                    if prefill_context and prefill_evt:
                        prefill_evt.set(tokens=1)
                        prefill_context.__exit__(None, None, None)

                        prefill_context = None
                        prefill_evt = None

                        decode_context = profiler.record_function("llm: decode")
                        decode_evt = decode_context.__enter__()
                    
                    delta_content = chunk.choices[0].delta.content
                    content_parts.append(delta_content)
                    token_count += 1
            
            if prefill_context and prefill_evt:
                prefill_evt.set(tokens=0)
                prefill_context.__exit__(None, None, None)

            if decode_context and decode_evt:
                decode_evt.set(tokens=token_count)
                decode_context.__exit__(None, None, None)
            
            content = ''.join(content_parts)
            return content.strip()

        except (APIError, APIConnectionError, APITimeoutError) as e:
            print(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
        except Exception as e:
            print(f"Error: Attempt {attempt + 1} failed with an unexpected error: {e}")

        if attempt < max_tries - 1:
            sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
            sleep_time = min(sleep_time, 30)
            print(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
        else:
            print("Error: All retry attempts have been exhausted. The call has failed.")
    
    return "vllm server error!!!"

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
        
        assert orig_call_server, "Agent has no call_server to wrap"
        assert orig_custom_call_tool, "Agent has no custom_call_tool to wrap"

        def wrapped_call_server(msgs, planning_port):
            return call_server_stream(self.agent, self.profiler, msgs, planning_port)
            # with self.profiler.record_function(f"llm:{getattr(self.agent, 'model', 'unknown')}"):
            #     return orig_call_server(msgs, planning_port)

        def wrapped_custom_call_tool(tool_name, tool_args, **kwargs):
            return call_tool(self.agent, self.profiler, tool_name, tool_args, **kwargs)
            # with self.profiler.record_function(f"tool: {tool_name}"):
            #     return orig_custom_call_tool(tool_name, tool_args, **kwargs)

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