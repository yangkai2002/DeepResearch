import threading
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from contextlib import contextmanager


class EventContext:
    """Mutable args container passed to the body of a profiled region.

    Usage:
        with profiler.record_function("llm: decode", {"model": model}) as evt:
            # in-context add more fields
            evt.set(tokens=tok_count, latency_ms=12.3)
            # or
            evt.update({"step": 2})

    All fields collected in this context are merged into the event's `args` at exit.
    """
    def __init__(self, initial: Optional[Dict[str, Any]] = None):
        self._args: Dict[str, Any] = dict(initial) if initial else {}

    def set(self, **kwargs: Any) -> None:
        self._args.update(kwargs)

    def add(self, key: str, value: Any) -> None:
        self._args[key] = value

    def update(self, data: Dict[str, Any]) -> None:
        self._args.update(data)

    @property
    def args(self) -> Dict[str, Any]:
        return self._args

class SimpleProfiler:
    def __init__(self, profile_dir: str = "./profiler_traces"):
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(exist_ok=True)
        self.profiler_start_time = time.perf_counter()
        
        self.events: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
    
    def _get_timestamp_us(self) -> int:
        return int((time.perf_counter() - self.profiler_start_time) * 1_000_000)
    
    @contextmanager
    def record_function(self, name: str, args: Optional[Dict[str, Any]] = None):
        thread_id = threading.get_ident()
        thread_name = threading.current_thread().name
        start_time = self._get_timestamp_us()

        ctx = EventContext(args)

        try:
            # Yield a mutable context so caller can populate extra fields during the block
            yield ctx
        finally:
            end_time = self._get_timestamp_us()
            duration = end_time - start_time

            event: Dict[str, Any] = {
                "name": name,
                "cat": "function",
                "ph": "X",  # Complete event
                "ts": start_time,
                "dur": duration,
                "pid": "main",
                "tid": f"{thread_name}_{thread_id}",
            }

            if ctx.args:
                event["args"] = ctx.args

            with self.lock:
                self.events.append(event)
    
    def export_trace(self, filename: Optional[str] = None) -> str:
        if filename is None:
            filename = f"trace_{int(time.time())}.json"
        
        filepath = self.profile_dir / filename
        trace_data = {"traceEvents": self.events.copy(), "displayTimeUnit": "ms"}
        with open(filepath, 'w') as f:
            json.dump(trace_data, f, indent=2)

        print(f"[SimpleProfiler] Trace exported to {filepath}, view in chrome://tracing/ by loading the file")
        return str(filepath)
    
    def clear_events(self):
        with self.lock:
            self.events.clear()