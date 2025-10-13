import threading
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from contextlib import contextmanager

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
    def record_function(self, name: str, args: Optional[Dict] = None):
        thread_id = threading.get_ident()
        thread_name = threading.current_thread().name
        start_time = self._get_timestamp_us()
        
        try:
            yield
        finally:
            end_time = self._get_timestamp_us()
            duration = end_time - start_time
            
            event = {
                "name": name,
                "cat": "function",
                "ph": "X",  # Complete event
                "ts": start_time,
                "dur": duration,
                "pid": "main",
                "tid": f"{thread_name}_{thread_id}",
            }
            
            if args:
                event["args"] = args
            
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