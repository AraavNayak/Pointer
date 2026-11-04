"""Time one full review-board session, to see where the seconds go.

Run:  python -m eval.latency [episode_id]
Makes real Muse Spark calls (several per run) and real MuJoCo re-runs.
"""

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from server import agent, board  # noqa: E402

if __name__ == "__main__":
    ep = agent.get_episode(sys.argv[1] if len(sys.argv) > 1 else None)
    t0 = time.time()
    for e in board.run(ep, "Why did this run fail?", 0, []):
        t = time.time() - t0
        if e["type"] == "agent_step":
            print(f"{t:6.1f}s  {e['agent']:<11} step {e['n']}  model {e['model_ms'] / 1000:.1f}s  {e['label']}")
        elif e["type"] == "hypothesis":
            print(f"{t:6.1f}s  {e['agent']:<11} hypothesis ({e['confidence']:.0%}): {e['claim']}")
        elif e["type"] == "challenge":
            print(f"{t:6.1f}s  skeptic     {e['target']}: {e['status']}")
        elif e["type"] == "experiment_result":
            print(f"{t:6.1f}s  {e['agent']:<11} MuJoCo {e['sim_ms'] / 1000:.1f}s  {e['verdict']['label']}")
        elif e["type"] == "decision":
            print(f"{t:6.1f}s  decision    {e['summary']}")
        elif e["type"] == "answer":
            print(f"{t:6.1f}s  findings    model {e['answer']['model_ms'] / 1000:.1f}s")
        elif e["type"] in ("agent_error", "experiment_error", "error"):
            print(f"{t:6.1f}s  ERROR {e.get('message', '')}")
