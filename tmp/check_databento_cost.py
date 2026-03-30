import databento as db
import os

client = db.Historical(os.environ["DATABENTO_API_KEY"])

cost = client.metadata.get_cost(
    dataset="GLBX.MDP3",
    symbols=["NQ.c.0"],
    stype_in="continuous",
    schema="trades",
    start="2025-03-27",
    end="2026-03-27",
)
print(f"실제 예상 비용: ${cost:.4f}")