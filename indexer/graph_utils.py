import json
from pathlib import Path
from collections import deque
from typing import Dict, List


def load_dependency_graph() -> Dict[str, List[str]]:
    """Load dependency graph from JSON and normalize paths."""
    graph_file = Path(__file__).with_name("dependency_graph.json")
    with graph_file.open("r", encoding="utf-8") as f:
        raw_graph: Dict[str, List[str]] = json.load(f)

    normalized: Dict[str, List[str]] = {}
    for key, values in raw_graph.items():
        norm_key = Path(key).as_posix()
        normalized[norm_key] = [Path(v).as_posix() for v in values]
    return normalized


def get_related_files(filepath: str, depth: int = 1) -> List[str]:
    """Return files related to ``filepath`` up to ``depth`` using BFS."""
    if depth < 1:
        return []

    graph = load_dependency_graph()
    start = Path(filepath).as_posix()

    visited = {start}
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    result: List[str] = []

    while queue:
        current, dist = queue.popleft()
        if dist == depth:
            continue
        neighbours = graph.get(current, [])
        for neighbour in sorted(neighbours):
            neighbour = Path(neighbour).as_posix()
            if neighbour not in visited:
                visited.add(neighbour)
                result.append(neighbour)
                queue.append((neighbour, dist + 1))

    return result

