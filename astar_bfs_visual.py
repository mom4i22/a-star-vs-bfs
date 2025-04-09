import heapq
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

maze = [
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0]
]

start = (0, 0)
goal = (4, 5)
rows, cols = len(maze), len(maze[0])
moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def valid(pos):
    r, c = pos
    return 0 <= r < rows and 0 <= c < cols and maze[r][c] == 0

def a_star(beginning, target):
    open_set = []

    manhattan_distance = heuristic(beginning, target)
    print("Result from Manhattan distance heuristic is:", manhattan_distance)

    heapq.heappush(open_set, (0 + heuristic(beginning, target), 0, beginning, [beginning]))
    visited = set()
    trace = []

    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        trace.append(current)

        if current == target:
            return path, trace

        if current in visited:
            continue
        visited.add(current)

        for move in moves:
            neighbor = (current[0] + move[0], current[1] + move[1])
            if valid(neighbor) and neighbor not in visited:
                heapq.heappush(open_set, (g + 1 + heuristic(neighbor, target), g + 1, neighbor, path + [neighbor]))

    return None, trace

def bfs(beginning, target):
    queue = deque([(beginning, [beginning])])
    visited = set()
    trace = []

    while queue:
        current, path = queue.popleft()
        trace.append(current)

        if current == target:
            return path, trace

        for move in moves:
            neighbor = (current[0] + move[0], current[1] + move[1])
            if valid(neighbor) and neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None, trace

a_star_path, a_star_trace = a_star(start, goal)
bfs_path, bfs_trace = bfs(start, goal)

def draw_maze(ax, maze_grid, path, trace, title):
    grid = np.array(maze_grid)
    ax.imshow(grid, cmap="Greys", origin='upper')

    for r, c in trace:
        ax.plot(c, r, 'o', color='lightblue')

    for r, c in path:
        ax.plot(c, r, 's', color='red')

    ax.plot(start[1], start[0], 'go')  # Start - green
    ax.plot(goal[1], goal[0], 'bo')    # Goal - blue

    ax.set_title(title)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.grid(True)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

draw_maze(axs[0], maze, a_star_path, a_star_trace, "A* Search")
draw_maze(axs[1], maze, bfs_path, bfs_trace, "BFS Search")

plt.tight_layout()
plt.show()