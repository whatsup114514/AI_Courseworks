# -*- coding: utf-8 -*-
# @Time    : 2025/10/11 21:03
# @Author  : Van
# @Email   : Wu.Bowen@uea.ac.uk
# @File    : n-Sliding-dfs-iddfs.py
# @Software: PyCharm

"""
To-do list:
    1) Implement DFS (Depth-First Search) and IDDFS (Iterative Deepening Depth-First Search) in Python.
    2) Compare solution times for the Tile Puzzle for N = 4, 8, 16, 32, 64, 128.
    3) Write a one-page report comparing performance and discussing your implementation insights.
"""

import random, time, heapq


def generate_state(n):
    initial_state = list(range(n ** 2))
    random.shuffle(initial_state)
    return tuple(initial_state)

def get_neighbors(state, n):
    neighbors = []
    zero_index = state.index(0)  # Index of the blank tile (0)
    x, y = divmod(zero_index, n)  # Convert 1D index -> 2D coordinates

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < n:
            new_index = nx * n + ny  # 2D -> 1D index
            new_state = list(state)
            # Swap blank with neighbor
            new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index]
            neighbors.append(tuple(new_state))

    return neighbors

def print_board(state, n):
    for i in range(n):
        row = state[i * n:(i + 1) * n]
        print(' '.join(str(x) if x != 0 else ' ' for x in row))
    print()

def count_inversions(state):
    inversions = 0
    arr = [x for x in state if x != 0]  # Filter out the blank (0)
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inversions += 1
    return inversions

def is_solvable(state, n):
    inversions = count_inversions(state)
    zero_index = state.index(0)
    zero_row_from_bottom = n - (zero_index // n)  # Row number from bottom (1-based)

    if n % 2 == 1:
        # Odd n: solvable if inversion count is even
        return inversions % 2 == 0
    else:
        # Even n: depends on blank tile row
        if zero_row_from_bottom % 2 == 0:
            return inversions % 2 == 1
        else:
            return inversions % 2 == 0

def generate_solvable_state(n):
    while True:
        state = generate_state(n)
        if is_solvable(state, n):
            return state

def dfs(state, goal, n, visited, path, stats, depth_limit=30):
    stats["visited"] += 1
    if stats["visited"] % 100000 == 0:
        print(f"Explored {stats['visited']} nodes so far...")
    path.append(state)

    if state == goal:
        return path

    if depth_limit == 0:
        path.pop()
        return None

    visited.add(state)

    for neighbor in get_neighbors(state, n):
        if neighbor not in visited:
            result = dfs(neighbor, goal, n, visited, path, stats, depth_limit - 1)
            if result:
                return result

    visited.remove(state)
    path.pop()
    return None

def solve_dfs(initial_state, goal_state, n, max_depth=30):
    start_time = time.time()
    stats = {"visited": 0}
    visited = set()
    path = []
    result = dfs(initial_state, goal_state, n, visited, path, stats, max_depth)
    elapsed = time.time() - start_time
    return result, stats["visited"], elapsed

def run_dfs(initial_state, goal_state, n, max_depth=30):
    print("\nStarting DFS search...")
    result, visited, elapsed_time = solve_dfs(initial_state, goal_state, n, max_depth)
    print(f"Total nodes visited: {visited}")
    if result:
        print(f"Solution found in {len(result) - 1} steps, time: {elapsed_time:.4f} seconds")
        print("\nLast three steps:")
        for step in result[-3:]:
            print_board(step, n)
    else:
        print(f"No solution found within depth {max_depth}, time: {elapsed_time:.4f} seconds")

def iddfs(initial_state, goal_state, n, max_depth=30):
    start_time = time.time()
    stats = {"visited": 0}

    for depth in range(max_depth + 1):
        print(f"Current depth limit: {depth}")
        visited = set()
        path = []
        result = dfs(initial_state, goal_state, n, visited, path, stats, depth)
        if result:
            elapsed = time.time() - start_time
            return result, stats["visited"], elapsed
    elapsed = time.time() - start_time
    return None, stats["visited"], elapsed

def run_iddfs(initial_state, goal_state, n, max_depth=30):
    print("\nStarting IDDFS search...")
    result, visited, elapsed = iddfs(initial_state, goal_state, n, max_depth)
    print(f"Total nodes visited: {visited}")
    if result:
        print(f"Solution found in {len(result) - 1} steps, time: {elapsed:.4f} seconds")
        print("\nLast three steps:")
        for step in result[-3:]:
            print_board(step, n)
    else:
        print(f"No solution found within depth {max_depth}, time: {elapsed:.4f} seconds")

def a_star(initial_state, goal_state, n):
    def manhattan_distance(state):
        distance = 0
        for num in range(1, n ** 2):
            x1, y1 = divmod(state.index(num), n)
            x2, y2 = divmod(goal_state.index(num), n)
            distance += abs(x1 - x2) + abs(y1 - y2)
        return distance

    open_list = []
    heapq.heappush(open_list, (manhattan_distance(initial_state), 0, initial_state, []))
    closed_set = set()
    steps = 0
    start_time = time.time()

    while open_list:
        f, g, state, path = heapq.heappop(open_list)
        steps += 1

        if state == goal_state:
            elapsed = time.time() - start_time
            return path + [state], steps, elapsed

        closed_set.add(state)

        for neighbor in get_neighbors(state, n):
            if neighbor not in closed_set:
                new_g = g + 1
                new_f = new_g + manhattan_distance(neighbor)
                heapq.heappush(open_list, (new_f, new_g, neighbor, path + [state]))

    return path + [state], steps, time.time() - start_time

def run_a_star(initial_state, goal_state, n):
    print("\nStarting A* search...")
    result, visited, cost = a_star(initial_state, goal_state, n)
    print(f"Total nodes visited: {visited}")
    if result:
        print(f"Solution found in {len(result) - 1} steps, time: {cost:.4f} seconds")
        print("\nLast three steps:")
        for step in result[-3:]:
            print_board(step, n)
    else:
        print(f"No solution found. Time: {cost:.4f} seconds")

def ida_star(initial_state, goal_state, n):
    def manhattan_distance(state):
        distance = 0
        for num in range(1, n ** 2):
            x1, y1 = divmod(state.index(num), n)
            x2, y2 = divmod(goal_state.index(num), n)
            distance += abs(x1 - x2) + abs(y1 - y2)
        return distance

    def search(path, g, bound):
        node = path[-1]
        f = g + manhattan_distance(node)
        if f > bound:
            return f, None
        if node == goal_state:
            return "FOUND", path.copy()
        min_bound = float("inf")
        for neighbor in get_neighbors(node, n):
            if neighbor not in path:
                path.append(neighbor)
                result, found_path = search(path, g + 1, bound)
                if result == "FOUND":
                    return "FOUND", found_path
                if result < min_bound:
                    min_bound = result
                path.pop()
        return min_bound, None

    bound = manhattan_distance(initial_state)
    path = [initial_state]
    start_time = time.time()
    steps = 0

    while True:
        result, found_path = search(path, 0, bound)
        steps += 1
        if result == "FOUND":
            return found_path, steps, time.time() - start_time
        if result == float("inf"):
            return None, steps, time.time() - start_time
        bound = result

def run_ida_star(initial_state, goal_state, n):
    print("\nStarting IDA* search...")
    result, visited, cost = ida_star(initial_state, goal_state, n)
    print(f"Total nodes visited: {visited}")
    if result:
        print(f"Solution found in {len(result) - 1} steps, time: {cost:.4f} seconds")
        print("\nLast three steps:")
        for step in result[-3:]:
            print_board(step, n)
    else:
        print(f"No solution found. Time: {cost:.4f} seconds")

def run_interface():
    while True:
        print("*" * 80)
        print(" Sliding Puzzle Solver — DFS / IDDFS / A* / IDA* ")
        print("1. Solve with DFS --- recommended n ≤ 4")
        print("2. Solve with IDDFS --- recommended n ≤ 4")
        print("3. Solve with A*")
        print("4. Solve with IDA*")
        print("5. Compare all algorithm performances")
        print("6. Exit program\n")
        print("*" * 80)

        try:
            selection = int(input("Please enter your choice (1~6): "))

            if selection == 6:
                print("Thank you for using the solver. Goodbye!")
                break

            n = int(input("Enter puzzle size n (e.g., 3 for a 3x3 puzzle): "))
            initial_state = generate_solvable_state(n)
            goal_state = tuple(list(range(1, n ** 2)) + [0])

            print("\nInitial state:")
            print_board(initial_state, n)
            print("Goal state:")
            print_board(goal_state, n)
            print("=" * 60)

            if selection == 1:
                print("Running DFS...")
                max_depth = int(input("Enter maximum depth limit (default 30): ") or 30)
                result, visited, cost = solve_dfs(initial_state, goal_state, n, max_depth)
                print(f"Total nodes visited: {visited}")
                if result:
                    print(f"Solution found in {len(result)-1} steps, time: {cost:.4f} seconds")
                    print("\nLast three steps:")
                    for step in result[-3:]:
                        print_board(step, n)
                else:
                    print(f"No solution found within depth limit. Time: {cost:.4f} seconds")

            elif selection == 2:
                print("Running IDDFS...")
                max_depth = int(input("Enter maximum depth limit (default 30): ") or 30)
                result, visited, cost = iddfs(initial_state, goal_state, n, max_depth)
                print(f"Total nodes visited: {visited}")
                if result:
                    print(f"Solution found in {len(result)-1} steps, time: {cost:.4f} seconds")
                    print("\nLast three steps:")
                    for step in result[-3:]:
                        print_board(step, n)
                else:
                    print(f"No solution found within depth limit. Time: {cost:.4f} seconds")

            elif selection == 3:
                print("Running A*...")
                result, visited, cost = a_star(initial_state, goal_state, n)
                print(f"Total nodes visited: {visited}")
                if result:
                    print(f"Solution found in {len(result)-1} steps, time: {cost:.4f} seconds")
                    print("\nLast three steps:")
                    for step in result[-3:]:
                        print_board(step, n)
                else:
                    print(f"No solution found. Time: {cost:.4f} seconds")

            elif selection == 4:
                print("Running IDA*...")
                result, visited, cost = ida_star(initial_state, goal_state, n)
                print(f"Total nodes visited: {visited}")
                if result:
                    print(f"Solution found in {len(result)-1} steps, time: {cost:.4f} seconds")
                    print("\nLast three steps:")
                    for step in result[-3:]:
                        print_board(step, n)
                else:
                    print(f"No solution found. Time: {cost:.4f} seconds")

            elif selection == 5:
                print("Comparing algorithm performance...")
                max_depth = int(input("Enter maximum depth limit (default 30): ") or 30)

                result_dfs, v_dfs, t_dfs = solve_dfs(initial_state, goal_state, n, max_depth)
                result_iddfs, v_iddfs, t_iddfs = iddfs(initial_state, goal_state, n, max_depth)
                result_astar, v_astar, t_astar = a_star(initial_state, goal_state, n)
                result_idastar, v_idastar, t_idastar = ida_star(initial_state, goal_state, n)

                print("\nPerformance Comparison:")
                print(f"DFS     -> {t_dfs:.4f} sec, visited {v_dfs} nodes")
                print(f"IDDFS   -> {t_iddfs:.4f} sec, visited {v_iddfs} nodes")
                print(f"A*      -> {t_astar:.4f} sec, visited {v_astar} nodes")
                print(f"IDA*    -> {t_idastar:.4f} sec, visited {v_idastar} nodes")

                fastest = min([
                    ("DFS", t_dfs),
                    ("IDDFS", t_iddfs),
                    ("A*", t_astar),
                    ("IDA*", t_idastar)
                ], key=lambda x: x[1])

                print(f"\nFastest algorithm: {fastest[0]}, time: {fastest[1]:.4f} seconds")

            else:
                print("Invalid input. Please enter 1~6.")

        except ValueError:
            print("Invalid input. Please enter a number.")


if __name__ == "__main__":
    run_interface()
