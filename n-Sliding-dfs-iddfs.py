# -*- coding: utf-8 -*-
# @Time    : 2025/10/11 21:03
# @Author  : Van
# @Email   : Wu.Bowen@uea.ac.uk
# @File    : n-Sliding-dfs-iddfs.py
# @Software: PyCharm

"""
To-do lists:
    1) The Python implementation of DFS(Depth-First Search) and IDDFS(Iterative Deepening Depth-First Search)
    2) The comparison of solution time for Tile problem, for N = 4, 8, 16, 32, 64, 128.
    3) A one-page report on the performance comparison and your thoughts about implementing the solution.
"""

import random, time, heapq


def generate_state(n):
    initial_state = list(range(n ** 2))
    random.shuffle(initial_state)
    return tuple(initial_state)

def get_neighbors(state, n):
    neighbors = []
    zero_index = state.index(0) # 取0的索引
    x,y = divmod(zero_index,n) # 一维索引 -> 二维坐标

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in moves:  # 如果一个列表里装的是 tuple，我们可以在 for 循环里同时取出 tuple 的两个元素
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < n:
            new_index = nx * n + ny  # 一维索引 = 行号 × 每行的长度 + 列号
            new_state = list(state)
            # 交换位置
            new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index] # 无需新增临时变量的方法（涉及元组）
            neighbors.append(tuple(new_state))

    return neighbors

def print_board(state, n):
    for i in range(n):  # 我们要打印一块 n×n 的拼图，所以自然需要打印 n 行
        row = state[i * n:(i + 1) * n]  #从一维的 tuple 中，切出对应行的元素
        print(' '.join(str(x) if x != 0 else ' ' for x in row))  # 把多个字符串连接成一行，中间用空格 ' ' 分隔 对于每一个 x（也就是这一行中的一个格子），如果 x != 0（不是空格），就返回 str(x)（数字转成字符串）；否则返回 ' '（空白格）。
    print()  # 空行分隔状态

def count_inversions(state):
    inversions = 0
    arr = [x for x in state if x != 0] # 过滤空格->0
    """
    等价写法：
    arr = []
    for x in state:
        if x != 0:
            arr.append(x)
    """
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inversions += 1
    """
    更python的写法
    for i in range(len(arr)):
    inversions += sum(1 for j in range(i + 1, len(arr)) if arr[i] > arr[j])
    """

    return inversions

def is_solvable(state, n):

    inversions = count_inversions(state)
    zero_index = state.index(0)
    zero_row_from_bottom = n - (zero_index // n)  # 从下往上第几行（1-based）

    if n % 2 == 1:
        # n 是奇数，只看逆序数
        return inversions % 2 == 0
    else:
        # n 是偶数，要考虑空格所在的行
        if zero_row_from_bottom % 2 == 0:
            # 空格在从下往上偶数行
            return inversions % 2 == 1
        else:
            # 空格在从下往上奇数行
            return inversions % 2 == 0

def generate_solvable_state(n):
    while True:
        state = generate_state(n)
        if is_solvable(state, n):
            return state

def dfs(state, goal, n, visited, path, stats, depth_limit = 30):

    stats["visited"] += 1
    if stats["visited"] % 100000 == 0:
        print(f"已探索 {stats['visited']} 个节点...")
    path.append(state) # 当前节点加入路径

    if state == goal:
        return path  # 找到目标

    # 到达深度限制，回溯
    if depth_limit == 0:
        path.pop()
        return None

    visited.add(state) # 防止走回头路

    for neighbor in get_neighbors(state, n):
        if neighbor not in visited:
            result = dfs(neighbor, goal, n, visited, path, stats, depth_limit - 1)
            if result:
                return result

    # 在所有子节点都访问完、并且没有找到解之后，
    # 我们准备返回上一层，这时候应该清除当前节点标记
    # 回溯清理
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
    print("\n开始 DFS 搜索...")
    result, visited, elapsed_time = solve_dfs(initial_state, goal_state, n, max_depth)
    print(f"总共访问节点：{visited} 个")
    if result:
        print(f"找到解，共 {len(result) - 1} 步，用时 {elapsed_time:.4f} 秒")
        print("\n最后三步：")
        for step in result[-3:]:
            print_board(step, n)
    else:
        print(f"未在深度 {max_depth} 内找到解，用时 {elapsed_time:.4f} 秒")

def iddfs(initial_state, goal_state, n, max_depth=30):

    start_time = time.time()
    stats = {"visited": 0}

    for depth in range(max_depth + 1):  # 从0层开始，一层层加深
        print(f"当前深度限制：{depth}")
        visited = set()
        path = []
        result = dfs(initial_state, goal_state, n, visited, path, stats, depth)
        if result:
            elapsed = time.time() - start_time
            return result, stats["visited"], elapsed
    elapsed = time.time() - start_time
    return None, stats["visited"], elapsed

def run_iddfs(initial_state, goal_state, n, max_depth=30):
    print("\n开始 IDDFS 搜索...")
    result, visited, elapsed = iddfs(initial_state, goal_state, n, max_depth)
    print(f"总共访问节点：{visited} 个")
    if result:
        print(f"找到解，共 {len(result) - 1} 步，用时 {elapsed:.4f} 秒")
        print("\n最后三步：")
        for step in result[-3:]:
            print_board(step, n)
    else:
        print(f"未在深度 {max_depth} 内找到解，用时 {elapsed:.4f} 秒")



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

    # 若未找到解
    return path + [state], steps, time.time() - start_time

def run_a_star(initial_state, goal_state, n):
    print("\n开始 A* 搜索...")
    result, visited, cost = a_star(initial_state, goal_state, n)
    print(f"总共访问节点：{visited} 个")
    if result:
        print(f" 找到解，共 {len(result) - 1} 步，用时 {cost:.4f} 秒")
        print("\n最后三步：")
        for step in result[-3:]:
            print_board(step, n)
    else:
        print(f" 未找到解。用时 {cost:.4f} 秒")

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
            return f, None  #  修复：返回两个值
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
        return min_bound, None  #  修复：返回两个值

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
    print("\n开始 IDA* 搜索...")
    result, visited, cost = ida_star(initial_state, goal_state, n)
    print(f"总共访问节点：{visited} 个")
    if result:
        print(f" 找到解，共 {len(result) - 1} 步，用时 {cost:.4f} 秒")
        print("\n最后三步：")
        for step in result[-3:]:
            print_board(step, n)
    else:
        print(f" 未找到解。用时 {cost:.4f} 秒")

def run_interface():
    while True:
        print("*" * 80)
        print(" Sliding Puzzle Solver — DFS / IDDFS / A* / IDA* ")
        print("1. 用 DFS 求解 --- n请不要超过4")
        print("2. 用 IDDFS 求解 --- n请不要超过4")
        print("3. 用 A* 求解")
        print("4. 用 IDA* 求解")
        print("5. 比较所有算法性能")
        print("6. 退出程序\n")
        print("*" * 80)

        try:
            selection = int(input("请输入选项 (1~6): "))

            if selection == 6:
                print("感谢使用，再见！")
                break

            n = int(input("请输入拼图尺寸 n（例如 3 表示 3x3 拼图）: "))
            initial_state = generate_solvable_state(n)
            goal_state = tuple(list(range(1, n ** 2)) + [0])

            print("\n初始状态：")
            print_board(initial_state, n)
            print("目标状态：")
            print_board(goal_state, n)
            print("=" * 60)

            if selection == 1:
                print(" 使用 DFS 搜索中...")
                max_depth = int(input("请输入最大深度限制 (默认 30): ") or 30)
                result, visited, cost = solve_dfs(initial_state, goal_state, n, max_depth)
                print(f"总共访问节点：{visited} 个")
                if result:
                    print(f" 找到解，共 {len(result)-1} 步，用时 {cost:.4f} 秒")
                    print("\n最后三步：")
                    for step in result[-3:]:
                        print_board(step, n)
                else:
                    print(f" 未在深度限制内找到解。用时 {cost:.4f} 秒")

            elif selection == 2:
                print(" 使用 IDDFS 搜索中...")
                max_depth = int(input("请输入最大深度限制 (默认 30): ") or 30)
                result, visited, cost = iddfs(initial_state, goal_state, n, max_depth)
                print(f"总共访问节点：{visited} 个")
                if result:
                    print(f" 找到解，共 {len(result)-1} 步，用时 {cost:.4f} 秒")
                    print("\n最后三步：")
                    for step in result[-3:]:
                        print_board(step, n)
                else:
                    print(f" 未在深度限制内找到解。用时 {cost:.4f} 秒")

            elif selection == 3:
                print(" 使用 A* 搜索中...")
                result, visited, cost = a_star(initial_state, goal_state, n)
                print(f"总共访问节点：{visited} 个")
                if result:
                    print(f" 找到解，共 {len(result) - 1} 步，用时 {cost:.4f} 秒")
                    print("\n最后三步：")
                    for step in result[-3:]:
                        print_board(step, n)
                else:
                    print(f" 未找到解。用时 {cost:.4f} 秒")

            elif selection == 4:
                print(" 使用 IDA* 搜索中...")
                result, visited, cost = ida_star(initial_state, goal_state, n)
                print(f"总共访问节点：{visited} 个")
                if result:
                    print(f" 找到解，共 {len(result)-1} 步，用时 {cost:.4f} 秒")
                    print("\n最后三步：")
                    for step in result[-3:]:
                        print_board(step, n)
                else:
                    print(f" 未找到解。用时 {cost:.4f} 秒")

            elif selection == 5:
                print(" 正在比较所有算法性能...")
                max_depth = int(input("请输入最大深度限制 (默认 30): ") or 30)

                # 依次运行各算法
                result_dfs, v_dfs, t_dfs = solve_dfs(initial_state, goal_state, n, max_depth)
                result_iddfs, v_iddfs, t_iddfs = iddfs(initial_state, goal_state, n, max_depth)
                result_astar, v_astar, t_astar = a_star(initial_state, goal_state, n)
                result_idastar, v_idastar, t_idastar = ida_star(initial_state, goal_state, n)

                print("\n 性能比较结果：")
                print(f"DFS     -> {t_dfs:.4f} 秒, 访问 {v_dfs} 个节点")
                print(f"IDDFS   -> {t_iddfs:.4f} 秒, 访问 {v_iddfs} 个节点")
                print(f"A*      -> {t_astar:.4f} 秒, 访问 {v_astar} 个节点")
                print(f"IDA*    -> {t_idastar:.4f} 秒, 访问 {v_idastar} 个节点")

                fastest = min([
                    ("DFS", t_dfs),
                    ("IDDFS", t_iddfs),
                    ("A*", t_astar),
                    ("IDA*", t_idastar)
                ], key=lambda x: x[1])

                print(f"\n 最快算法：{fastest[0]}，用时 {fastest[1]:.4f} 秒")

            else:
                print(" 输入无效，请输入 1~6。")

        except ValueError:
            print(" 输入错误，请输入数字。")



if __name__ == "__main__":
    run_interface()
    # n = 4
    # initial_state = generate_solvable_state(n)
    # goal_state = tuple(list(range(1, n ** 2)) + [0])
    #
    # print("初始状态：")
    # print_board(initial_state, n)
    # print("目标状态：")
    # print_board(goal_state, n)
    # print("=" * 40)
    # run_a_star(initial_state, goal_state, n)
    # print("=" * 40)
    # run_ida_star(initial_state, goal_state, n)
    # run_dfs(initial_state, goal_state, n, max_depth=30)
    # print("=" * 40)
    # run_iddfs(initial_state, goal_state, n, max_depth=30)