# -*- coding: utf-8 -*-
# @Time    : 2025/10/11 22:01
# @Author  : Van
# @Email   : Wu.Bowen@uea.ac.uk
# @File    : n-Queens-dfs-iddfs.py
# @Software: PyCharm

"""
To-do lists:
    1) The Python implementation of DFS(Depth-First Search) and IDDFS(Iterative Deepening Depth-First Search)
    2) The comparison of solution time for N-Queen, for N = 4, 8, 16, 32, 64, 128.
    3) A one-page report on the performance comparison and your thoughts about implementing the solution.
"""
import time

def dfs_method(n):
    start_time = time.time()
    solution = []
    # 储存找到的解

    position = [-1] * n
    # 长度为n的列表，index来表示行，value表示列

    def rules_checker(position, row, col):
        for r in range(row):
            c = position[r] # 遍历已经放好的皇后 每个皇后的坐标是 (r, c)
            if ((c == col) # 如果之前某个皇后 c 的列号和现在要放的 col 相同
            or (abs(row - r) == abs(col - c))): # 如果两个皇后行号的差和列号的差相同 那它们就在同一条对角线上
                return False
        return True

    def dfs(row):
        """
        DFS（深度优先搜索）的思路是：
        我先放好第 0 行，再去放第 1 行，再放第 2 行……
        一直放到最后一行，如果都能放下，那就找到一个完整的解。
        如果 row == n，说明所有行都放好了

        我们把整个 dfs(row) 看成一棵 递归树 的节点。
        每个节点表示：当前在第 row 行放皇后；
        每条路径表示：从第 0 行放到第 row 行的一种尝试；
        当走到 row == n 时，就走到了一条完整路径的底部（即一个成功的方案）。
        """
        if row == n:
            # position 是一个可变列表，它会在回溯时被修改。
            # 如果直接保存 position 本身，那么以后它会被覆盖。
            solution.append(position.copy())
            return

        for col in range(n):
            if rules_checker(position, row, col):
                position[row] = col # 把皇后放在 (row, col) 位置上
                dfs(row + 1) # 第 row 行放好了，接下来去放第 row+1 行的皇后
                position[row] = -1 # 回溯

    dfs(0) # 启动递归，从第 0 行开始搜索
    end_time = time.time()

    elapsed_time = end_time - start_time
    return solution, elapsed_time

def iddfs_method(n):
    """
    IDDFS其实是带深度限制的 DFS，在此基础上加了逐步加深的机制

    """

    start_time = time.time()
    position = [-1] * n
    solutions = []

    def rules_checker(position, row, col):
        for r in range(row):
            c = position[r]  # 遍历已经放好的皇后 每个皇后的坐标是 (r, c)
            if ((c == col)  # 如果之前某个皇后 c 的列号和现在要放的 col 相同
            or (abs(row - r) == abs(col - c))):  # 如果两个皇后行号的差和列号的差相同 那它们就在同一条对角线上
                return False

        return True

    def depth_limited_collector(row, limit):
        if row == n:
            # position 是一个可变列表，它会在回溯时被修改。
            # 如果直接保存 position 本身，那么以后它会被覆盖。
            solutions.append(position.copy())
            return
        if row == limit:
            return
        for col in range(n):
            if rules_checker(position, row, col):
                position[row] = col
                depth_limited_collector(row + 1, limit) # 递归
                position[row] = -1 # 回溯

    # 从浅到深搜索，这也是为什么它叫iddfs算法
    for limit in range(1, n + 1):
        depth_limited_collector(0, limit)

    end_time = time.time()
    elapsed_time = end_time - start_time
    return solutions, elapsed_time

def print_board(position):
    n = len(position)
    for row in range(n):
        line = ""
        for col in range(n):
            line += "Q " if position[row] == col else ". "
        print(line)
    print("\n")




while True:

    print("*" * 80)
    print("This program implements DFS and IDDFS to solve N-Queen problem.")
    print("1. To solve the problem with DFS")
    print("2. To solve the problem with IDDFS")
    print("3. To solve the problem and compare the performance of DFS and IDDFS")
    print("4. Exit the program\n")
    print("*" * 80)
    selection = int(input("Please select your choice using 1/2/3/4: "))


    try:
        if selection == 1:
            n = int(input("Please enter the number of n: "))
            print("Now we solve N-Queen problem with DFS...")

            solutions, cost_time = dfs_method(n)
            print(f"DFS 解决 {n} 皇后用了 {cost_time:.6f} 秒，共找到 {len(solutions)} 个解。")
            print("Now we display the last 3 solutions...")
            for i in solutions[-3:]: # 打印后三个解
                print_board(i)
            print("*" * 80)

        elif selection == 2:
            n = int(input("Please enter the number of n: "))
            print("Now we solve N-Queen problem with IDDFS...")

            solutions, cost_time = iddfs_method(n)
            print(f"IDDFS 解决 {n} 皇后用了 {cost_time:.6f} 秒，共找到 {len(solutions)} 个解。")
            print("Now we display the last 3 solutions...")
            for i in solutions[-3:]:  # 打印后三个解
                print_board(i)
            print("*" * 80)


        elif selection == 3:

            n = int(input("Please enter the number of n: "))

            print("Now we compare the performance of DFS and IDDFS...")

            dfs_solutions, dfs_time = dfs_method(n)
            iddfs_solutions, iddfs_time = iddfs_method(n)

            print(f"\nDFS 用时: {dfs_time:.6f} 秒，共 {len(dfs_solutions)} 个解")
            print(f"IDDFS 用时: {iddfs_time:.6f} 秒，共 {len(iddfs_solutions)} 个解")
            print(f"\nDFS 比 IDDFS 大约快了 {iddfs_time / dfs_time:.2f} 倍 ")
            print("*" * 80)

        elif selection == 4:
            print("Thank you for testing this program!")
            quit()

    except ValueError:
        print("Invalid input, please try again.")







