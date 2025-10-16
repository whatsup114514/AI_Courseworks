# -*- coding: utf-8 -*-
# @Time    : 2025/10/11 22:01
# @Author  : Van
# @Email   : Wu.Bowen@uea.ac.uk
# @File    : n-Queens-dfs-iddfs.py
# @Software: PyCharm

"""
To-do list:
    1) Implement DFS (Depth-First Search) and IDDFS (Iterative Deepening Depth-First Search) in Python.
    2) Compare the solution time for the N-Queens problem for N = 4, 8, 16, 32, 64, 128.
    3) Write a one-page report on performance comparison and your thoughts on the implementation.
"""
import time


def dfs_method(n):
    start_time = time.time()
    solution = []
    # Store all found solutions

    position = [-1] * n
    # A list of length n: index represents the row, value represents the column

    def rules_checker(position, row, col):
        for r in range(row):
            c = position[r]  # Iterate through already placed queens, each has coordinates (r, c)
            if ((c == col)  # Same column
            or (abs(row - r) == abs(col - c))):  # Same diagonal
                return False
        return True

    def dfs(row):
        """
        The idea of DFS (Depth-First Search) is:
        Place a queen in row 0, then row 1, then row 2, and so on.
        If all rows are successfully filled, a valid solution is found.
        When row == n, all rows are filled.

        Think of dfs(row) as a node in a recursion tree.
        Each node represents placing a queen in a certain row.
        Each path represents an attempt from row 0 up to row n-1.
        When row == n, we reach a leaf node (a successful solution).
        """
        if row == n:
            # position is mutable; copy it to avoid being overwritten during backtracking
            solution.append(position.copy())
            return

        for col in range(n):
            if rules_checker(position, row, col):
                position[row] = col  # Place queen at (row, col)
                dfs(row + 1)         # Proceed to the next row
                position[row] = -1   # Backtrack

    dfs(0)  # Start recursion from row 0
    end_time = time.time()

    elapsed_time = end_time - start_time
    return solution, elapsed_time


def iddfs_method(n):
    """
    IDDFS is essentially a DFS with depth limitation, combined with a gradual deepening mechanism.
    """

    start_time = time.time()
    position = [-1] * n
    solutions = []

    def rules_checker(position, row, col):
        for r in range(row):
            c = position[r]
            if ((c == col)
            or (abs(row - r) == abs(col - c))):
                return False
        return True

    def depth_limited_collector(row, limit):
        if row == n:
            # Copy position to preserve the found solution
            solutions.append(position.copy())
            return
        if row == limit:
            return
        for col in range(n):
            if rules_checker(position, row, col):
                position[row] = col
                depth_limited_collector(row + 1, limit)  # Recursive call
                position[row] = -1  # Backtrack

    # Search from shallow to deep — that’s why it’s called IDDFS
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
    print("This program implements DFS and IDDFS to solve the N-Queens problem.")
    print("1. Solve the problem using DFS")
    print("2. Solve the problem using IDDFS")
    print("3. Solve the problem and compare the performance of DFS and IDDFS")
    print("4. Exit the program\n")
    print("*" * 80)
    selection = int(input("Please select your choice (1/2/3/4): "))

    try:
        if selection == 1:
            n = int(input("Please enter the value of n: "))
            print("Solving the N-Queens problem with DFS...")

            solutions, cost_time = dfs_method(n)
            print(f"DFS solved {n}-Queens in {cost_time:.6f} seconds, finding {len(solutions)} solutions.")
            print("Displaying the last 3 solutions...")
            for i in solutions[-3:]:
                print_board(i)
            print("*" * 80)

        elif selection == 2:
            n = int(input("Please enter the value of n: "))
            print("Solving the N-Queens problem with IDDFS...")

            solutions, cost_time = iddfs_method(n)
            print(f"IDDFS solved {n}-Queens in {cost_time:.6f} seconds, finding {len(solutions)} solutions.")
            print("Displaying the last 3 solutions...")
            for i in solutions[-3:]:
                print_board(i)
            print("*" * 80)

        elif selection == 3:
            n = int(input("Please enter the value of n: "))
            print("Comparing the performance of DFS and IDDFS...")

            dfs_solutions, dfs_time = dfs_method(n)
            iddfs_solutions, iddfs_time = iddfs_method(n)

            print(f"\nDFS time: {dfs_time:.6f} seconds, {len(dfs_solutions)} solutions found.")
            print(f"IDDFS time: {iddfs_time:.6f} seconds, {len(iddfs_solutions)} solutions found.")
            print(f"\nDFS is approximately {iddfs_time / dfs_time:.2f} times faster than IDDFS.")
            print("*" * 80)

        elif selection == 4:
            print("Thank you for testing this program!")
            quit()

    except ValueError:
        print("Invalid input. Please try again.")
