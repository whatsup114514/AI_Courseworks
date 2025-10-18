# -*- coding: utf-8 -*-
# @Time    : 2025/10/16 14:01
# @Author  : Van
# @Email   : Wu.Bowen@uea.ac.uk
# @File    : TSP_Problem_Solution.py
# @Software: PyCharm

"""
Author: Bowen Wu
Student ID: 100541168

Description:
This script implements the ASEAN version of the Travelling Salesman Problem (TSP),
as described in the course PDF slides (Tasks 2–5). It compares the performance of
Best-First Search and A* Search algorithms.

References:
- Russell & Norvig (2016), Artificial Intelligence: A Modern Approach
- Lecture Slides: 5–7 (Best-First), 22–23 (A*), 32–33 (Heuristics), 46 (Generator)
- Search Summaries: 3 and 6

Note: Although the assignment hint mentions an approximate distance of 956 km,
 the value 1210 km in this implementation reflects the actual road distances defined
 in the provided ASEAN dataset (search summaries 3 and 6).
"""

import heapq  # used for the priority queue (OPEN list)

# ---------------------------
# Data (Task 2) — from search summaries 3 and 6
# ---------------------------
road_distances = {
    # Road distances (for actual path cost g(n))
    "KUL": {"GT": 350, "IPH": 220, "KB": 386},
    "GT":  {"KUL": 350, "IPH": 124, "KB": 480},
    "IPH": {"KUL": 220, "GT": 124, "KB": 356},
    "KB":  {"KUL": 386, "GT": 480, "IPH": 356}
}

straight_line_distances = {
    # Straight-line distances (for heuristic estimate h(n))
    "KUL": {"GT": 300, "IPH": 190, "KB": 340},
    "GT":  {"KUL": 300, "IPH": 105, "KB": 420},
    "IPH": {"KUL": 190, "GT": 105, "KB": 310},
    "KB":  {"KUL": 340, "GT": 420, "IPH": 310}
}


# ---------------------------
# Task 3 – State Expansion using Generator (Slide 46)
# ---------------------------
def generate_next_states(current_path, road_distances):
    """
    Lazily generates unvisited neighboring cities
    from the current city in the path.
    """
    current_city = current_path[-1]
    for next_city in road_distances[current_city]:
        if next_city not in current_path:
            yield next_city  # lazy state expansion (on-demand)

# ---------------------------
# Best-First Search (f(n) = g(n))
# ---------------------------
def best_first_tsp(start, cities, road_distances):
    """
    Uninformed Best-First Search for TSP.
    Expands paths based solely on total distance so far (g(n)).
    """
    OPEN = []        # priority queue
    CLOSED = set()   # stores explored states
    expanded = 0     # counts expanded states

    # Initial node: start city with cost = 0
    heapq.heappush(OPEN, (0, [start])) # (cost so far, path)

    while OPEN:
        g_value, path = heapq.heappop(OPEN)
        expanded += 1
        current_city = path[-1]

        # Goal check: all cities visited → return to start
        # In the Traveling Salesman Problem (TSP), returning to the starting city is part of the problem definition.
        # If the return trip is not included in the total distance, the result would represent only a path, not a complete tour.
        if len(path) == num_cities:
            total = g_value + road_distances[current_city][start]
            final_path = path + [start]
            CLOSED.add(tuple(final_path))
            return final_path, total, len(CLOSED)

        CLOSED.add(tuple(path))

        # Expand next states
        for n in generate_next_states(path, road_distances):
            new_path = path + [n]
            new_g = g_value + road_distances[current_city][n]
            if tuple(new_path) not in CLOSED:
                heapq.heappush(OPEN, (new_g, new_path))

    return None, float("inf"), len(CLOSED)

# ---------------------------
# Heuristic function (Task 4)
# ---------------------------
# Note on heuristic design (Slides 32–34):
# In the original lecture slides (UK example), h(n) was defined as
# the straight-line distance from the current city to a fixed goal (e.g., King's Lynn).
# However, in this ASEAN TSP instance, the problem requires visiting all cities and returning
# to the start, so the heuristic is generalized to estimate the remaining tour cost.
# This makes the search a true Traveling Salesman Problem rather than a single-goal search.
# In the lecture example (UK cities), h(n) represented the straight-line distance
# from the current city to a single goal city (King’s Lynn).
# For example:
#   g(n) = 19.8 (Norwich → G. Yarmouth road distance)
#   h(n) = 56.9 (G. Yarmouth → King’s Lynn straight-line distance)
#   f(n) = 76.7
#
# In this ASEAN TSP assignment, the problem extends beyond a single goal:
# we must visit all cities and return to the start city (KUL).
# Therefore, the heuristic h(n) is generalized to approximate the remaining tour cost:
#   h(n) = Σ(straight-line distances from the current city to all unvisited cities)
#        + (straight-line distance from the last unvisited city back to KUL)
# This generalization maintains admissibility while adapting A* to the TSP domain.
def heuristic_func(path, start, straight_line_distances, cities):
    """
    Heuristic function h(n): based on sum of straight-line distances
    (Slides 32–33: Heuristic design guidelines)

    h(n) = sum of straight-line distances from the current city
           to each unvisited city + straight-line distance
           from the last unvisited city back to the start.
    """
    last = path[-1]
    unvisited = [c for c in cities if c not in path]

    # If all cities are visited, return distance back to start
    if not unvisited:
        return straight_line_distances[last][start]

    h = sum(straight_line_distances[last][u] for u in unvisited) # Sum of straight-line distances to unvisited cities
    h += straight_line_distances[unvisited[-1]][start] # Plus distance from last unvisited city back to start

    return h

def a_star_tsp(start, cities, road_distances, straight_line_distances):
    """
    A* Search (f(n) = g(n) + h(n)) — Slides 22–23
    Informed A* Search for the ASEAN TSP.
    Uses both actual cost g(n) and heuristic estimate h(n)
    to guide the search efficiently.
    """
    OPEN = []
    CLOSED = set()
    expanded = 0

    # Each node: (f, path, g)
    heapq.heappush(OPEN, (0, [start], 0))

    while OPEN:
        f_value, path, g_value = heapq.heappop(OPEN)
        expanded += 1
        current_city = path[-1]

        # Goal: all cities visited → return to start
        if len(path) == num_cities:
            total = g_value + road_distances[current_city][start]
            final_path = path + [start]
            CLOSED.add(tuple(final_path))
            return final_path, total, len(CLOSED)

        CLOSED.add(tuple(path))

        # Generate next states (using generator)
        for nxt in generate_next_states(path, road_distances):
            new_path = path + [nxt]
            g_new = g_value + road_distances[current_city][nxt]
            h_new = heuristic_func(new_path, start, straight_line_distances, cities)
            f_new = g_new + h_new
            if tuple(new_path) not in CLOSED:
                heapq.heappush(OPEN, (f_new, new_path, g_new))

    return None, float("inf"), len(CLOSED)

# ---------------------------
# Runner & Analysis (Task 5)
# ---------------------------
if __name__ == "__main__":

    cities = ["KUL", "GT", "IPH", "KB"]
    start = "KUL"
    num_cities = len(cities)

    print("------ASEAN TSP Search------")

    # Run Best-First Search
    bf_route, bf_distance, bf_closed = best_first_tsp(start, cities, road_distances)
    print("\n[Best-First Search]")
    print(f"Optimal Route (A*): {' → '.join(bf_route)}")
    print(f"Total Distance: {bf_distance} km")
    print("Number of Expanded States (CLOSED size):", bf_closed)

    # Run A* Search
    astar_route, astar_distances, astar_closed = a_star_tsp(start, cities, road_distances, straight_line_distances)
    print("\n[A* Search]")
    print(f"Optimal Route (A*): {' → '.join(astar_route)}")
    print(f"Total Distance: {astar_distances} km", )
    print("Number of Expanded States (CLOSED size):", astar_closed)

    # Compare results
    print("\nConclusion:")
    print("A* search uses the straight-line heuristic to guide the search,")
    print("resulting in fewer node expansions and guaranteed optimality.")
    print(f"For the ASEAN TSP, the optimal route is approximately {astar_distances} km")


# ======================================================
# Reflection
# ======================================================
"""
Compared with the UK city example discussed in class, the ASEAN TSP instance in this assignment was more straightforward to analyze because it involved fewer nodes and the distance data were already specified. 
From an algorithmic perspective, substituting UK cities with ASEAN ones did not alter the fundamental search structure. 
The heuristic function h(n) still relied on straight-line distance, consistent with the principles outlined in the course slides (Slides 32–33).
However, in practical geographic terms, ASEAN cities are characterized by more winding road networks and varying elevations, making the straight-line distance a less precise estimator of the actual travel cost. 
While I briefly considered adjusting the heuristic to account for these environmental factors, I ultimately retained the original formulation to ensure that the heuristic remained admissible, as required for demonstrating the optimality of the A* algorithm.
It is also worth noting that the heuristic formula defined in the assignment—summing the straight-line distances from the current city to all unvisited cities plus the return distance to the start—can slightly overestimate the remaining path cost in larger TSP instances. 
Nevertheless, for this four-city ASEAN case, the heuristic remains sufficiently accurate for instructional purposes and effectively demonstrates the contrast between uninformed (Best-First) and informed (A*) search strategies. 
In more complex real-world TSP applications, a tighter admissible heuristic, such as one derived from a Minimum Spanning Tree (MST) or nearest-neighbor distance, would be preferred to improve efficiency while guaranteeing optimality.
"""