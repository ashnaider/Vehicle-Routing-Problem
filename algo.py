import math
from textwrap import indent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from IPython.display import display

from pickle import NONE
# from sty import fg, bg, ef, rs

from functools import reduce


test_O = (10, 15)
test_x = [17, 6, 13, 9, 19, 8, 4, 17, 12, 6, 19, 12]
test_y = [15, 15, 3, 20, 7, 8, 14, 2, 22, 12, 17, 8]
test_orderings = [0, 450, 400, 400, 200, 150, 450, 250, 200, 450, 300, 475, 550]

my_O = (25, 25)
my_x = [29, 20, 10, 11, 7, 4, 2, 8, 15, 21, 46, 0]
my_y = [18, 27, 44, 34, 41, 0, 19, 32, 1, 30, 43, 19]
my_orderings = [0, 300, 225, 625, 225, 450, 400, 700, 325, 250, 550, 200, 650]


def plot_points(x, y, center, connect_radial=False, routes=None):
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y, marker='D')
    plt.scatter([center[0]], [center[1]], marker='8', s=70)

    plt.annotate("center", (center[0]+0.3, center[1]), fontsize=14)
    for i in range(len(x)):
        plt.annotate(str(i+1), (x[i]+0.5, y[i]), fontsize=14)

    colors = ['red', 'blue', 'green', 'orange', 'cyan', 'magenta']

    if routes is not None:
        for j, route in enumerate(routes):
            if len(route) > 1:
                for i in range(1, len(route)):
                    a = route[i] - 1
                    b = route[i-1] - 1
                    plt.plot([x[a], x[b]], [y[a], y[b]], color=colors[j])
            
            start = route[0] - 1
            end = route[-1] - 1
            plt.plot([center[0], x[start]], [center[1], y[start]], color=colors[j])
            plt.plot([center[0], x[end]], [center[1], y[end]], color=colors[j])


    elif connect_radial:
        for i in range(len(x)):
            plt.plot([center[0], x[i]], [center[1], y[i]], color='red')

    plt.grid()
    plt.show()


def print_matrix(mat):
    np.set_printoptions(precision=1)
    print()
    for i, row in enumerate(mat):
        for j, val in enumerate(row):
            if i == j:
                aligned_ij = f"{float(i):>4.1f}"
                bg_ij = bg.blue + fg.white + aligned_ij + fg.rs + bg.rs
                print(bg_ij, end="  ")
            else:                
                print(f"{val:>4.1f}", end="  ")
        print()
    print()

def calc_dist_matrix(x, y):
    res = np.zeros((len(x), len(x)))

    for i in range(len(x)):
        for j in range(i, len(x)):
            res[i, j] = euclid_dist((x[i], y[i]), (x[j], y[j]))

    return res

def euclid_dist(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def calc_win_matrix(x, y, dist_m=None):
    if dist_m is None:
        dist_m = calc_dist_matrix(x, y)

    len_x = len(x)
    res = np.zeros((len_x, len_x))
    for j in range(len_x):
        for i in range(j):
            res[i, j] = dist_m[0, i] + dist_m[0, j] - dist_m[i, j]

    return res.T


def ClarkeRight(x, y, orderings, capacity):
    dist_m = calc_dist_matrix(x, y)
    win_m = calc_win_matrix(x, y, dist_m=dist_m)

    columns = ['iteration', 'i*', 'j*', 'S max', 'C1', 'C2', 'C3', 'q1', 'q2', 'q1 + q2 <= c', 'Route', 'NKB']
    info_df = pd.DataFrame(columns=columns)

    Smax = 0
    counter = 0
    i, j = 0, 0

    # generate default(radial) routes
    routes = [[i] for i in range(1, len(x))]
    NKB = []

    while True:
        counter += 1
        if counter > 5000:
            break

        win_m[i, j] = -1
        curr_max = np.max(win_m)
        if curr_max <= 0:
            break

        curr_max_ind = np.argmax(win_m)
        i, j = curr_max_ind // win_m.shape[1], curr_max_ind % win_m.shape[1]

        curr_row = counter - 1
        d = dict.fromkeys(columns, '-')
        info_df = info_df.append(d, ignore_index=True) 
        info_df.loc[curr_row, 'iteration'] = counter
        info_df.loc[curr_row, 'i*'] = i
        info_df.loc[curr_row, 'j*'] = j
        info_df.loc[curr_row, 'S max'] = curr_max
        
 
        ### STEP 1
        ### Perform checking of three conditions 
        # 1. check if cell used
        if win_m[i, j] < 0:
            continue

        info_df.loc[curr_row, 'C1'] = '+'
        info_df.loc[curr_row, 'C2'] = '+'
        info_df.loc[curr_row, 'C3'] = '+'

        broke_C2 = False
        broke_C3 = False
        i_route, j_route = None, None
        i_route_ind, j_route_ind = -1, -1
        found_i_route, found_j_route = False, False
        for route_ind, route in enumerate(routes):
            # 2. check if i and j are not in the same route
            if i in route and j in route:
                broke_C2 = True
                info_df.loc[curr_row, 'C1'] = '-'

            # 3.1 check if i is start or end, not node of the route
            if i in route and (i not in route[1:-1] or len(route) == 1) and not found_i_route:
                i_route = route
                i_route_ind = route_ind
                found_i_route = True
            # 3.2 check if j is start or end, not node of the route
            if j in route and (j not in route[1:-1] or len(route) == 1) and not found_j_route: 
                j_route = route
                j_route_ind = route_ind
                found_j_route = True

            if broke_C2:
                i_route, j_route = None, None

            if i in route[1:-1] or j in route[1:-1]:
                info_df.loc[curr_row, 'C2'] = '-'

        if i_route is None or j_route is None:
            continue

        # reverse j_route
        if i == i_route[0] and j == j_route[0]:
            i_route.reverse()
        if i == i_route[-1] and j == j_route[-1]:
            j_route.reverse()
        if i == i_route[0] and j == j_route[-1]:
            i_route.reverse()
            j_route.reverse()
            
        assert i == i_route[-1]
        assert j == j_route[0]

        ### STEP 2
        # Calculate the total volume of deliveries on routes 1 and 2
        q1, q2 = 0, 0
        for k in i_route:
            q1 += orderings[k]
        for k in j_route:
            q2 += orderings[k]

        info_df.loc[curr_row, 'q1'] = q1
        info_df.loc[curr_row, 'q2'] = q2

        ### STEP 3
        # check if q1 + q1 <= capacity
        if q1 + q2 > capacity:
            continue

        info_df.loc[curr_row, 'q1 + q2 <= c'] = '+'

        if i_route_ind > j_route_ind:
            i_route_ind, j_route_ind = j_route_ind, i_route_ind

        new_route = i_route + j_route
        routes.pop(j_route_ind)
        routes.pop(i_route_ind)
        routes.append(new_route)
        
        Smax += curr_max

        info_df.loc[curr_row, 'Route'] = [0] + new_route + [0]
        info_df.loc[curr_row, 'NKB'] = Smax

        NKB.append(curr_max)

    res_cols = ['№', 'Маршрут', 'Обсяг постачання, од', 'Пробіг, км']
    res = pd.DataFrame(columns=res_cols)
    total_vol = 0
    total_milage = 0
    for i, route in enumerate(routes):
        vol = 0
        for v in route:
            vol += orderings[v]
        total_vol += vol

        t = dist_m[0, route[0]]
        milage = t
        print("t: ", t)
        t = dist_m[0, route[-1]]
        milage += t
        print("t: ", t)
        if len(route) > 1:
            for k in range(len(route)-1):
                t = dist_m[min(route[k], route[k+1]), max(route[k], route[k+1])] 
                print("t: ", t)
                milage += t
        total_milage += milage
        print()

        dr = dict(zip(res_cols, [i+1, [0] + route + [0], vol, milage]))
        res = res.append(dr, ignore_index=True)

    return routes, NKB, total_vol, total_milage, res, info_df


# def find_i_j_route(i, j, routes):

#     i_route, j_route = None, None
#     i_route_ind, j_route_ind = -1, -1
#     found_i_route, found_j_route = False, False
#     for route_ind, route in enumerate(routes):
#         # 2. check if i and j are not in the same route
#         if i in route and j in route:
#             break
#         # 3.1 check if i is start or end, not node of the route
#         if i in route and (i not in route[1:-1] or len(route) == 1) and not found_i_route:
#             i_route = route
#             i_route_ind = route_ind
#             found_i_route = True

#         # 3.2 check if j is start or end, not node of the route
#         if j in route and (j not in route[1:-1] or len(route) == 1) and not found_j_route: 
#             j_route = route
#             j_route_ind = route_ind
#             found_j_route = True

#     return i_route, i_route_ind, j_route, j_route_ind


def solve_task(test=False):
    pd.set_option('precision', 2)

    # plot_points(test_x, test_y, test_O, connect_radial=True)
    # plot_points(my_x, my_y, my_O, connect_radial=True)
    

    if test:
        x = test_x[:]
        y = test_y[:]
        x.insert(0, test_O[0])
        y.insert(0, test_O[1])
        orderings = test_orderings
        cap = 1500
    else:
        x = my_x[:]
        y = my_y[:]
        x.insert(0, my_O[0])
        y.insert(0, my_O[1])
        orderings = my_orderings
        cap = 1500

    dists = calc_dist_matrix(x, y)
    win_m = calc_win_matrix(x, y)

    print("L = 2*(", end="")
    for val in dists[0]:
        print(f"{val:.1f} + ", end="")
    before = 2 * np.sum(dists[0])
    print(f") = {before:.1f}")

    print_matrix(dists)

    print_matrix(win_m)
    
    routes, NKB, total_vol, total_milage, res, info_df = ClarkeRight(x, y, orderings=orderings, capacity=cap)
    
    pd.options.display.max_rows = 10000000
    # log_table.style.hide_index()
    # log_table.style.hide_index()
    display(info_df)
    # display(log_table.to_string(index=False))


    print("routes: ", routes)
    print("Res: ")
    display(res)
    print("NKB = ", end="")
    for i in NKB:
        print(f"{i:.2f} + ", end="")
    print(f"= {sum(NKB)}")
    print("Before: ", before)
    print("Total milage: ", total_milage)
    print("Total mil 2: ", before - sum(NKB))
    print("Total volume: ", total_vol)

    # plot_points(my_x, my_y, my_O, connect_radial=False, routes=routes)
    plot_points(test_x, test_y, test_O, connect_radial=False, routes=routes)



def main():
    solve_task()

if __name__ == "__main__":
    main()