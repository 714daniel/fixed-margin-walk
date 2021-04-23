from random import randint, sample, uniform
from copy import deepcopy 
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import re, threading
from multiprocessing import Pool

#UPPER_BOUND = 10
BURN_IN_TIME = 2500
SAMPLE_SIZE = 200
STEP_SIZE = 100
USE_INDEPENDENT_SAMPLES = True
K = 0
CONC_PROC = 8

def memoize(fn):
    """Creates a new version of fn that caches previous values."""
    return fn
    cache = { }
    def memoized_function(*args):
        cur_state = args[0]
        if type(cur_state) in [list,tuple, np.ndarray] and type(cur_state[0]) in [list,tuple, np.ndarray]:
            key = tuple(tuple(i) for i in cur_state)
            if key not in cache:
                cache[key] = fn(*args)
        else:
            key = tuple(args)
            if key not in cache:
                cache[key] = fn(*key)
            #else: 
            #    print('CACHE HIT FOR FN ', fn.__name__)
        return cache[key]
    return memoized_function

def build_valid_moves(UPPER_BOUND):
    """Builds a dictionary where each key is a 2x2 matrix [a,b],[c,d] encoded as string 'abcd', and each value is the list of all possible moves from that state.

    """
    move_list = dict()
    state_list = dict()
    for i1 in range(UPPER_BOUND):
        for i2 in range(UPPER_BOUND):
            for j1 in range(UPPER_BOUND):
                for j2 in range(UPPER_BOUND):
                    str_sum_of_row_cols = str(i1 + i2) + str(i1 + j1) + str(j1 + j2) + str(i2 + j2)
                    str_cells = str(i1) + str(i2) + str(j1) + str(j2)
                    if str_sum_of_row_cols not in move_list.keys():
                        move_list[str_sum_of_row_cols] = list()
                    move_list[str_sum_of_row_cols].append(str_cells)
                    if str_cells not in state_list:
                        state_list[str_cells] = move_list[str_sum_of_row_cols]
    return {state: [substate for substate in state_list[state] if state != substate] for state in state_list.keys() if len(state_list[state]) > 1}

@memoize
def get_2_by_2_score(a,b,c,d):
    return (abs((a-c) * (b-d))/ (a + b + c + d)) if (a + b + c + d) != 0 else 0
    #print(a,b,c,d)
    #print((abs(a + d - b - c) / (a + b + c + d)) if (a + b + c + d) != 0 else 0)
    #return (abs(a + d - b - c) / (a + b + c + d)) if (a + b + c + d) != 0 else 0

def get_ast_2_by_2_score(a,b,c,d):
    return ((a - c) ** 2 + (b - d) ** 2) / (a ** 2 + b ** 2 + c ** 2 + d ** 2) if (a+b+c+d) != 0 else 0

def get_ast_score(mat, K):
    score = 0
    for row1 in range(len(mat)):
        for row2 in range(row1 + 1, len(mat)):
            for col1 in range(len(mat[0])):
                for col2 in range(col1 + 1, len(mat[0])):
                    score += get_ast_2_by_2_score(mat[row1][col1], mat[row1][col2], mat[row2][col1], mat[row2][col2])
    return 4 * score / K

@memoize
def get_comp_score(mat, K):
    score = 0
    cu = 0
    for row1 in range(len(mat)):
        for row2 in range(row1 + 1, len(mat)):
            for col1 in range(len(mat[0])):
                for col2 in range(col1 + 1, len(mat[0])):
                    #print(row1, row2, col1, col2)
                    if (mat[row1][col1] == 1 and mat[row2][col2] == 1 and mat[row1][col2] == 0 and mat[row2][col1] == 0):
                        cu += 1
                        #print('checkerboard')
                    if (mat[row1][col1] == 0 and mat[row2][col2] == 0 and mat[row1][col2] == 1 and mat[row2][col1] == 1):
                        cu += 1
                        #print('checkerboard')
                    #print(abs(mat[row1][col1] + mat[row2][col2] - mat[row1][col2] - mat[row2][col1]))
                    #print(mat[row1][col1], mat[row2][col2],  mat[row1][col2], mat[row2][col1])
                    score += get_2_by_2_score(mat[row1][col1], mat[row1][col2], mat[row2][col1], mat[row2][col2])
                    #score += abs(mat[row1][col1] + mat[row2][col2] - mat[row1][col2] - mat[row2][col1])
    #print('CU: ', cu)
    #print('K: ', K)
    return score / K
    #return cu

@memoize
def count_2x2_degree(a,b,c,d):
    col_sum_1 = a + c
    col_sum_2 = b + d
    row_sum_1 = a + b 
    row_sum_2 = c + d
    return min(col_sum_1,col_sum_2,row_sum_1,row_sum_2)

@memoize
def count_degree_fast(mat):
    count = 0
    for row1 in range(len(mat)):
        for row2 in range(row1, len(mat)):
            if row1 != row2:
                for col1 in range(len(mat[0])):
                    for col2 in range(col1, len(mat[0])):
                        if col1 != col2:
                            count += count_2x2_degree(mat[row1][col1], mat[row1][col2], mat[row2][col1], mat[row2][col2])
                            #print('2STATE IS ',str(mat[row1][col1]) + str(mat[row1][col2])  + str(mat[row2][col1]) + str(mat[row2][col2]), ' DEGREE IS ',count_2x2_degree(mat[row1][col1], mat[row1][col2], mat[row2][col1], mat[row2][col2]))
                            #count += max(mat[row1][col1],mat[row1][col2],mat[row2][col1],mat[row2][col2]) - 1
    return count

def swap_new(old_mat, move_list, log = False):
    mat = deepcopy(old_mat)
    row_indexes = sample(list(range(0, len(mat))),2) 
    col_indexes = sample(list(range(0, len(mat[0]))),2) 
    i1, i2 = mat[row_indexes[0]][col_indexes[0]], mat[row_indexes[0]][col_indexes[1]]        #mat[0][0], mat[0], [1]
    j1,j2 = mat[row_indexes[1]][col_indexes[0]], mat[row_indexes[1]][col_indexes[1]]
    cur_state = str(i1) + str(i2) + str(j1) + str(j2)
    if cur_state not in move_list:
        return None
    selected_move_index = randint(0, len(move_list[cur_state]) - 1)
    new_move_str = move_list[cur_state][selected_move_index]
    new_val_i1, new_val_i2, new_val_j1, new_val_j2 = int(new_move_str[0]),int(new_move_str[1]), int(new_move_str[2]), int(new_move_str[3])
    mat[row_indexes[0]][col_indexes[0]] = new_val_i1
    mat[row_indexes[0]][col_indexes[1]] = new_val_i2
    mat[row_indexes[1]][col_indexes[0]] = new_val_j1
    mat[row_indexes[1]][col_indexes[1]] = new_val_j2
    return mat

def swap_no_precompute(old_mat):
    row_indexes = sample(list(range(0, len(old_mat))),2) 
    col_indexes = sample(list(range(0, len(old_mat[0]))),2) 
    i1, i2 = old_mat[row_indexes[0]][col_indexes[0]], old_mat[row_indexes[0]][col_indexes[1]]        #mat[0][0], mat[0], [1]
    j1,j2 = old_mat[row_indexes[1]][col_indexes[0]], old_mat[row_indexes[1]][col_indexes[1]]
    row_sum_1, row_sum_2, col_sum_1, col_sum_2 = i1 + i2, j1 + j2, i1 + j1, i2 + j2
    max_val = min(row_sum_1, col_sum_1)
    #print('MAX VAL IS ', max_val)
    random_i = randint(0, max_val)
    new_i1 = random_i
    new_i2 = row_sum_1 - new_i1
    new_j1 = col_sum_1 - new_i1
    new_j2 = row_sum_2 - new_j1
    if new_i1 < 0 or new_j1 < 0 or new_i2 < 0 or new_j2 < 0:
        return None

    mat = deepcopy(old_mat)    
    mat[row_indexes[0]][col_indexes[0]] = new_i1
    mat[row_indexes[0]][col_indexes[1]] = new_i2
    mat[row_indexes[1]][col_indexes[0]] = new_j1
    mat[row_indexes[1]][col_indexes[1]] = new_j2
    #print('CHANGED ',i1,i2, j1, j2, 'TO',new_i1, new_i2, new_j1, new_j2)
    return mat




@memoize
def count_degree(mat, move_list):
    count = 0
    for row1 in range(len(mat)):
        for row2 in range(row1, len(mat)):
            if row1 != row2:
                for col1 in range(len(mat[0])):
                    for col2 in range(col1, len(mat[0])):
                        if col1 != col2:
                            state = str(mat[row1][col1]) + str(mat[row1][col2])  + str(mat[row2][col1]) + str(mat[row2][col2])
                            if state in move_list:
                                count += len(move_list[state])
                                #print('STATE IS ', state, ' DEGREE IS ', len(move_list[state]))
                            #else: 
                            #    print(state)
    return count

def graph_scores(scores, nbins=10):
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(scores, nbins)
    ax.set(xlabel="Checkerboard sum", ylabel="Frequency", title="Distribution of checkerboard sums")
    #ax.scatter(scores, cu)
    fig.savefig('dist.png')
    plt.show()


def fast_random_walk(seed, n):
    matrices = list()
    cur_mat = seed
    i = 0
    while i < BURN_IN_TIME + n:
        swap_result = swap_no_precompute(cur_mat)
        if type(swap_result) != np.ndarray and swap_result == None:
            continue
        i += 1
        degree_old, degree_new = count_degree_fast(cur_mat), count_degree_fast(swap_result)
        if degree_old == 0:
            print(cur_mat)
            exit()
        if degree_new == 0:
            print(swap_result)
            exit()
        rand_number = uniform(0,1)
        if rand_number < min(1,degree_old / degree_new):
            cur_mat = swap_result
        if i > BURN_IN_TIME:
            matrices.append(cur_mat)
    return cur_mat if n == 1 else matrices[-1 * n:]



def random_walk(seed, move_list, n, log = False):
    matrices = list()
    cur_mat = seed
    i = 0
    while i < BURN_IN_TIME + n:
        #swap_result = swap_new(cur_mat, move_list, log)
        swap_result = swap_no_precompute(cur_mat)
        if type(swap_result) != np.ndarray and swap_result == None:
            continue
        i += 1
        degree_old, degree_new = count_degree_fast(cur_mat), count_degree_fast(swap_result)
        rand_number = uniform(0,1)
        if rand_number <= min(1,degree_old / degree_new):
            cur_mat = swap_result
        if i > BURN_IN_TIME:
            matrices.append(cur_mat)
    return cur_mat if n == 1 else matrices[-1 * n:]

def get_sample(seed, move_list):
    sample = list()
    if USE_INDEPENDENT_SAMPLES:
        n = 1
        for _ in range(SAMPLE_SIZE):
            if _ % 10 == 0:
                print(_)
            sample.append(random_walk(seed, move_list, n))
    else:
        sample = random_walk(seed, move_list, SAMPLE_SIZE)
    return sample

def get_scores(matrices, score_fn):
    mat = matrices[0]
    K = (len(mat) * (len(mat) - 1) * len(mat[0]) * (len(mat[0]) - 1))
    return [score_fn(m, K) for m in matrices]

def get_prop(score, matrices, score_fn):
    #print(matrices)
    scores = get_scores(matrices, score_fn)
    return len([s for s in scores if s >= score]) / len(scores)

def find_upper_bound(A):
    max_row = max([sum(A[i]) for i in range(len(A))])
    col_sums = [0] * len(A[0])
    for r in A:
        for i in range(len(r)):
            col_sums[i] += r[i]
    max_col = max(col_sums)
    return max(max_row, max_col)

def get_mat_list(filename = 'Matrices.txt', cast_to_int = True):
    matrices = dict()
    has_num_re = re.compile('\d')
    with open(filename, 'r') as f:
        cur_mat = []
        title = ''
        for line in f:
            if line.isspace():
                matrices[title] = cur_mat
                cur_mat = []
                title = ''
                continue
            if not has_num_re.match(line):
                if title == '':
                        title = line.split()[0][:-1]
                continue
            if cast_to_int:
                split_line_nums = [int(float(n)) for n in line.split()[1:]]
            else:
                split_line_nums = [float(n) for n in line.split()[1:]]
            cur_mat.append(split_line_nums)
    return matrices

def print_mat_dim(test_mat_list):
    for name,m in test_mat_list.items():
        if name == '':
            continue
        if type(m) == list:
                print(name)
                print(len(m),'x',len(m[0]))

def run_sim(test_mat, name):
    start = timer()
    print('RUNNING ',name)
    outfile = open('output_files/' + name + '.txt', 'w')
    K = (len(test_mat) * (len(test_mat) - 1) * len(test_mat[0]) * (len(test_mat[0]) - 1))
    outfile.write('\nDIMENSIONS: ' + str(len(test_mat)) + 'x' + str(len(test_mat[0])))
    outfile.write('\nTEST SCORE CS: ' + str(get_comp_score(test_mat, K)))
    outfile.write('\nTEST SCORE AST: ' + str(get_ast_score(test_mat, K)))
    sample = get_sample(test_mat, [])
    scores = get_scores(sample, get_comp_score)
    print(sorted(scores))
    #p = [get_prop(s, sample, get_comp_score) for s in scores]
    #outfile.write(sorted(p))
    #graph_scores(scores)
    
    p = get_prop(get_comp_score(test_mat, K), sample, get_comp_score)
    outfile.write('\nPROPORTION CS: ' +  str(p))
    p = get_prop(get_ast_score(test_mat, K), sample, get_ast_score)
    outfile.write('\nPROPORTION AST: ' + str(p))
    end = timer()
    outfile.write('\nTOOK ' + str(end - start) +' SECONDS')
    outfile.close()
    print('FINISHED ', name)

def run():
    test_mat_list = get_mat_list()
    mat_names = sorted(test_mat_list.keys())
    print(mat_names)
    start_name = ''
    if start_name == '':
        start_name = mat_names[0]
    mat_names = mat_names[mat_names.index(start_name):]
    mat_list = set()
    args = list()
    count = 0
    for m in mat_names:
        if m == '':
            continue
        count += 1
        args.append(tuple([test_mat_list[m],m]))
    start = timer()
    #while len(args) > 0:
    #    cur_args = list()
    #    for _ in range(10):
    #        cur_args.append(args.pop())
    #    with Pool(10) as p:
    #        #print(p)
    #        p.starmap(run_sim, cur_args)
    with Pool() as p:
        p.starmap(run_sim, args)
    end = timer()
    print('TOOK ', end - start, ' SECONDS')

def run_single():
    test_mat_list = get_mat_list()
    run_sim(test_mat_list['Egg83'], 'Egg83')

if __name__ == "__main__":
    #run()
    run_single()