from random import randint, sample


UPPER_BOUND = 5
def gen_all_mat(m=3, n = 3, r=3, c=3):
    matrices = set()
    
    for rep in range(10000):
        curMat = [None] * n
        for rows in range(n):
            row = None
            if randint(0,1) == 0:
                row = 1,1,1,
            else:
                row = tuple(sample([0,1,2], n))
                #print(row)
            curMat[rows] = row
        #print(curMat)
        #exit()
        valid = True
        for row in range(n):
            if curMat[0][row] + curMat[1][row] + curMat[2][row] != c:
                valid = False
        if valid:
            matrices.add(tuple(curMat))
    return matrices

def build_valid_moves():
    move_list = dict()
    state_list = dict()
    counter = set()
    for i1 in range(UPPER_BOUND):
        for i2 in range(UPPER_BOUND):
            for j1 in range(UPPER_BOUND):
                for j2 in range(UPPER_BOUND):
                    str_sum_of_row_cols = str(i1 + i2) + str(i1 + j1) + str(j1 + j2) + str(i2 + j2)
                    str_cells = str(i1) + str(i2) + str(j1) + str(j2)
                    if str_sum_of_row_cols not in move_list.keys():
                        move_list[str_sum_of_row_cols] = list()
                    move_list[str_sum_of_row_cols].append(str_cells)
                    counter.add(str_cells)
                    if str_cells not in state_list:
                        state_list[str_cells] = move_list[str_sum_of_row_cols]
    return {state: [substate for substate in state_list[state] if state != substate] for state in state_list.keys() if len(state_list[state]) > 1}
    #print([state for state in state_list.keys() if len(state_list[state]) > 1])
    #return [list(state_list[state]) for state in state_list.items() if state_list[state] > 1]

    #print(len(move_list))
    #print(move_list)





def swap_new(mat, move_list, log = False):
    row_indexes = sample(list(range(0, len(mat))),2) 
    col_indexes = sample(list(range(0, len(mat[0]))),2) 
    i1, i2 = mat[row_indexes[0]][col_indexes[0]], mat[row_indexes[0]][col_indexes[1]]        #mat[0][0], mat[0], [1]
    j1,j2 = mat[row_indexes[1]][col_indexes[0]], mat[row_indexes[1]][col_indexes[1]]
    cur_state = str(i1) + str(i2) + str(j1) + str(j2)
    print(cur_state)
    if cur_state not in move_list:
        return mat
    selected_move_index = randint(0, len(move_list[cur_state]) - 1)
    new_move_str = move_list[cur_state][selected_move_index]
    new_val_i1, new_val_i2, new_val_j1, new_val_j2 = int(new_move_str[0]),int(new_move_str[1]), int(new_move_str[2]), int(new_move_str[3])
    mat[row_indexes[0]][col_indexes[0]] = new_val_i1
    mat[row_indexes[0]][col_indexes[1]] = new_val_i2
    mat[row_indexes[1]][col_indexes[0]] = new_val_j1
    mat[row_indexes[1]][col_indexes[1]] = new_val_j2
    return mat




def swap(mat, log = False):
    """selects two random rows and cols. Tests if the swap on these is valid, swaps if it is. """
    #sample two rows/cols no replacement
    row_indexes = sample(list(range(0, len(mat))),2) 
    col_indexes = sample(list(range(0, len(mat[0]))),2) 
    i1, i2 = mat[row_indexes[0]][col_indexes[0]], mat[row_indexes[0]][col_indexes[1]]        #mat[0][0], mat[0], [1]
    j1,j2 = mat[row_indexes[1]][col_indexes[0]], mat[row_indexes[1]][col_indexes[1]]
    
    if log:
            print('VALS', i1, i2, j1, j2)
            print('INDEXES',row_indexes, col_indexes)
            print(mat)
    #We have either (0,0),(0,0), or (1,1),(1,1). If it is (1,1),(1,1), we can swap to (2,0),(0,2)
    if i1 == j1 and i2 == j2 and i1 == i2:
        if i1 + j1 < UPPER_BOUND:
            mat[row_indexes[0]][col_indexes[0]] = i1 + j1
            mat[row_indexes[1]][col_indexes[1]] = i1 + j1
            mat[row_indexes[0]][col_indexes[1]] -= j1
            mat[row_indexes[1]][col_indexes[0]] -= j2
        return mat

    #We have (a,b),(c,d) with a+b = c+d. In this case, swap (a,b) and (c,d). This could be (0,1) and (1,0), or (1,1) and (2,0), or (2,1) and (1,2), or (2,0) and (0,2)
    if i1 + i2 == j1 + j2:
        if log:
            print(i1, i2, j1, j2)
            print(row_indexes, col_indexes)
            print(mat)
        if(i1 == 2 and i2 == 0 and j1 == 0 and j2 == 2):
            mat[row_indexes[0]][col_indexes[0]] = 1
            mat[row_indexes[1]][col_indexes[1]] = 1
            mat[row_indexes[0]][col_indexes[1]] = 1
            mat[row_indexes[1]][col_indexes[0]] = 1
        else:
            mat[row_indexes[0]][col_indexes[0]],mat[row_indexes[1]][col_indexes[0]]  = mat[row_indexes[1]][col_indexes[0]], mat[row_indexes[0]][col_indexes[0]]
            mat[row_indexes[0]][col_indexes[1]], mat[row_indexes[1]][col_indexes[1]] = mat[row_indexes[1]][col_indexes[1]], mat[row_indexes[0]][col_indexes[1]]

    #We have (a,b),(c,d) with a+b = c+d. In this case, swap (a,b) and (c,d). This could be (0,1) and (1,0), or (1,1) and (2,0), or (2,1) and (1,2), or (2,0) and (0,2)
    elif i1 + j1 == i2 + j2:
        if(i1 == 2 and i2 == 0 and j1 == 0 and j2 == 2):
            mat[row_indexes[0]][col_indexes[0]] = 1
            mat[row_indexes[1]][col_indexes[1]] = 1
            mat[row_indexes[0]][col_indexes[1]] = 1
            mat[row_indexes[1]][col_indexes[0]] = 1
        mat[row_indexes[0]][col_indexes[0]],mat[row_indexes[0]][col_indexes[1]]  = mat[row_indexes[0]][col_indexes[1]], mat[row_indexes[0]][col_indexes[0]]
        mat[row_indexes[1]][col_indexes[0]], mat[row_indexes[1]][col_indexes[1]] = mat[row_indexes[1]][col_indexes[1]], mat[row_indexes[1]][col_indexes[0]]
            

def test_swap_mat(seed, move_list, log = False):
    matrices = set()
    counter = {0:0, 1:0, 2:0}
    print('SEED: ',seed)
    for i in range(10000):
        #count = sum([row.count(2) for row in seed])
        #counter[count] += 1
        #print(counter)
        swap_new(seed, move_list, log)
        matrices.add(tuple(tuple(i) for i in seed))
    [print(m) for m in matrices]
    return matrices


#all_mat = gen_all_mat()
#while len(res) < 31:
#    res = gen_all_mat()
#res2 = test_swap_mat(sample([list([list(r) for r in l]) for l in res], 1)[0])
#print(list(range(0,5)))
#res2 = test_swap_mat([[1,1,1],[1,1,1],[1,1,1]])
#diff = res - res2
#res3 = test_swap_mat(list([list(r) for r in list(diff)[2]]), True)
#print(len(res3))
#[print(r) for r in res3]
moves = build_valid_moves()
#print(moves)
res = test_swap_mat([[2,2,2],[0,1,0],[0,0,0]], moves, True)
#print(len(res))
#sprint(len(all_mat - res))

