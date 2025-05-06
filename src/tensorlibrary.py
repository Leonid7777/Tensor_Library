import numpy as np
import scipy.linalg as lg
import scipy.sparse.linalg as slg

# RSVD
def rsvd(matrix, rank = 0):

    n, m = matrix.shape

    flag = 1

    if n > m:
        matrix = matrix.T
        n, m = m, n
        flag = 0

    if not rank and rank > min(m, n):
        rank = min(m, n)

    N = np.random.rand(m, rank * max(1, int(np.log10(m))))

    Q, _ = np.linalg.qr(matrix @ N)

    U, s, V = slg.svds(Q.T @ matrix, k=rank)

    U = Q @ U

    if flag:
        return U, s, V
    else:
        return V.T, s, U.T
    
# PQR
def PQR(mat, rank = 0, eps = 0):

    matrix = mat.copy()

    n, m = matrix.shape

    if not rank or rank > min(m, n):
        rank = min(m, n)

    Q = np.zeros((n, rank))
    R = np.zeros((rank, m))
    P = np.arange(m)

    tol = eps + 1
    i = 0

    while i < rank and tol > eps:

        norms = np.array([np.linalg.norm(matrix[:, j]) for j in range(i, m)])
        tol = np.sqrt(np.sum(norms**2))
        args = np.argsort(norms)[::-1]
        args += i

        P[[i, args[0]]] = P[[args[0], i]]
        R[:, [i, args[0]]]  = R[:, [args[0], i]]
        matrix[:, [i, args[0]]] = matrix[:, [args[0], i]]

        R[i, i] = norms[args[0] - i]
        Q[:, i] = matrix[:, i] / R[i, i]

        for j in range(i + 1, m):
            R[i, j] = matrix[:, j] @ Q[:, i]
            matrix[:, j] -= R[i, j] * Q[:, i]

        i += 1

    return Q, R, P, i

# MaxVol
def maxvol(mat, rank=0, eps=0, transpose=False, first_vec = False, return_cols=False, cols=False):

    mat = mat.copy()

    if transpose:
        mat = mat.T

    n, m = mat.shape

    if not rank or rank > min(n, m):
          rank = min(n, m)

    if first_vec:
        if not cols:
            print("Argument cols missed")
            return
        rows = np.arange(n)
        rows = np.delete(rows, cols)
        rows = np.append(cols, rows)
    else:
        rows = np.random.permutation(n)

    B = mat[rows[:rank], :]
    _, _, cols, rank = PQR(B, rank, eps)

    rev_rows = np.zeros(n, dtype=int)
    rev_cols = np.zeros(m, dtype=int)

    for i in range(n):
        rev_rows[rows[i]] = i
    for i in range(m):
        rev_cols[cols[i]] = i

    matrix = mat[np.ix_(rows, cols)]

    R = np.arange(rank)

    C = matrix[:, :rank]
    A = C[:rank, :]

    inv_A = np.linalg.pinv(A)

    prod = C @ inv_A

    max_val_ind = np.unravel_index(np.argmax(np.abs(prod), axis=None), prod.shape)

    iter = 0

    while np.abs(prod[max_val_ind]) > 1.000001 and max_val_ind[0] >= rank:

        iter += 1

        vec = prod[max_val_ind[0], :].copy()
        vec[max_val_ind[1]] -= 1
        vec /= prod[max_val_ind]

        prod -= prod[:, max_val_ind[1]].reshape((-1, 1)) @ vec.reshape((1, -1))

        R[max_val_ind[1]] = max_val_ind[0]

        max_val_ind = np.unravel_index(np.argmax(np.abs(prod), axis=None), prod.shape)

    prod = prod[rev_rows, :]
    R = matrix[np.ix_(R, rev_cols)]

    if return_cols:
        return cols
    else:
        return prod, R
    
# Cross Approximation
def cross_approx(matrix, rank = 0, eps = 0):

    n, m = matrix.shape

    if not rank and rank > min(m, n):
        rank = min(m, n)

    Q = np.zeros((n, rank))
    R = np.zeros((rank, m))

    i = 0
    err = eps + 1

    cols = np.zeros((n, rank + 1))
    ind_cols = np.ones(rank + 1, dtype=int) * m
    ind_cols[0] = np.random.randint(m)
    cols[:, 0] = matrix[:, ind_cols[0]]
    free_col = 1

    vec_cols = np.arange(m)

    rows = np.zeros((rank + 1, m))
    ind_rows = np.ones(rank + 1, dtype=int) * n
    ind_rows[0] = np.argmax(np.abs(cols[:, 0]))
    rows[0, :] = matrix[ind_rows[0], :]
    free_row = 1

    vec_rows = np.arange(n)

    while i < rank and err > eps:

        row_1, col_1 = np.unravel_index(np.argmax(np.abs(cols[:, :free_col])), cols[:, :free_col].shape)
        row_2, col_2 = np.unravel_index(np.argmax(np.abs(rows[:free_row, :])), rows[:free_row, :].shape)

        if row_1 == ind_rows[row_2] and col_2 == ind_cols[col_1]:

            Q[:, i] = cols[:, col_1] / cols[row_1, col_1]
            R[i, :] = rows[row_2, :]

            cols = np.delete(cols, col_1, 1)
            rows = np.delete(rows, row_2, 0)

            ind_cols = np.delete(ind_cols, col_1)
            ind_rows = np.delete(ind_rows, row_2)

            free_col -= 1
            free_row -= 1

            cols[:, :free_col] -= Q[:, i].reshape((-1, 1)) @ R[i, ind_cols[:free_col]].reshape((1, -1))
            rows[:free_row, :] -= Q[ind_rows[:free_row], i].reshape((-1, 1)) @ R[i, :].reshape((1, -1))

            ind = np.where(vec_cols == col_2)[0][0]
            vec_cols = np.delete(vec_cols, ind)

            ind = np.where(vec_rows == row_1)[0][0]
            vec_rows = np.delete(vec_rows, ind)

        elif np.abs(cols[row_1, col_1]) > np.abs(rows[row_2, col_2]):

            rows[free_row, :] = matrix[row_1, :] - Q[row_1, :i] @ R[:i, :]
            col = np.argmax(np.abs(rows[free_row, :]))

            if col == ind_cols[col_1]:

                Q[:, i] = cols[:, col_1] / cols[row_1, col_1]
                R[i, :] = rows[free_row, :]

                cols = np.delete(cols, col_1, 1)
                ind_cols = np.delete(ind_cols, col_1)
                free_col -= 1

                ind = np.where(vec_rows == row_1)[0][0]

            else:

                ind_rows[free_row] = row_1
                free_row += 1

                Q[:, i] = matrix[:, col] - Q[:, :i] @ R[:i, col]
                row = np.argmax(np.abs(Q[:, i]))

                R[i, :] =  (matrix[row, :] - Q[row, :i] @ R[:i, :]) / Q[row, i]

                ind = np.where(vec_rows == row)[0][0]

            vec_rows = np.delete(vec_rows, ind)

            cols[:, :free_col] -= Q[:, i].reshape((-1, 1)) @ R[i, ind_cols[:free_col]].reshape((1, -1))
            rows[:free_row, :] -= Q[ind_rows[:free_row], i].reshape((-1, 1)) @ R[i, :].reshape((1, -1))

            ind = np.where(vec_cols == col)[0][0]
            vec_cols = np.delete(vec_cols, ind)

        else:

            cols[:, free_col] = matrix[:, col_2] - Q[:, :i] @ R[:i, col_2]
            row = np.argmax(np.abs(cols[:, free_col]))

            if row == ind_rows[row_2]:

                R[i, :] = rows[row_2, :] / rows[row_2, col_2]
                Q[:, i] = cols[:, free_col]

                rows = np.delete(rows, row_2, 0)
                ind_rows = np.delete(ind_rows, row_2)
                free_row -= 1

                ind = np.where(vec_cols == col_2)[0][0]

            else:

                ind_cols[free_col] = col_2
                free_col += 1

                R[i, :] = matrix[row, :] - Q[row, :i] @ R[:i, :]
                col = np.argmax(np.abs(R[i, :]))

                Q[:, i] =  (matrix[:, col] - Q[:, :i] @ R[:i, col]) / R[i, col]

                ind = np.where(vec_cols == col)[0][0]

            vec_cols = np.delete(vec_cols, ind)

            cols[:, :free_col] -= Q[:, i].reshape((-1, 1)) @ R[i, ind_cols[:free_col]].reshape((1, -1))
            rows[:free_row, :] -= Q[ind_rows[:free_row], i].reshape((-1, 1)) @ R[i, :].reshape((1, -1))

            ind = np.where(vec_rows == row)[0][0]
            vec_rows = np.delete(vec_rows, ind)

        i += 1

        if not free_col:
            ind_cols[0] = vec_cols[np.random.randint(m - i)]
            cols[:, 0] = matrix[:, ind_cols[0]] - Q[:, :i] @ R[:i, ind_cols[0]]
            free_col = 1

        if not free_row:
            ind_rows[0] = vec_rows[np.random.randint(n - i)]
            rows[0, :] = matrix[ind_rows[0], :] - Q[ind_rows[0], :i] @ R[:i, :]
            free_row = 1

        err = np.linalg.norm(rows[:free_row, :])**2 + np.linalg.norm(cols[:, :free_col])**2 - np.linalg.norm(rows[:free_row, ind_cols[:free_col]])**2

        err = np.sqrt(err / (n * free_col + m * free_row - free_col * free_row) * (n - i) * (m - i))

    return Q, R, i

# ALS
def als(tensor, rank, seed=42, tol = 10**(-8), n_iteration = 1000):
    dimensions = tensor.shape
    len_dim = len(dimensions)

    rng = np.random.RandomState(seed)
    factors = [np.array(rng.normal(loc=0.0, scale=1.0, size=(dimensions[i], rank)), dtype=tensor.dtype) for i in range(len_dim)]

    eps = 1.0
    it = 0

    while eps > tol and it < n_iteration:

        eps = 0.0

        permutat = np.arange(len_dim)

        BB = [factors[len_dim - 1].T @ factors[len_dim - 1]]
        prom = np.array(tensor.reshape(-1, dimensions[len_dim - 1]) @ factors[len_dim - 1][:, 0])
        for r in range(1, rank):
            prom = np.vstack((prom, np.array(tensor.reshape(-1, dimensions[len_dim - 1]) @ factors[len_dim - 1][:, r])))
        BT = [prom.T]

        for i in range(1, len_dim - 1):
            BB.append((factors[len_dim - 1 - i].T @ factors[len_dim - 1 - i]) * BB[i - 1])

            if rank != 1:  
                prom = np.array(BT[i - 1][:, 0].reshape(-1, dimensions[len_dim - 1 - i]) @ factors[len_dim - 1 - i][:, 0])
            else:
                prom = np.array(BT[i - 1].reshape(-1, dimensions[len_dim - 1 - i])  @ factors[len_dim - 1 - i][:, 0])

            for r in range(1, rank):
                prom = np.vstack((prom, np.array(BT[i - 1][:, r].reshape(-1, dimensions[len_dim - 1 - i]) @ factors[len_dim - 1 - i][:, r])))

            BT.append(prom.T)

        ad_prod = np.ones((rank, rank))

        for i in range(len_dim):

            if i != 0 and i != len_dim - 1:
                BB_i = ad_prod * BB[len_dim - 2 - i]

                small_permute = np.array([i])
                small_permute = np.append(small_permute, np.arange(i))
                small_dim = np.array(dimensions[i])
                small_dim = np.append(small_dim, dimensions[:i])

                B_i = None
                for r in range(rank):

                    if rank != 1:
                        prom = np.transpose(BT[len_dim - 2 - i][:, r].reshape(dimensions[:(i + 1)]), small_permute)
                    else:
                        prom = np.transpose(BT[len_dim - 2 - i].reshape(dimensions[:(i + 1)]), small_permute)

                    for j in range(i - 1, -1, -1):
                        prom = prom.reshape(-1, dimensions[j]) @ factors[j][:, r]
                        prom = prom.reshape(small_dim[:(j + 1)])
                        small_permute[0] -= 1
                    if B_i is None:
                        B_i = prom
                    else:
                        B_i = np.vstack((B_i, prom))
                    small_permute[0] = i

                B_i = B_i.T

            elif i == 0:
                BB_i = BB[len_dim - 2]
                B_i = BT[len_dim - 2]
            else:
                BB_i = ad_prod

                small_permute = np.array([len_dim - 1])
                small_permute = np.append(small_permute, np.arange(len_dim - 1))
                small_dim = np.array(dimensions[len_dim - 1])
                small_dim = np.append(small_dim, dimensions[:len_dim - 1])

                B_i = None
                for r in range(rank):
                    small_permute[0] = len_dim - 1
                    prom = np.transpose(tensor, small_permute)
                    for j in range(len_dim - 1, 0, -1):
                        prom = prom.reshape(-1, small_dim[j]) @ factors[j - 1][:, r]
                        prom = prom.reshape(small_dim[:j])
                        small_permute[0] -= 1
                    if B_i is None:
                        B_i = prom
                    else:
                        B_i = np.vstack((B_i, prom))
                B_i = B_i.T

            if rank != 1:
                L = lg.cholesky(BB_i, lower=True)
                y = lg.solve_triangular(L, B_i.T, lower=True)
                new_factor = (lg.solve_triangular(L.T, y, lower=False)).T
            else:
                new_factor = (B_i.T / BB_i).T

            eps = max(eps, np.linalg.norm(new_factor - factors[i]) / np.linalg.norm(factors[i]))

            ad_prod *= new_factor.T @ new_factor

            factors[i] = new_factor

            permutat[i], permutat[-1] = permutat[-1], permutat[i]

        it += 1

    return factors, it

# Levenberg
def count_layers(len_dim, factors):
    layers = []
    layer = []
    for n in range(len_dim):
        layer.append(factors[n].T @ factors[n])
    layers.append(layer)

    for n in range(1, len_dim - 1):
        layer = []
        for r in range(len_dim - n):
            layer.append(layers[n - 1][r] * layers[0][r + n])
        layers.append(layer)
    return layers

def count_W(len_dim, layers):
    W = []
    for row in range(len_dim):
        start_layer = row - 1
        end_layer = len_dim - row - 2
        if start_layer >= 0:
            W_col = layers[start_layer][0]
            if end_layer >= 0:
                W_col = W_col * layers[end_layer][-1]
        else:
            W_col = layers[end_layer][-1]

        W_row = [W_col]

        for col in range(row + 1, len_dim):
            start_layer = row - 1
            middle_layer = col - row - 2
            end_layer = len_dim - col - 2

            if start_layer >= 0:
                W_col = layers[start_layer][0]
                if middle_layer >= 0:
                    W_col = W_col * layers[middle_layer][row + 1]
                if end_layer >= 0:
                    W_col = W_col * layers[end_layer][-1]
            elif middle_layer >= 0:
                W_col = layers[middle_layer][row + 1]
                if end_layer >= 0:
                    W_col = W_col * layers[end_layer][-1]
            else:
                W_col = layers[end_layer][-1]


            W_row.append(W_col)

        W.append(W_row)

    return W

def count_JJ_d(len_dim, W, pos, jacob_size, dimensions, rank, factors):
    JJ = np.zeros((jacob_size, jacob_size))
    for row in range(len_dim):
        BB = W[row][0]
        for r in range(dimensions[row]):
            now_pos = pos[row] + r * rank
            JJ[now_pos:now_pos+rank, now_pos:now_pos+rank] = BB

    for alpha in range(len_dim):
        now_alpha = pos[alpha]
        for i in range(alpha + 1, len_dim):
            now_i = pos[i]

            now_W = W[alpha][i - alpha]
            for betta in range(dimensions[alpha]):
                now_a_b = now_alpha + betta * rank
                for j in range(dimensions[i]):
                    now_i_j = now_i + j * rank
                    for gamma in range(rank):
                        now_a_b_g = now_a_b + gamma
                        for k in range(rank):
                            now_i_j_k = now_i_j + k
                            JJ[now_a_b_g, now_i_j_k] = now_W[gamma][k] * factors[i][j][gamma] * factors[alpha][betta][k]
                            JJ[now_i_j_k, now_a_b_g] = now_W[k][gamma] * factors[i][j][gamma] * factors[alpha][betta][k]
    return JJ

def calculate_JA(len_dim, factors, rank, dimensions, W, jacob_size):
    JA = np.zeros(jacob_size)
    for i in range(len_dim):
        BB = W[i][0]
        pos_i = int((np.sum(dimensions[:i])) * rank)
        for j in range(dimensions[i]):
            pos_j = pos_i + j * rank
            for k in range(rank):
                pos_k = pos_j + k
                val = 0
                for r in range(rank):
                    val += BB[k, r] * factors[i][j, r]
                JA[pos_k] = val

    return JA

def calculate_JT(len_dim, factors, rank, dimensions, jacob_size, tensor):
    JT = np.zeros(jacob_size)

    prom = np.array(tensor.reshape(-1, dimensions[len_dim - 1]) @ factors[len_dim - 1][:, 0])

    for r in range(1, rank):
        prom = np.vstack((prom, np.array(tensor.reshape(-1, dimensions[len_dim - 1]) @ factors[len_dim - 1][:, r])))
    BT = [prom.T]

    for i in range(1, len_dim - 1):

        if rank !=1:
            prom = np.array(BT[i - 1][:, 0].reshape(-1, dimensions[len_dim - 1 - i]) @ factors[len_dim - 1 - i][:, 0])
        else:
            prom = np.array(BT[i - 1].reshape(-1, dimensions[len_dim - 1 - i]) @ factors[len_dim - 1 - i][:, 0])

        for r in range(1, rank):
            prom = np.vstack((prom, np.array(BT[i - 1][:, r].reshape(-1, dimensions[len_dim - 1 - i]) @ factors[len_dim - 1 - i][:, r])))

        BT.append(prom.T)

    for i in range(len_dim):

        if i != 0 and i != len_dim - 1:

            small_permute = np.array([i])
            small_permute = np.append(small_permute, np.arange(i))
            small_dim = np.array(dimensions[i])
            small_dim = np.append(small_dim, dimensions[:i])

            B_i = None
            for r in range(rank):

                if rank != 1:
                    prom = np.transpose(BT[len_dim - 2 - i][:, r].reshape(dimensions[:(i + 1)]), small_permute)
                else:
                    prom = np.transpose(BT[len_dim - 2 - i].reshape(dimensions[:(i + 1)]), small_permute)

                for j in range(i - 1, -1, -1):
                    prom = prom.reshape(-1, dimensions[j]) @ factors[j][:, r]
                    prom = prom.reshape(small_dim[:(j + 1)])
                    small_permute[0] -= 1
                if B_i is None:
                    B_i = prom
                else:
                    B_i = np.vstack((B_i, prom))
                small_permute[0] = i

            if rank != 1:
                JT[int(np.sum(dimensions[:i]) * rank) : int(np.sum(dimensions[:(i+1)]) * rank)] = np.transpose(B_i, (1, 0)).reshape((-1,))
            else:
                JT[int(np.sum(dimensions[:i]) * rank) : int(np.sum(dimensions[:(i+1)]) * rank)] = B_i

        elif i == 0:
            B_i = BT[len_dim - 2]
            JT[:dimensions[0] * rank] = B_i.reshape((-1,))
        else:

            small_permute = np.array([len_dim - 1])
            small_permute = np.append(small_permute, np.arange(len_dim - 1))
            small_dim = np.array(dimensions[len_dim - 1])
            small_dim = np.append(small_dim, dimensions[:len_dim - 1])

            B_i = None
            for r in range(rank):
                small_permute[0] = len_dim - 1
                prom = np.transpose(tensor, small_permute)
                for j in range(len_dim - 1, 0, -1):
                    prom = prom.reshape(-1, small_dim[j]) @ factors[j - 1][:, r]
                    prom = prom.reshape(small_dim[:j])
                    small_permute[0] -= 1
                if B_i is None:
                    B_i = prom
                else:
                    B_i = np.vstack((B_i, prom))
            if rank != 1: 
                JT[-dimensions[-1] * rank :] = np.transpose(B_i, (1, 0)).reshape((-1,))
            else:
                JT[-dimensions[-1] * rank :] = B_i

    return JT

def levenberg(tensor, rank, seed=42, tol = 10**(-8), n_iteration = 100):

    dimensions = tensor.shape
    len_dim = len(dimensions)

    jacob_size = int(np.sum(dimensions) * rank)

    rng = np.random.RandomState(seed)
    factors = [np.array(rng.normal(loc=0.0, scale=1.0, size=(dimensions[i], rank)), dtype=tensor.dtype) for i in range(len_dim)]

    eps = 1.0
    it = 0

    pos = [0]
    for i in range(len_dim - 1):
        pos.append(pos[i] + dimensions[i] * rank)

    now_lambda = 0

    while eps > tol and it < n_iteration:

        layers = count_layers(len_dim, factors)
        W = count_W(len_dim, layers)
        JJ_d = count_JJ_d(len_dim, W, pos, jacob_size, dimensions, rank, factors)
        JA = calculate_JA(len_dim, factors, rank, dimensions, W, jacob_size)
        JT = calculate_JT(len_dim, factors, rank, dimensions, jacob_size, tensor)

        b = JA.reshape((-1,)) - JT.reshape((-1,))

        D_s = np.sqrt(np.diag(JJ_d))
        D_s_i = 1 / D_s


        b = D_s_i * b
        B = D_s_i.reshape((-1, 1)) * JJ_d * D_s_i
        T, U = lg.hessenberg(B, calc_q=True)

        b = U.T @ b

        mas = range(1, 3)

        min_lambda = None
        min_er = 0

        for alpha in mas:
            new_T = T + alpha * np.eye(jacob_size)
            y = lg.solve(new_T, b, assume_a='her')
            delta = U @ y
            delta = D_s_i * delta

            new_factors = []
            for i in range(len_dim):
                position = int(np.sum(dimensions[:i]) * rank)
                factor = np.zeros((dimensions[i], rank))
                for j in range(dimensions[i]):
                    factor[j, :] = factors[i][j, :] - (delta[position + j * rank:position + (j+1)*rank]).reshape(-1,)
                new_factors.append(factor)

            sweep_l = np.ones(rank).reshape(1, -1)
            for j in range(len_dim):
                sweep_l = lg.khatri_rao(sweep_l, new_factors[j])

            er = np.linalg.norm(np.sum(sweep_l, axis=1) - tensor.reshape(-1,)) / np.linalg.norm(tensor.reshape(-1,))
            if not min_lambda or er < min_er:
                min_lambda = alpha
                min_er = er

        now_lambda = min_lambda
        eps = min_er

        new_T = T + now_lambda * np.eye(jacob_size)
        y = lg.solve(new_T, b, assume_a='her')
        delta = U @ y
        delta = D_s_i * delta
        for i in range(len_dim):
            position = int(np.sum(dimensions[:i]) * rank)
            factor = np.array((dimensions[i], rank))
            for j in range(dimensions[i]):
                factors[i][j, :] -= (delta[position + j * rank:position + (j+1)*rank]).reshape(-1,)
        it += 1

    return factors, it

# CP to tensor
def cp_to_tensor(factors, sizes=None):
    len_dim = len(factors)
    rank = factors[0].shape[1]

    if sizes is None:
        sizes = [0] * len_dim
        for i in range(len_dim):
            sizes[i] = factors[i].shape[0]
    
    T = np.ones(rank).reshape(1, -1)
    for j in range(len_dim):
        T = lg.khatri_rao(T, factors[j])
    
    T = np.sum(T, axis=1)
    return T.reshape(sizes)


# choose rank
def use_rank(s, eps):
    k = 1
    while k < s.shape[0] and np.sum(s[k:]**2) >= eps:
        k = k + 1
    return k, np.sum(s[k:]**2)

# st-HOSVD
def st_HOSVD(T, eps=10**(-8), ranks=None):
    eps = (eps * np.linalg.norm(T.reshape(-1,)))**2
    factors = []
    dims = list(T.shape)
    d = len(dims)

    for i in range(d):
        permute = np.array([i])
        permute = np.append(permute, np.arange(i))
        permute = np.append(permute, np.arange(i+1, d))

        T = np.transpose(T, permute)
        T = T.reshape((dims[i], -1))

        U, s, Vh = np.linalg.svd(T, full_matrices=False)

        if ranks:
            rank = ranks[i]
        else:
            rank, m_eps = use_rank(s, eps / (d - i))
            eps = eps - m_eps

        factors.append(U[:, :rank])

        T = s[:rank].reshape((-1, 1)) * Vh[:rank, :]

        dims[i] = rank

        now_dim = np.array(dims[i]).astype(int)
        now_dim = np.append(now_dim, np.array(dims[:i])).astype(int)
        now_dim = np.append(now_dim, np.array(dims[i+1:])).astype(int)


        T = T.reshape(now_dim)

        permute = np.arange(1, i+1)
        permute = np.append(permute, 0)
        permute = np.append(permute, np.arange(i+1, d))

        T = np.transpose(T, permute)

    return T, factors, dims

# Tucker format to tensor 
def tucker_to_tensor(T, factors, ranks):
    tensor = T.copy()

    d = len(ranks)
    dims = np.zeros(d)

    for i in range(d):
        permute = np.array([i])
        permute = np.append(permute, np.arange(i))
        permute = np.append(permute, np.arange(i+1, d))

        tensor = np.transpose(tensor, permute)
        tensor = tensor.reshape((ranks[i], -1))

        tensor = factors[i] @ tensor

        dims[i] = factors[i].shape[0]

        now_dim = np.array(dims[i]).astype(int)
        now_dim = np.append(now_dim, np.array(dims[:i])).astype(int)
        now_dim = np.append(now_dim, np.array(ranks[i+1:])).astype(int)

        tensor = tensor.reshape(now_dim)

        permute = np.arange(1, i+1)
        permute = np.append(permute, 0)
        permute = np.append(permute, np.arange(i+1, d))

        tensor = np.transpose(tensor, permute)
    return tensor, dims 

# TT-SVD
def TT_SVD(T, eps=10**(-8), ranks=None):
    eps = (eps * np.linalg.norm(T.reshape(-1,)))**2

    factors = []
    dims = list(T.shape)
    d = len(dims)

    if ranks and len(ranks) != d - 1:
        print(f"Ranks must have {d - 1} elements")
        return 0

    T = T.reshape(dims[0], -1)

    for i in range(d - 1):

        U, s, Vh = np.linalg.svd(T, full_matrices=False)

        if ranks:
            rank = ranks[i]
        else:
            rank, m_eps = use_rank(s, eps / (d - i))
            eps = eps - m_eps

        T = s[:rank].reshape((-1, 1)) * Vh[:rank, :]

        if i == 0:
            factors.append(U[:, :rank])
            T = T.reshape(rank * dims[i + 1], -1)
        elif i != d - 2:
            factors.append(U[:, :rank].reshape(dims[i - 1], dims[i], rank))
            T = T.reshape(rank * dims[i + 1], -1)
        else:
            factors.append(U[:, :rank].reshape(dims[i - 1], dims[i], rank))
            factors.append(T)

        dims[i] = rank

    return factors, dims[:-1]

# TT-orthogonalization
def TT_orthogonalization(factors, ranks):
    d = len(factors)
    new_ranks = []

    for i in range(d):

        if not i:
            factors[i], R = np.linalg.qr(factors[i])
            new_ranks.append(R.shape[0])
        elif i != d - 1:
            timeless_factor = R @ factors[i].reshape(ranks[i - 1], -1)
            timeless_factor = timeless_factor.reshape(ranks[i - 1] * factors[i].shape[1], -1)
            timeless_factor, R = np.linalg.qr(timeless_factor)
            new_ranks.append(R.shape[0])
            factors[i] = timeless_factor.reshape(new_ranks[-2], factors[i].shape[1], new_ranks[-1])
        else:
            factors[i] = R @ factors[i]

    return factors, new_ranks

# TT compression
def TT_compression(factors, ranks, eps=10**(-8), new_ranks=None, is_ort=False):
    if not is_ort:
        factors, ranks = TT_orthogonalization(factors, ranks)

    norm_tensor = np.linalg.norm(factors[-1].reshape(-1,))

    eps = (eps * norm_tensor)**2
    d = len(factors)

    for i in range(d - 1, -1, -1):
        if i == d - 1:
            U, s, Vh = np.linalg.svd(factors[i], full_matrices=False)

            if new_ranks:
                rank = new_ranks[i - 1]
            else:
                rank, m_eps = use_rank(s, eps / i)
                eps = eps - m_eps

            factors[i] = Vh[:rank, :]
            Z = U[:, :rank] * s[:rank]

            ranks[i - 1] = rank

        elif i != 0:
            timeless_factor = factors[i].reshape(ranks[i - 1] * factors[i].shape[1], -1) @ Z
            U, s, Vh = np.linalg.svd(timeless_factor.reshape(ranks[i - 1], -1), full_matrices=False)

            if new_ranks:
                rank = new_ranks[i - 1]
            else:
                rank, m_eps = use_rank(s, eps / i)
                eps = eps - m_eps

            factors[i] = Vh[:rank, :].reshape(rank, factors[i].shape[1], ranks[i])

            Z = U[:, :rank] * s[:rank]
            ranks[i - 1] = rank
        else:
            factors[i] = factors[i] @ Z


    return factors, ranks

# TT format to tensor
def TT_to_tensor(factors, ranks):
    d = len(factors)

    T = factors[0]
    dims = [factors[0].shape[0]]

    for i in range(1, d):
        if i != d - 1:
            dims.append(factors[i].shape[1])
            T = T @ factors[i].reshape(ranks[i - 1], -1)
            T = T.reshape(-1, ranks[i])
        else:
            dims.append(factors[i].shape[1])
            T = T @ factors[i].reshape(ranks[i - 1], -1)
            T = T.reshape(dims)

    return T, dims
