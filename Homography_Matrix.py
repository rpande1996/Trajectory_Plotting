import numpy as np

def get_svd(mat_in):
    
    V_mat_in = np.matmul(mat_in.T, mat_in)
    U_mat_in = np.matmul(mat_in, mat_in.T)

    w_v, V = np.linalg.eig(V_mat_in)
    w_u, U = np.linalg.eig(U_mat_in)

    min_num = min(len(w_v), len(w_u))
    threshold = w_v > 1e-6
    w_v = w_v[threshold]
    if len(w_v) < min_num:
        w_v = np.append(w_v, np.zeros(min_num - len(w_v)))
    
    threshold = w_u > 1e-6
    w_u = w_u[threshold]
    if len(w_u) < min_num:
        w_u = np.append(w_u, np.zeros(min_num - len(w_u)))


    s_mat = np.zeros(mat_in.shape)
    np.fill_diagonal(s_mat, np.sqrt(w_u))

    return U, s_mat, V

A = np.zeros((8, 9))

in_mat = np.asarray([[5, 5, 100, 100], [150, 5, 200, 80], [150, 150, 220, 80], [5, 150, 100, 200]])

for i in range(in_mat.shape[0]):
    row = in_mat[i, :]
    x = row[0]
    y = row[1]
    xp = row[2]
    yp = row[3]

    row_i = np.asarray([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
    row_i_1 = np.asarray([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
    A[i, :] = row_i
    A[(i + 1), :] = row_i_1

U, S, V = get_svd(A)
i = np.argmin(S)

V = V.T
x = V[i, :]
H = np.reshape(x, (3, 3))
print(H)

