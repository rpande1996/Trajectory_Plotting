import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

def ols_parabola(X, Y):
    xmax = max(X)
    ymax = max(Y)
    if xmax > ymax:
        maxls = xmax
    else:
        maxls = ymax
    xact = np.asarray([i/maxls for i in X])
    yact = np.asarray([i/maxls for i in Y])

    xact = xact + 1
    yact = yact + 1
    sum_4 = np.sum(np.power(xact, 4))
    sum_3 = np.sum(np.power(xact, 3))
    sum_2 = np.sum(np.power(xact, 2))
    sum_1 = np.sum(np.power(xact, 1))
    sum_0 = np.sum(np.power(xact, 0))
    sum_x_y = np.sum(xact * yact)
    sum_x2_y = np.sum(np.power(xact, 2) * yact)
    sum_y = np.sum(yact)

    lhs_mat = np.asarray([[sum_4, sum_3, sum_2], [sum_3, sum_2, sum_1], [sum_2, sum_1, sum_0]])
    rhs_mat = np.asarray([[sum_x2_y, sum_x_y, sum_y]])
    Rhs_mat = rhs_mat.T
    weight_mat = np.matmul(np.linalg.pinv(lhs_mat), Rhs_mat)
    a_new = weight_mat[0]/maxls
    b_new = 2*weight_mat[0] + weight_mat[1]
    c_new = (weight_mat[0] + weight_mat[1] + weight_mat[2] - 1)*maxls

    new_wmat = [a_new, b_new, c_new]

    new_wmat = np.asarray(new_wmat)

    return new_wmat

def tls_parabola(X, Y):
    Xtls = np.asarray([np.power(X, 2), X, np.ones(len(X)), -Y]).T
    U, s, V = la.svd(Xtls)
    V = V.T
    i = np.argmin(s)
    B = V[:, i]
    B = B[0:3] / B[3]
    y_tls_out = np.matmul(Xtls[:, 0:3], B)
    return y_tls_out


def fit(X, Y, max_prob=0.99, sample_points=3, tol_dist=0.01):

    weight = np.asarray([1 for _ in range(3)])
    if not isinstance(max_prob, float):
        try:
            max_prob = float(max_prob)
        except:
            raise TypeError("max_prob need to be float, or type convertible to float")

    random_sampler = [1 for _ in range(sample_points)] + [0 for _ in range(X.shape[0] - sample_points)]
    random_sampler = np.asarray(random_sampler).astype(np.int)
    in_points = 0
    out_points = 0
    N = 1e6
    sample_count = 0
    while sample_count < N:
        np.random.shuffle(random_sampler)
        sample_X = X[random_sampler == 1]
        sample_Y = Y[random_sampler == 1]

        X_mat = [[item ** 2, item, 1] for item in sample_X]
        X_mat = np.asarray(X_mat)
        weight = np.linalg.pinv(X_mat) @ sample_Y

        test_X = X
        test_Y = Y

        _, y_dist = score(test_X, test_Y, weight)
        in_points = len(y_dist < tol_dist)
        out_points = len(y_dist >= tol_dist)
        e = 1 - (in_points / (in_points + out_points))
        N = np.log(1 - max_prob) / np.log(1 - (1 - e) ** sample_points)
        sample_count += 1

    return weight


def dist(y_out, y_actual):

    return np.sqrt(np.power(y_out - y_actual, 2))

def score(X, y_actual, weight):
    X = np.asarray([np.power(X, 2), X, np.ones(X.shape[0])])
    y_out = np.matmul(X.T, weight)
    y_dist = dist(y_out, y_actual)
    return y_out, y_dist

def get_center(frame):
    val_dict_x = {}
    val_dict_y = {}
    for i in range(frame.shape[1]):
        if np.any(frame[:, i] < 200):
            low_val_y = 0
            high_val_y = 0
            for j in range(frame.shape[0]):
                if frame[j, i] < 200 and low_val_y == 0:
                    low_val_y = j
                elif low_val_y != 0 and j > high_val_y:
                    high_val_y = j

        try:
            val_dict_y[i] = (low_val_y, high_val_y)
        except NameError:
            continue

    maxy = 0
    for keyy in val_dict_y:
        if val_dict_y[keyy][1] - val_dict_y[keyy][0] > maxy:
            maxy = val_dict_y[keyy][1] - val_dict_y[keyy][0]

    max_key_y = keyy
    top_point, bottom_point = val_dict_y[max_key_y]
    for a in range(frame.shape[0]):
        if np.any(frame[a, :] < 200):
            low_val_x = 0
            high_val_x = 0
            for b in range(frame.shape[1]):
                if frame[a, b] < 200 and low_val_x == 0:
                    low_val_x = b
                elif low_val_x != 0 and b > high_val_x:
                    high_val_x = b
        try:
            val_dict_x[a] = (low_val_x, high_val_x)
        except NameError:
            continue

    maxx = 0
    for keyx in val_dict_x:
        if val_dict_x[keyx][1] - val_dict_x[keyx][0] > maxx:
            maxx = val_dict_x[keyx][1] - val_dict_x[keyx][0]
    max_key_x = keyx
    left_point, right_point = val_dict_x[max_key_x]
    xcoor = int((right_point + left_point) / 2)
    ycoor = int((top_point + bottom_point) / 2)
    return xcoor, frame.shape[1] - ycoor

sel = int(input("Enter video selection: "))

if sel == 1:
    cap = cv2.VideoCapture('../media/input/Ball_travel_10fps.mp4')
elif sel == 2:
    cap = cv2.VideoCapture('../media/input/Ball_travel_2_updated.mp4')
else:
    print("Invalid selection")
    exit(0)
ycoor = []
xcoor = []
lost = []
while (cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if sel == 1:
        frame = cv2.resize(frame, (1200, 838))
    elif sel == 2:
        frame = cv2.resize(frame, (1320, 933))
    else:
        print("Invalid selection")
        exit(0)
    try:
        cv2.imshow('Frame', frame)
    except TypeError:
        continue
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    avg_x, avg_y = get_center(frame)

    xcoor.append(avg_x)
    ycoor.append(avg_y)
for d in range(len(lost)):
    try:
        lost[d][int(ycoor[d]), int(xcoor[d]), 0] = 0
        lost[d][int(ycoor[d]), int(xcoor[d]), 1] = 0
        lost[d][int(ycoor[d]), int(xcoor[d]), 2] = 0
        cv2.imshow('frame', lost[d])
    except TypeError:
        continue

x3 = [1 for _ in range(len(xcoor))]
sq = [z ** 2 for z in xcoor]
x = np.asarray([sq, xcoor, x3])
y = np.asarray([ycoor])
X = np.transpose(x)
Y = np.transpose(y)

new_wmat = ols_parabola(xcoor, ycoor)
Yls = np.matmul(X, new_wmat)

Rx = np.asarray(xcoor)
Ry = np.asarray(ycoor)

weights = fit(Rx, Ry)
y_out, dist = score(Rx, Ry, weights)

y_tls_out = tls_parabola(Rx, Ry)

plt.scatter(xcoor, ycoor, color="black")
plt.plot(xcoor, Yls, color="red")
plt.plot(xcoor, y_out, color="green")
plt.plot(xcoor, y_tls_out, color="orange")
plt.legend(['Total Least Squares', 'RANSAC', 'Ordinary Least Squares', 'Raw Data'])
plt.savefig(f"../media/output/Plot_for_video{sel}.jpg")
plt.show()