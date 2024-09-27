import numpy as np
import copy
import random


def fx(a, b, c):
    substi1 = (a - 1) ** 2 + (b - 2) ** 2 + (c - 4) ** 2
    return substi1


def restriction(a1, b1, c1):  # 假设其有界闭箱是666，在真实fermat点求法中，有界闭箱是所有数据点的上界到下界
    if b1 + c1 - a1 <= 3.5 and a1 + b1 - c1 <= 6 and 6 >= a1 >= 0 and 6 >= b1 >= 0 and 6 >= c1 >= 0:
        res = 1
    else:
        res = 0

    return res


def search_grow_point(base1, step): # 不是这一段怎么就把base改了？？？？我不明白4
    base0 = copy.deepcopy(base1)
    total_growth = []
    for i in range(len(base1)):  # 此处是否要对每个生长点都进行判断可行性？这样是否会显著地降低性能？

        for j in range(len(base0)):
            base1[j] = base0[j]
        while restriction(base1[0], base1[1], base1[2]):
            total_growth.append([base1[0], base1[1], base1[2]])
            base1[i] += step

        for j in range(len(base0)):
            base1[j] = base0[j]
        while restriction(base1[0], base1[1], base1[2]):
            total_growth.append([base1[0], base1[1], base1[2]])
            base1[i] -= step

    list1 = []
    [list1.append(x) for x in total_growth if x not in list1]
    return list1, base0


def minfx(latest_list):  # 这一步的目的是为了收敛
    growth_score = []
    for growth_point in latest_list:
        score = fx(growth_point[0], growth_point[1], growth_point[2])
        growth_score.append(score)
    minIndex = np.array(growth_score).argsort()
    fmin_new = growth_score[minIndex[0]]

    xmin = latest_list[minIndex[0]]
    return xmin, fmin_new


def growth_probability(growth_list1, base2):  # 这一步的目的是为了下一步的生长
    fx0 = fx(base2[0], base2[1], base2[2])  # 求基底
    growth_list0 = copy.deepcopy(growth_list1)
    prob = []
    delta = 0
    length = len(growth_list1)
    for i in range(length):  # 这个地方如果不用一些方法来分化，该矩阵的长度将变得相当可怕
        now_growth_list = growth_list1[i]
        j = 0
        for growth_point in now_growth_list:
            growth_prob = fx(growth_point[0], growth_point[1], growth_point[2])
            if growth_prob >= fx0:
                prob.append([[i, j], 0])
            else:
                sub1 = fx0 - growth_prob
                prob.append([[i, j], sub1])
                delta += sub1
            j += 1
    for item in prob:
        if delta == 0:
            return base2, growth_list0
        else:
            item[1] /= delta
    #  到这一步为止，所有树干和树枝上面的生长概率已经全部计算完毕，接下来是生成新基点的步骤
    n = random.random()
    prob1 = 0
    x = 0
    while prob1 < n:
        prob1 += prob[x][1]
        x += 1
    new_base_index = prob[x-1][0]
    # print('new_base_index', new_base_index)
    # print('growth_list1',growth_list1)
    base_source = growth_list1[new_base_index[0]]
    # print('base_source)',base_source)
    new_base = base_source[new_base_index[1]]
    # print('new_base', new_base)
    return new_base, growth_list0


if __name__ == '__main__':
    lamda = 1
    base = [0, 0, 0]
    fmin = fx(base[0], base[1], base[2])
    K_times = 999
    #  以上为初始设置
    k = 1
    growth_list = []
    fmin_list = []
    answer = []
    fmin_list.append(fmin)
    repeat_condition = 0
    xmin = 0
    fmin_new = 0
    while k < K_times:
        print(k)
        new_list, base = search_grow_point(base, lamda)
        growth_list.append(new_list)
        xmin, fmin_new = minfx(new_list)
        base, growth_list = growth_probability(growth_list, base)
        print(base)
        fmin_list.append(fmin_new)
        if fmin_list[-1] == fmin_list[-2]:
            repeat_condition += 1
        if repeat_condition == 3:

            print(growth_list[0])
            base = random.choice(growth_list[0])
            answer.append([xmin, fmin_new])
        k += 1
    answer_list = []
    [answer_list.append(x) for x in answer if x not in answer_list]
    print(answer_list)
