import csv
from functools import reduce

if __name__ == '__main__':
    f1 = "/home/dizzi/Desktop/rl-werewolf/Resources/CSVs/practical_ww_unite.csv"
    f2 = "/home/dizzi/Desktop/rl-werewolf/Resources/CSVs/practical_ww_revenge.csv"

    with open(f1, "r+") as f:
        c = csv.reader(f, delimiter=',')
        rows1 = list(c)

    with open(f2, "r+") as f:
        c = csv.reader(f, delimiter=',')
        rows2 = list(c)

    h1 = rows1[0]
    h2 = rows2[0]

    idx1 = h1.index("mean")
    idx2 = h2.index("mean")

    res_dict = {float(k[0]): [float(k[idx1])] for k in rows1[1:]}

    for r in rows2[1:]:
        k = float(r[0])
        if k in res_dict.keys():
            res_dict[k] += [float(r[idx2])]

    res_dict = {k: v for k, v in res_dict.items() if len(v) > 1}
    res_dict = [reduce(lambda x, y: x - y, v) ** 2 for v in res_dict.values()]
    mse = reduce(lambda x, y: x + y, res_dict) / len(res_dict)

    print(f"MSE between two csv is {mse} on {len(res_dict)} elements")
