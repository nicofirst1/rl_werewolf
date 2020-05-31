from math import floor, sqrt

from tqdm import tqdm

from evaluation.utils.theo_winWw import save_results


def outcome(ww,vill,prob,res):
    vill -= 1

    if ww == 0 or vill ==ww or vill<=0:
        if f"{ww},{vill}" not in res.keys():
            res[f"{ww},{vill}"]=[]

        res[f"{ww},{vill}"].append(prob)
        return res



    np=ww+vill
    p_ww=ww/np*prob
    p_vill=vill/np*prob

    n_ww=ww
    n_vill=vill

    try:
        res = outcome(n_ww - 1, n_vill, p_ww, res)
        res = outcome(n_ww, n_vill - 1, p_vill, res)
    except RecursionError:
        a=1

    return res

def experiment(n_player):

    ww = floor(sqrt(n_player))
    vill = n_player - ww

    res = outcome(ww, vill, 1, {})
    # tot_leaves = sum(len(elem) for elem in res.values())
    # leaves = {k: len(v) for k, v in res.items()}
    res = {k: sum(v) for k, v in res.items()}
    p_win_ww = sum([v for k, v in res.items() if k[0] != '0'])
    return p_win_ww

if __name__ == '__main__':

    rng=range(5,102)

    rows=[]
    for idx in tqdm(rng):
        res=experiment(idx)
        rows.append((idx,res))

        save_results(list(rows),"outcomes.csv")



