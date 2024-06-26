import multiprocessing
from functools import partial
from itertools import chain
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions as tdist
from einops import rearrange, reduce, repeat
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance
from torch import BoolTensor, FloatTensor

Feats = Union[FloatTensor, List[FloatTensor]]
Layout = Tuple[np.ndarray, np.ndarray]

# set True to disable parallel computing by multiprocessing (typically for debug)
# DISABLED = False
DISABLED = True

def convert_xywh_to_ltrb(bbox: Union[np.ndarray, FloatTensor]):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]

def compute_iou(
    box_1: Union[np.ndarray, FloatTensor],
    box_2: Union[np.ndarray, FloatTensor],
    generalized: bool = False,
) -> Union[np.ndarray, FloatTensor]:
    # box_1: [N, 4]  box_2: [N, 4]

    if isinstance(box_1, np.ndarray):
        lib = np
    elif isinstance(box_1, FloatTensor):
        lib = torch
    else:
        raise NotImplementedError(type(box_1))

    l1, t1, r1, b1 = convert_xywh_to_ltrb(box_1.T)
    l2, t2, r2, b2 = convert_xywh_to_ltrb(box_2.T)
    a1, a2 = (r1 - l1) * (b1 - t1), (r2 - l2) * (b2 - t2)

    # intersection
    l_max = lib.maximum(l1, l2)
    r_min = lib.minimum(r1, r2)
    t_max = lib.maximum(t1, t2)
    b_min = lib.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = lib.where(cond, (r_min - l_max) * (b_min - t_max), lib.zeros_like(a1[0]))

    au = a1 + a2 - ai
    iou = ai / au

    if not generalized:
        return iou

    # outer region
    l_min = lib.minimum(l1, l2)
    r_max = lib.maximum(r1, r2)
    t_min = lib.minimum(t1, t2)
    b_max = lib.maximum(b1, b2)
    ac = (r_max - l_min) * (b_max - t_min)

    giou = iou - (ac - au) / ac

    return giou

def __compute_maximum_iou_for_layout(layout_1: Layout, layout_2: Layout) -> float:
    score = 0.0
    (bi, li), (bj, lj) = layout_1, layout_2
    N = len(bi)
    for l in list(set(li.tolist())):
        _bi = bi[np.where(li == l)]
        _bj = bj[np.where(lj == l)]
        n = len(_bi)
        ii, jj = np.meshgrid(range(n), range(n))
        ii, jj = ii.flatten(), jj.flatten()
        iou = compute_iou(_bi[ii], _bj[jj]).reshape(n, n)
        # note: maximize is supported only when scipy >= 1.4
        ii, jj = linear_sum_assignment(iou, maximize=True)
        score += iou[ii, jj].sum().item()
    return score / N


def __compute_maximum_iou(layouts_1_and_2: Tuple[List[Layout]]) -> np.ndarray:
    layouts_1, layouts_2 = layouts_1_and_2
    N, M = len(layouts_1), len(layouts_2)
    ii, jj = np.meshgrid(range(N), range(M))
    ii, jj = ii.flatten(), jj.flatten()
    scores = np.asarray(
        [
            __compute_maximum_iou_for_layout(layouts_1[i], layouts_2[j])
            for i, j in zip(ii, jj)
        ]
    ).reshape(N, M)
    ii, jj = linear_sum_assignment(scores, maximize=True)
    return scores[ii, jj]


def __get_cond2layouts(layout_list: List[Layout]) -> Dict[str, List[Layout]]:
    out = dict()
    for bs, ls in layout_list:
        cond_key = str(sorted(ls.tolist()))
        if cond_key not in out.keys():
            out[cond_key] = [(bs, ls)]
        else:
            out[cond_key].append((bs, ls))
    return out


def compute_maximum_iou(
    layouts_1: List[Layout],
    layouts_2: List[Layout],
    disable_parallel: bool = DISABLED,
    n_jobs: Optional[int] = None,
):
    """
    Computes Maximum IoU [Kikuchi+, ACMMM'21]
    """
    c2bl_1 = __get_cond2layouts(layouts_1)
    keys_1 = set(c2bl_1.keys())
    c2bl_2 = __get_cond2layouts(layouts_2)
    keys_2 = set(c2bl_2.keys())
    keys = list(keys_1.intersection(keys_2))
    args = [(c2bl_1[key], c2bl_2[key]) for key in keys]
    # to check actual number of layouts for evaluation
    # ans = 0
    # for x in args:
    #     ans += len(x[0])
    if disable_parallel:
        scores = [__compute_maximum_iou(a) for a in args]
    else:
        with multiprocessing.Pool(n_jobs) as p:
            scores = p.map(__compute_maximum_iou, args)
    scores = np.asarray(list(chain.from_iterable(scores)))
    if len(scores) == 0:
        return 0.0
    else:
        return scores.mean().item()
    
def preprocess_layouts(layouts, types):
    processed_layouts = []
    for layout, type_ in zip(layouts, types):
        # PyTorch 텐서를 CPU로 이동시키고 NumPy 배열로 변환합니다.
        layout_cpu = layout.cpu().numpy() if layout.is_cuda else layout.numpy()
        type_cpu = type_.cpu().numpy() if type_.is_cuda else type_.numpy()
        
        # type이 0이 아닌 요소의 인덱스를 찾습니다.
        valid_indices = np.where(type_cpu != 0)[0]
        # type이 0이 아닌 요소에 해당하는 valid의 요소만을 선택합니다.
        valid_layout = layout_cpu[valid_indices]
        valid_type = type_cpu[valid_indices]
        processed_layouts.append((valid_layout, valid_type))
    return processed_layouts
    
def maximum_iou_one_by_one(layout_set_1, layout_set_2, types):
    
    layout_set_1 = preprocess_layouts(layout_set_1, types)
    layout_set_2 = preprocess_layouts(layout_set_2, types)
    
    total_iou = 0.0
    for layout_1, layout_2 in zip(layout_set_1, layout_set_2):
        max_iou = __compute_maximum_iou_for_layout(layout_1, layout_2)
        total_iou += max_iou

    average_iou = total_iou / len(layout_set_1)
    return average_iou