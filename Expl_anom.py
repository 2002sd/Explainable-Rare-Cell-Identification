import numpy as np
from itertools import combinations
import random
fix_seed =42
random.seed(fix_seed)
np.random.seed(fix_seed)
def beam(q, L_max, D, W, k):
    """
    Beam 算法（仅使用 SiNNE 得分）：
    
    参数：
        q    : 查询实例，1D numpy 数组，维度 d
        L_max: 最大子空间维数（搜索深度），不会超过数据维数 d
        D    : 数据集，二维 numpy 数组，形状 (n, d)
        W    : Beam 宽度，每层保留的候选子空间数量
        k    : 最终返回的 top-k 子空间数量
        
    返回：
        S    : 一个列表，每个元素为一个候选子空间（以元组形式保存属性索引）
    """
    fix_seed =42
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    n, d = D.shape
    L_max = min(L_max, d)
    zero_density = [] 
    S = []            
    S_scores = []     
    psi, t = 24, 100  
    params = (psi, t)
    
    # 第一层：单属性子空间
    L = [(i,) for i in range(d)]
    L_scores = sinne_scores(L, D, q, params)
    
    print('finish first layer score')
    # 记录密度为 0（即得分为 -inf）的单属性子空间
    for candidate, score in zip(L, L_scores):
        if score == -np.inf:
            zero_density.append(frozenset(candidate))
    S, S_scores, L = update_top_subspaces(S, S_scores, L, L_scores, k, W)
    print('finish first layer')
    
    # 移除单属性中得分为 -inf 的属性
    removed_attrs = {candidate[0] for candidate, score in zip([(i,) for i in range(d)], L_scores) if score == -np.inf}
    remaining_attrs = [i for i in range(d) if i not in removed_attrs]
    if len(remaining_attrs) < 2:
        return S
    print('finish removing -inf attribute')

    # 第二层：二元子空间（仅考虑剩余属性）
    L = list(combinations(remaining_attrs, 2))
    print('begin layer 2 score')
    L_scores = sinne_scores(L, D, q, params)
    
    print('finish second layer score')
    for candidate, score in zip(L, L_scores):
        if score == -np.inf:
            zero_density.append(frozenset(candidate))
    S, S_scores, L = update_top_subspaces(S, S_scores, L, L_scores, k, W)
    
    # 递归扩展到更高维子空间
    for l in range(3, L_max + 1):
        L_new = []
        for candidate in L:
            candidate_set = set(candidate)
            for attr in range(d):
                if attr not in candidate_set:
                    new_candidate = tuple(sorted(candidate + (attr,)))
                    L_new.append(new_candidate)
        # 去除重复候选
        L_new = list(set(L_new))
        # 剪枝：去除包含已有低密度子空间的候选
        L_filtered = []
        for candidate in L_new:
            candidate_set = set(candidate)
            skip = False
            for zd in zero_density:
                if zd.issubset(candidate_set):
                    skip = True
                    break
            if not skip:
                L_filtered.append(candidate)
        L_new = L_filtered
        if not L_new:
            break
        L_scores = sinne_scores(L_new, D, q, params)
        print('finish layer score: ',l)
        for candidate, score in zip(L_new, L_scores):
            if score == -np.inf:
                zero_density.append(frozenset(candidate))
        S, S_scores, L = update_top_subspaces(S, S_scores, L_new, L_scores, k, W)
    return S

def update_top_subspaces(S, S_scores, L, L_scores, k, W):
    """
    更新全局 top-k 子空间集合和候选子空间集合。
    
    参数：
        S, S_scores: 当前全局 top-k 子空间及其得分（列表形式）
        L, L_scores: 当前候选子空间及其得分（列表形式）
        k         : 返回的 top-k 子空间数量
        W         : 下一层候选子空间保留的数量
        
    返回：
        更新后的 S, S_scores, 以及下一层候选子空间 L（保留 W 个最佳候选）
    """
    # 将当前候选子空间加入全局集合
    S_extended = S + list(L)
    S_scores_extended = S_scores + list(L_scores)
    if len(S_scores_extended) > 0:
        # 按得分从小到大排序（得分越低表示越异常）
        sorted_indices = np.argsort(S_scores_extended)
        sorted_indices = sorted_indices[:k]
        S_new = [S_extended[i] for i in sorted_indices]
        S_scores_new = [S_scores_extended[i] for i in sorted_indices]
    else:
        S_new, S_scores_new = [], []
        
    # 下一层候选：从当前候选中选取得分最低的 W 个子空间
    if len(L) > 0:
        sorted_indices_L = np.argsort(L_scores)
        sorted_indices_L = sorted_indices_L[:min(W, len(L))]
        L_new = [L[i] for i in sorted_indices_L]
    else:
        L_new = []
    return S_new, S_scores_new, L_new

def sinne_scores(L, D, q, params):
    """
    计算 SiNNE 得分：
    对于每个候选子空间（属性索引元组），
    通过 t 次随机采样，每次从该子空间的数据中随机抽取 psi 个样本，
    并计算查询点是否落入随机生成“球体”的范围内，最后取 t 次试验中至少一次成功的比例作为得分。
    
    参数：
        L     : 子空间候选列表，每个元素为属性索引元组
        D     : 数据集，numpy 数组，形状 (n, d)
        q     : 查询点，1D numpy 数组，长度 d
        params: (psi, t) 参数，其中 psi 为每次随机采样的样本数，t 为试验次数
        
    返回：
        scores: numpy 数组，每个候选子空间对应的得分；若得分低于机器精度，则返回 -np.inf
    """
    fix_seed =42
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    psi, t = params
    n, _ = D.shape
    scores = []
    eps = np.finfo(float).eps
    for candidate in L:
        # 提取候选子空间的数据和查询点对应的分量
        data = D[:, candidate]      # 形状 (n, len(candidate))
        query = q[list(candidate)]    # 1D 数组
        # 初始化布尔矩阵：t 次试验，每次 psi 个结果
        fm = np.zeros((t, psi), dtype=bool)
        for j in range(t):
            # 随机采样 psi 个数据点（若样本数不足则允许重复采样）
            if n < psi:
                sample_indices = np.random.choice(n, psi, replace=True)
            else:
                sample_indices = np.random.choice(n, psi, replace=False)
            spheres = data[sample_indices, :]  # 形状 (psi, subspace_dim)
            
            # 计算 spheres 内部各点之间的平方欧氏距离矩阵
            diff = spheres[:, np.newaxis, :] - spheres[np.newaxis, :, :]
            dist_matrix = np.sum(diff**2, axis=2)  # 形状 (psi, psi)
            # 对角线置为无穷大（避免选到自身距离0）
            np.fill_diagonal(dist_matrix, np.inf)
            # 每个球体的半径：对应行中最小的距离（即第二小的距离，因为最小为自身0）
            radii = np.min(dist_matrix, axis=1)  # 形状 (psi,)
            
            # 计算查询点到每个球心的平方欧氏距离
            diff_query = spheres - query  # 形状 (psi, subspace_dim)
            dists_query = np.sum(diff_query**2, axis=1)  # 形状 (psi,)
            # 判断查询点是否落在每个球体内
            fm[j, :] = dists_query <= radii
        
        # 得分为 t 次试验中至少一次成功（落入某球体）的比例
        score = np.mean(np.any(fm, axis=1))
        if score < eps:
            score = -np.inf
        scores.append(score)
    return np.array(scores)