"""
pipeline/reranking.py
=====================
k-Reciprocal Encoding Re-ranking (Zhong et al., CVPR 2017)

참고: "Re-ranking Person Re-Identification with k-Reciprocal Encoding"
      https://arxiv.org/abs/1701.08398

알고리즘 요약:
  1. (query + gallery) 전체 쌍 거리 행렬 계산
  2. 각 샘플의 k-reciprocal nearest neighbor 집합 구성
  3. 집합 간 Jaccard distance 계산
  4. 원래 거리(cosine)와 Jaccard distance 가중 합산 → 최종 거리
"""

from __future__ import annotations

import numpy as np


def re_ranking(
    q_feats: np.ndarray,
    g_feats: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """
    k-reciprocal re-ranking으로 최종 거리 행렬 반환.

    Parameters
    ----------
    q_feats : (Nq, D) L2 정규화된 query 특징 벡터
    g_feats : (Ng, D) L2 정규화된 gallery 특징 벡터
    k1      : k-reciprocal neighbor 수 (기본 20)
    k2      : local query expansion 수 (기본 6)
    lambda_value : 원래 거리 가중치 (기본 0.3)

    Returns
    -------
    dist_final : (Nq, Ng) 최종 거리 행렬
    """
    num_q = q_feats.shape[0]
    num_g = g_feats.shape[0]

    # query + gallery 합쳐서 전체 pairwise 거리 계산
    feat_all = np.concatenate([q_feats, g_feats], axis=0)  # (N, D)
    n = feat_all.shape[0]

    # cosine distance (L2 정규화 가정)
    sim = feat_all @ feat_all.T
    dist_all = np.clip(1.0 - sim, 0.0, 2.0).astype(np.float32)

    # 초기 순위 리스트 (거리 오름차순)
    initial_rank = np.argsort(dist_all, axis=1)  # (N, N)

    # k-reciprocal 특징 행렬 V: V[i, j] = i의 k-reciprocal set에서 j의 가중치
    V = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        # i의 forward k1 neighbors (자기 자신 포함)
        fwd = initial_rank[i, : k1 + 1]

        # reciprocal: fwd 중 i를 자신의 k1 neighbor로 포함하는 것
        bwd = initial_rank[fwd, : k1 + 1]
        reciprocal_mask = np.any(bwd == i, axis=1)
        k_rec = fwd[reciprocal_mask]

        # k-reciprocal expansion
        k_rec_exp = k_rec.copy()
        for cand in k_rec:
            half_k1 = int(round(k1 / 2))
            cand_fwd = initial_rank[cand, : half_k1 + 1]
            cand_bwd = initial_rank[cand_fwd, : half_k1 + 1]
            cand_rec_mask = np.any(cand_bwd == cand, axis=1)
            cand_rec = cand_fwd[cand_rec_mask]
            overlap = len(np.intersect1d(cand_rec, k_rec))
            if overlap > 2 / 3 * len(cand_rec):
                k_rec_exp = np.union1d(k_rec_exp, cand_rec)

        # 가중치: e^(-dist) 정규화
        w = np.exp(-dist_all[i, k_rec_exp])
        V[i, k_rec_exp] = w / (w.sum() + 1e-8)

    # Local query expansion (k2)
    V_qe = V.copy()
    for i in range(n):
        k2_idx = initial_rank[i, :k2]
        V_qe[i] = V[k2_idx].mean(axis=0)

    # Jaccard distance
    # J(i,j) = 1 - |V[i] ∩ V[j]| / |V[i] ∪ V[j]|
    # 벡터화: min/max 연산
    # 메모리 절약을 위해 query 부분(num_q 행)만 계산
    jaccard = np.zeros((num_q, num_g), dtype=np.float32)
    for i in range(num_q):
        vi = V_qe[i]                        # (N,)
        vg = V_qe[num_q:]                   # (Ng, N)
        intersection = np.minimum(vi, vg).sum(axis=1)
        union        = np.maximum(vi, vg).sum(axis=1)
        jaccard[i]   = 1.0 - intersection / (union + 1e-8)

    # 최종 거리 = (1-λ) * Jaccard + λ * cosine
    cosine_dist = dist_all[:num_q, num_q:]
    dist_final  = (1.0 - lambda_value) * jaccard + lambda_value * cosine_dist

    return dist_final.astype(np.float32)
