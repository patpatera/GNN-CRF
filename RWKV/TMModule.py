# mypy: disable-error-code="override"
"""Provides a numerically-stable implementation of the WKV computation.

This implementation follows the official implementation.
"""

from typing import cast

import torch
from torch import Tensor
from torch.autograd.function import Function, FunctionCtx, once_differentiable


@torch.jit.script
def wkv_with_eps_forward(w: Tensor, u: Tensor, k: Tensor, v: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    ns, tsz, chans = k.shape

    assert w.shape == u.shape == (chans,)
    assert v.shape == (ns, tsz, chans)
    assert state.shape == (ns, 3, 1, chans)

    alpha, beta, eps = state[:, :, -1].chunk(3, dim=1)  # (B, 1, D), (B, 1, D), (B, 1, D)

    wkvs = []
    alphas = [alpha]
    betas = [beta]
    epss = [eps]

    # Iterate over random walks
    for t in range(tsz):
        kt, vt = k[:, t : t + 1], v[:, t : t + 1]

        ukt = u + kt
        tau = torch.maximum(ukt, eps)
        e1 = torch.exp(eps - tau)
        e2 = torch.exp(ukt - tau)
        wkv = (e1 * alpha + e2 * vt) / (e1 * beta + e2)

        wkvs.append(wkv)

        w_eps = -w + eps
        eps = torch.maximum(w_eps, kt)
        e1 = torch.exp(w_eps - eps)
        e2 = torch.exp(kt - eps)
        alpha = e1 * alpha + e2 * vt
        beta = e1 * beta + e2

        alphas.append(alpha)
        betas.append(beta)
        epss.append(eps)

    alpha = torch.stack(alphas, dim=2)
    beta = torch.stack(betas, dim=2)
    eps = torch.stack(epss, dim=2)

    wkvs = torch.cat(wkvs, 1)
    return wkvs, torch.cat((alpha, beta, eps), dim=1)


@torch.jit.script
def wkv_with_eps_backward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    grad_wkv: Tensor,
    grad_state: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    bs, tsz, chans = k.shape

    assert w.shape == u.shape == (chans,)
    assert v.shape == (bs, tsz, chans)
    assert state.shape == (bs, 3, tsz + 1, chans)
    assert grad_wkv.shape == (bs, tsz, chans)
    assert grad_state.shape == (bs, 3, 1, chans)

    alpha, beta, eps = state.chunk(3, dim=1)  # (B, 1, T + 1, D), (B, 1, T + 1, D), (B, 1, T + 1, D)
    grad_alpha, grad_beta, grad_eps = grad_state[:, :, 0].chunk(3, dim=1)  # (B, 1, D), (B, 1, D), (B, 1, D)
    grad_eps = grad_eps.clone()

    grad_w = torch.zeros_like(w)
    grad_u = torch.zeros_like(u)
    grad_k = torch.zeros((bs, tsz, chans), device=k.device)
    grad_v = torch.zeros((bs, tsz, chans), device=v.device)


    for t in range(tsz - 1, -1, -1):
        kt, vt = k[:, t : t + 1], v[:, t : t + 1]

        alpha_prev, beta_prev, eps_prev = alpha[:, :, t], beta[:, :, t], eps[:, :, t]
        alpha_curr, beta_curr, eps_curr = alpha[:, :, t + 1], beta[:, :, t + 1], eps[:, :, t + 1]
        ukt = u + kt
        tau = torch.maximum(ukt, eps_prev)
        e1 = torch.exp(eps_prev - tau)
        e2 = torch.exp(ukt - tau)

        euke = torch.exp(ukt + eps_prev - 2 * tau)

        denom = e1 * beta_prev + e2
        denom_sq = denom * denom

        grad_wkvt = grad_wkv[:, t : t + 1]

        # Backpropagates wkv gradients.
        grad_uk = grad_wkvt * e2 * (e1 * beta_prev * vt - e1 * alpha_prev) / denom_sq
        grad_u += grad_uk.flatten(0, -2).sum(0)

        grad_k[:, t : t + 1] += grad_uk
        grad_v[:, t : t + 1] += grad_wkvt * e2 / denom

        grad_alpha_wkv = grad_wkvt * e1 / denom
        grad_beta_wkv = -grad_wkvt * e1 * (e2 * vt + e1 * alpha_prev) / denom_sq
        grad_eps_wkv = grad_wkvt * euke * (alpha_prev - vt * beta_prev) / (e1 * beta_prev + e2) ** 2

        e1 = torch.exp(-w + eps_prev - eps_curr)
        e2 = torch.exp(kt - eps_curr)

        # Backpropagates alpha gradients.
        grad_alpha_we = grad_alpha * e1 * alpha_prev
        grad_w += grad_alpha_we.flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += grad_alpha * e2 * vt
        grad_v[:, t : t + 1] += grad_alpha * e2
        grad_eps += grad_alpha * -alpha_curr

        # Backpropagates beta gradients.
        grad_beta_we = grad_beta * e1 * beta_prev
        grad_w += grad_beta_we.flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += grad_beta * e2
        grad_eps += grad_beta * -beta_curr

        # Backpropagates epsilon gradients.
        eps_grad_mask = -w + eps_prev > kt
        grad_eps_we = torch.where(eps_grad_mask, grad_eps, torch.zeros_like(grad_eps))
        grad_w += grad_eps_we.flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += torch.where(eps_grad_mask, torch.zeros_like(grad_eps), grad_eps)

        # Computes gradients for alpha, beta and epsilon.
        grad_alpha = grad_alpha * e1 + grad_alpha_wkv
        grad_beta = grad_beta * e1 + grad_beta_wkv
        grad_eps = grad_alpha_we + grad_beta_we + grad_eps_we + grad_eps_wkv

    return -grad_w, grad_u, grad_k, grad_v, torch.stack((grad_alpha, grad_beta, grad_eps), dim=1)


class WkvWithEps(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        w: Tensor,
        u: Tensor,
        k: Tensor,
        v: Tensor,
        state: Tensor
    ) -> tuple[Tensor, Tensor]:
        wkv, state_out = wkv_with_eps_forward(w, u, k, v, state)
        ctx.save_for_backward(w, u, k, v, state_out)
        return wkv, state_out[:, :, -1:]

    @staticmethod
    @once_differentiable
    def backward(
        ctx: FunctionCtx,
        grad_wkv: Tensor,
        grad_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        w, u, k, v, state = cast(tuple[Tensor, ...], ctx.saved_tensors)
        return wkv_with_eps_backward(w, u, k, v, state, grad_wkv, grad_state)


def initial_state_with_eps(emb_dim: int) -> Tensor:
    return torch.zeros(1, 3, 1, emb_dim)


def wkv_with_eps(w: Tensor, u: Tensor, k: Tensor, v: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    """Runs the core WKV computation.

    Args:
        w: The decay tensor, with shape (D)
        u: The output multiplier tensor, with shape (D)
        k: The K tensor, with shape (B, T, D)
        v: The V tensor, with shape (B, T, D)
        state: The state tensor, with shape (B, 3, T, D), consisting of the
            alpha, beta and eps tensors, each with shape (B, 1, T, D)

    Returns:
        The WKV tensor, with shape (B, T, D), and the next state, with shape
        (B, 3, 1, D), consisting of the next alpha, beta and eps tensors, each
        with shape (B, 1, 1, D)
    """
    
    return WkvWithEps.apply(w, u, k, v, state)



#++++++++++++++++++++++++++++++++++++++++++++++++

# mypy: disable-error-code="override"
"""Provides a numerically-stable implementation of the WKV computation.

This implementation uses log-space state variables, verses the original
implementation which offsets the exponents.
"""

import math
from typing import cast

import torch
from torch import Tensor
from torch.autograd.function import Function, FunctionCtx, once_differentiable

EPS = 1e-4


@torch.jit.script
def logaddexp(a: Tensor, b: Tensor) -> Tensor:
    max_ab = torch.maximum(a, b)
    return max_ab + torch.log(torch.exp(a - max_ab) + torch.exp(b - max_ab))


@torch.jit.script
def logsubexp(a: Tensor, b: Tensor, log_eps: float) -> Tensor:
    max_ab = torch.clamp_min(torch.maximum(a, b), log_eps)
    return max_ab + torch.log(torch.exp(a - max_ab) - torch.exp(b - max_ab))


@torch.jit.script
def wkv_log_space_forward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    eps: float = EPS,
    normalize: bool = False,
) -> tuple[Tensor, Tensor]:
    bsz, tsz, chans = k.shape

    assert w.shape == u.shape == (chans,)
    assert v.shape == (bsz, tsz, chans)
    assert state.shape == (bsz, 3, 1, chans)

    ln_alpha_p, ln_alpha_m, ln_beta = state[:, :, -1].chunk(3, dim=1)

    log_eps = math.log(eps)

    wkvs = []
    ln_alpha_ps = [ln_alpha_p]
    ln_alpha_ms = [ln_alpha_m]
    ln_betas = [ln_beta]

    for t in range(tsz):
        kt, vt = k[:, t : t + 1], v[:, t : t + 1]

            
        vt_p, vt_m = torch.clamp_min(vt, 0) + eps, torch.clamp_min(-vt, 0) + eps
        ln_v_p, ln_v_m = torch.log(vt_p), torch.log(vt_m)

        if True:
            ln_alpha_pm = torch.minimum(ln_alpha_p, ln_alpha_m) - eps
            ln_alpha_p = logsubexp(ln_alpha_p, ln_alpha_pm, log_eps)
            ln_alpha_m = logsubexp(ln_alpha_m, ln_alpha_pm, log_eps)

        ln_wkv_p = logaddexp(u + kt + ln_v_p, ln_alpha_p) - logaddexp(u + kt, ln_beta)
        ln_wkv_m = logaddexp(u + kt + ln_v_m, ln_alpha_m) - logaddexp(u + kt, ln_beta)

        wkv = torch.exp(ln_wkv_p) - torch.exp(ln_wkv_m)
        wkvs.append(wkv)

        ln_alpha_p = logaddexp(-w + ln_alpha_p, kt + ln_v_p)
        ln_alpha_m = logaddexp(-w + ln_alpha_m, kt + ln_v_m)
        ln_beta = logaddexp(-w + ln_beta, kt)

        ln_alpha_ps.append(ln_alpha_p)
        ln_alpha_ms.append(ln_alpha_m)
        ln_betas.append(ln_beta)

    ln_alpha_p = torch.stack(ln_alpha_ps, dim=2)
    ln_alpha_m = torch.stack(ln_alpha_ms, dim=2)
    ln_beta = torch.stack(ln_betas, dim=2)

    return torch.cat(wkvs, 1), torch.cat((ln_alpha_p, ln_alpha_m, ln_beta), dim=1)


@torch.jit.script
def wkv_log_space_backward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    grad_wkv: Tensor,
    grad_state: Tensor,
    eps: float = EPS,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    bsz, tsz, chans = k.shape

    assert w.shape == u.shape == (chans,)
    assert v.shape == (bsz, tsz, chans)
    assert state.shape == (bsz, 3, tsz, chans)
    assert grad_wkv.shape == (bsz, tsz, chans)
    assert grad_state.shape == (bsz, 3, 1, chans)

    grad_ln_alpha_p, grad_ln_alpha_m, grad_ln_beta = grad_state[:, :, 0].chunk(3, dim=1)

    grad_w = torch.zeros_like(w)
    grad_u = torch.zeros_like(u)
    grad_k = torch.zeros_like(k)
    grad_v = torch.zeros_like(v)

    for t in range(tsz - 1, -1, -1):
        kt, vt = k[:, t : t + 1], v[:, t : t + 1]

        vt_p, vt_m = torch.clamp_min(vt, 0) + eps, torch.clamp_min(-vt, 0) + eps
        ln_v_p, ln_v_m = torch.log(vt_p), torch.log(vt_m)

        ln_alpha_p_prev, ln_alpha_m_prev, ln_beta_prev = state[:, :, t].chunk(3, dim=1)

        uk = u + kt
        ukv_p, ukv_m = uk + ln_v_p, uk + ln_v_m

        ukb = logaddexp(uk, ln_beta_prev)
        wkv_p = torch.exp(logaddexp(ukv_p, ln_alpha_p_prev) - ukb)
        wkv_m = torch.exp(logaddexp(ukv_m, ln_alpha_m_prev) - ukb)

        grad_wkvt = grad_wkv[:, t : t + 1]
        grad_ln_wkv_p, grad_ln_wkv_m = grad_wkvt * wkv_p, grad_wkvt * -wkv_m

        # Backpropagates wkv gradients.
        e_num_p = torch.exp(ln_alpha_p_prev - ukv_p)
        e_num_m = torch.exp(ln_alpha_m_prev - ukv_m)
        e_den = torch.exp(ln_beta_prev - uk)

        grad_wkv_den_p = grad_ln_wkv_p / (1 + e_den)
        grad_wkv_den_m = grad_ln_wkv_m / (1 + e_den)
        
        grad_kv_p = grad_ln_wkv_p / (1 + e_num_p)
        grad_kv_m = grad_ln_wkv_m / (1 + e_num_m)
        grad_uk = grad_kv_p + grad_kv_m - grad_wkv_den_p - grad_wkv_den_m
        grad_u += grad_uk.flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += grad_uk
        grad_v[:, t : t + 1] += torch.where(vt > 0, grad_kv_p / vt_p, grad_kv_m / -vt_m)

        grad_ln_alpha_wkv_p = grad_ln_wkv_p / (1 + (1 / e_num_p))
        grad_ln_alpha_wkv_m = grad_ln_wkv_m / (1 + (1 / e_num_m))
        grad_ln_beta_wkv = -grad_ln_wkv_p / (1 + (1 / e_den)) - grad_ln_wkv_m / (1 + (1 / e_den))

        # Backpropagates alpha gradients.
        e_alpha_p = torch.exp(kt + ln_v_p - (-w + ln_alpha_p_prev))
        e_alpha_m = torch.exp(kt + ln_v_m - (-w + ln_alpha_m_prev))
        grad_wa_p = grad_ln_alpha_p / (1 + e_alpha_p)
        grad_wa_m = grad_ln_alpha_m / (1 + e_alpha_m)
        grad_w += (grad_wa_p + grad_wa_m).flatten(0, -2).sum(0)
        grad_kv_p = grad_ln_alpha_p / (1 + (1 / e_alpha_p))
        grad_kv_m = grad_ln_alpha_m / (1 + (1 / e_alpha_m))
        grad_k[:, t : t + 1] += grad_kv_p + grad_kv_m
        grad_v[:, t : t + 1] += torch.where(vt > 0, grad_kv_p / vt_p, -grad_kv_m / vt_m)

        # Backpropagates beta gradients.
        e_beta = torch.exp(kt - (-w + ln_beta_prev))
        grad_wb = grad_ln_beta / (1 + e_beta)
        grad_w += grad_wb.flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += grad_ln_beta / (1 + (1 / e_beta))

        # Compute gradients for log alpha and log beta.
        grad_ln_alpha_p = grad_wa_p + grad_ln_alpha_wkv_p
        grad_ln_alpha_m = grad_wa_m + grad_ln_alpha_wkv_m
        grad_ln_beta = grad_wb + grad_ln_beta_wkv

    return -grad_w, grad_u, grad_k, grad_v, torch.stack((grad_ln_alpha_p, grad_ln_alpha_m, grad_ln_beta), dim=1)


class WkvLogSpace(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        w: Tensor,
        u: Tensor,
        k: Tensor,
        v: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        wkv, state_out = wkv_log_space_forward(w, u, k, v, state)
        ctx.save_for_backward(w, u, k, v, state_out[:, :, :-1])
        return wkv, state_out[:, :, -1:]

    @staticmethod
    @once_differentiable
    def backward(
        ctx: FunctionCtx,
        grad_wkv: Tensor,
        grad_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        w, u, k, v, state = cast(tuple[Tensor, ...], ctx.saved_tensors)
        return wkv_log_space_backward(w, u, k, v, state, grad_wkv, grad_state)


def initial_state_log_space(emb_dim: int) -> Tensor:
    return torch.full((1, 3, 1, emb_dim), float("-inf"))


def wkv_log_space(w: Tensor, u: Tensor, k: Tensor, v: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    """Runs the core WKV computation.

    Args:
        w: The decay tensor, with shape (D)
        u: The output multiplier tensor, with shape (D)
        k: The K tensor, with shape (B, T, D)
        v: The V tensor, with shape (B, T, D)
        state: The state tensor, with shape (B, 3, D), consisting of the
            alpha plus, alpha minus and beta tensors, each with shape (B, 1, D)

    Returns:
        The WKV tensor, with shape (B, T, D), and the next state, with shape
        (B, 2, D), consisting of the next alpha plus, alpha minus and beta
        tensors, each with shape (B, 1, D)
    """
    return WkvLogSpace.apply(w, u, k, v, state)


#++++++++++++++++++++++++++++++++++++++++++++

# mypy: disable-error-code="override"
"""Provides a vanilla implementation of the WKV computation.

This implementation is not numerically stable, it is provided mainly for
educational purposes.
"""

import torch
from torch import Tensor
from torch.autograd.function import Function, FunctionCtx, once_differentiable


@torch.jit.script
def wkv_vanilla_forward(w: Tensor, u: Tensor, k: Tensor, v: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    bsz, tsz, chans = k.shape

    assert w.shape == u.shape == (chans,)
    assert v.shape == (bsz, tsz, chans)
    assert state.shape == (bsz, 2, 1, chans)

    alpha, beta = state[:, :, -1].chunk(2, dim=1)  # (B, 1, D), (B, 1, D)

    ew = torch.exp(-w)

    wkvs = []
    alphas = [alpha]
    betas = [beta]

    for t in range(tsz):
        #kt, vt = k[:, t : t + 1], v[:, t : t + 1]
        
        if t == (tsz-1):
            kt, vt = k[:, t : t + 1], v[:, 0 : 1]
        else:
            kt, vt = k[:, t : t + 1], v[:, t + 1 : t + 2]
        
        
        euk = torch.exp(u + kt)
        wkv = (alpha + euk * vt) / (beta + euk)
        wkvs.append(wkv)

        ek = torch.exp(kt)
        alpha = ew * alpha + ek * vt
        beta = ew * beta + ek

        alphas.append(alpha)
        betas.append(beta)

    alpha = torch.stack(alphas, dim=2)
    beta = torch.stack(betas, dim=2)

    return torch.cat(wkvs, 1), torch.cat((alpha, beta), dim=1)


@torch.jit.script
def wkv_vanilla_backward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    grad_wkv: Tensor,
    grad_state: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    bsz, tsz, chans = k.shape

    assert w.shape == u.shape == (chans,), f"{w.shape}, {u.shape} != {(chans,)}"
    assert v.shape == (bsz, tsz, chans), f"{v.shape} != {(bsz, tsz, chans)}"
    assert state.shape == (bsz, 2, tsz, chans), f"{state.shape} != {(bsz, 2, tsz, chans)}"
    assert grad_wkv.shape == (bsz, tsz, chans), f"{grad_wkv.shape} != {(bsz, tsz, chans)}"
    assert grad_state.shape == (bsz, 2, 1, chans), f"{grad_state.shape} != {(bsz, 2, 1, chans)}"

    alpha, beta = state.chunk(2, dim=1)  # (B, 1, T + 1, D), (B, 1, T + 1, D)
    grad_alpha, grad_beta = grad_state[:, :, 0].chunk(2, dim=1)  # (B, 1, D), (B, 1, D)

    ew = torch.exp(-w)

    grad_w = torch.zeros_like(w)
    grad_u = torch.zeros_like(u)
    grad_k = torch.zeros_like(k)
    grad_v = torch.zeros_like(v)

    for t in range(tsz - 1, -1, -1):
        #kt, vt = k[:, t : t + 1], v[:, t : t + 1]

        if t == (tsz-1):
            kt, vt = k[:, t : t + 1], v[:, 0 : 1]
        else:
            kt, vt = k[:, t : t + 1], v[:, t + 1 : t + 2]
        
        
        alpha_prev, beta_prev = alpha[:, :, t], beta[:, :, t]
        euk = torch.exp(u + kt)
        ek = torch.exp(kt)

        denom = beta_prev + euk
        denom_sq = denom * denom

        grad_wkvt = grad_wkv[:, t : t + 1]

        # Backpropagates wkv gradients.
        grad_uk = grad_wkvt * euk * (beta_prev * vt - alpha_prev) / denom_sq
        grad_u += grad_uk.flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += grad_uk
        grad_v[:, t : t + 1] += grad_wkvt * euk / denom

        grad_alpha_wkv = grad_wkvt / denom
        grad_beta_wkv = -grad_wkvt * (euk * vt + alpha_prev) / denom_sq

        # Backpropagates alpha gradients.
        grad_w += (grad_alpha * ew * alpha_prev).flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += grad_alpha * ek * vt
        grad_v[:, t : t + 1] += grad_alpha * ek

        # Backpropagates beta gradients.
        grad_w += (grad_beta * ew * beta_prev).flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += grad_beta * ek

        # Computes gradients for alpha and beta.
        grad_alpha = grad_alpha * ew + grad_alpha_wkv
        grad_beta = grad_beta * ew + grad_beta_wkv

    return -grad_w, grad_u, grad_k, grad_v, torch.stack((grad_alpha, grad_beta), dim=1)


class WkvVanilla(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        w: Tensor,
        u: Tensor,
        k: Tensor,
        v: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        wkv, state_out = wkv_vanilla_forward(w, u, k, v, state)
        ctx.save_for_backward(w, u, k, v, state_out[:, :, :-1])
        return wkv, state_out[:, :, -1:]

    @staticmethod
    @once_differentiable
    def backward(
        ctx: FunctionCtx,
        grad_wkv: Tensor,
        grad_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        w, u, k, v, state = ctx.saved_tensors
        return wkv_vanilla_backward(w, u, k, v, state, grad_wkv, grad_state)


def initial_state_vanilla(emb_dim: int) -> Tensor:
    return torch.zeros(1, 2, 1, emb_dim)


def wkv_vanilla(w: Tensor, u: Tensor, k: Tensor, v: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    """Runs the core WKV computation.

    Args:
        w: The decay tensor, with shape (D)
        u: The output multiplier tensor, with shape (D)
        k: The K tensor, with shape (B, T, D)
        v: The V tensor, with shape (B, T, D)
        state: The state tensor, with shape (B, 2, T, D), consisting of the
            alpha and beta tensors, each with shape (B, 1, T, D)

    Returns:
        The WKV tensor, with shape (B, T, D), and the next state, with shape
        (B, 2, 1, D), consisting of the next alpha and beta tensors, each with
        shape (B, 1, 1, D)
    """
    return WkvVanilla.apply(w, u, k, v, state)


#===================================================

# mypy: disable-error-code="override"
"""Provides a numerically-stable implementation of the WKV computation.

This implementation uses log-space state variables, verses the original
implementation which offsets the exponents.
"""

import math
from typing import cast

import torch
from torch import Tensor
from torch.autograd.function import Function, FunctionCtx, once_differentiable

EPS = 1e-4


@torch.jit.script
def wkv_mylog_space_forward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    eps: float = EPS,
    normalize: bool = False,
) -> tuple[Tensor, Tensor]:
    bsz, tsz, chans = k.shape

    assert w.shape == u.shape == (chans,)
    assert v.shape == (bsz, tsz, chans)
    assert state.shape == (bsz, 2, 1, chans)

    ln_alpha_p, ln_beta = state[:, :, -1].chunk(2, dim=1)

    log_eps = math.log(eps)

    wkvs = []
    ln_alpha_ps = [ln_alpha_p]
    #ln_alpha_ms = [ln_alpha_m]
    ln_betas = [ln_beta]

    for t in range(tsz):
        kt, vt = k[:, t : t + 1], v[:, t : t + 1]
        ln_v_p = vt

        ln_wkv_p = logaddexp(u + kt + ln_v_p, ln_alpha_p) - logaddexp(u + kt, ln_beta)
        wkv = ln_wkv_p
        wkvs.append(wkv)

        #TODO: sign changed before 'w' value
        ln_alpha_p = logaddexp(-w + ln_alpha_p, kt + ln_v_p)
        ln_beta = logaddexp(-w + ln_beta, kt)

        ln_alpha_ps.append(ln_alpha_p)
        ln_betas.append(ln_beta)

    ln_alpha_p = torch.stack(ln_alpha_ps, dim=2)
    ln_beta = torch.stack(ln_betas, dim=2)

    return torch.cat(wkvs, 1), torch.cat((ln_alpha_p, ln_beta), dim=1)


@torch.jit.script
def wkv_mylog_space_backward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    grad_wkv: Tensor,
    grad_state: Tensor,
    eps: float = EPS,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    bsz, tsz, chans = k.shape

    assert w.shape == u.shape == (chans,)
    assert v.shape == (bsz, tsz, chans)
    assert state.shape == (bsz, 2, tsz, chans)
    assert grad_wkv.shape == (bsz, tsz, chans)
    assert grad_state.shape == (bsz, 2, 1, chans)

    grad_ln_alpha_p, grad_ln_beta = grad_state[:, :, 0].chunk(2, dim=1)

    grad_w = torch.zeros_like(w)
    grad_u = torch.zeros_like(u)
    grad_k = torch.zeros_like(k)
    grad_v = torch.zeros_like(v)

    for t in range(tsz - 1, -1, -1):
        kt, vt = k[:, t : t + 1], v[:, t : t + 1]

        vt_p = vt
        ln_v_p = vt_p

        ln_alpha_p_prev, ln_beta_prev = state[:, :, t].chunk(2, dim=1)

        uk = u + kt
        ukv_p = uk + ln_v_p

        ukb = logaddexp(uk, ln_beta_prev)
        wkv_p = torch.exp(logaddexp(ukv_p, ln_alpha_p_prev) - ukb)
        grad_wkvt = grad_wkv[:, t : t + 1]
        grad_ln_wkv_p = grad_wkvt * wkv_p

        # Backpropagates wkv gradients.
        e_num_p = torch.exp(ln_alpha_p_prev - ukv_p)
        e_den = torch.exp(ln_beta_prev - uk)
        grad_wkv_den_p = grad_ln_wkv_p / (1 + e_den)
        grad_kv_p = grad_ln_wkv_p / (1 + e_num_p)
        grad_uk = grad_kv_p - grad_wkv_den_p #- grad_wkv_den_m
        grad_u += grad_uk.flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += grad_uk
        

        grad_ln_alpha_wkv_p = grad_ln_wkv_p / (1 + (1 / e_num_p))
        grad_ln_beta_wkv = -grad_ln_wkv_p / (1 + (1 / e_den))

        # Backpropagates alpha gradients.
        e_alpha_p = torch.exp(kt + ln_v_p - (-w + ln_alpha_p_prev))
        grad_wa_p = grad_ln_alpha_p / (1 + e_alpha_p)
        grad_w += (grad_wa_p).flatten(0, -2).sum(0)
        grad_kv_p = grad_ln_alpha_p / (1 + (1 / e_alpha_p))
        grad_k[:, t : t + 1] += grad_kv_p  #+ grad_kv_m
        grad_v[:, t : t + 1] += grad_kv_p / vt_p #torch.where(vt > 0, grad_kv_p / vt_p, 0)

        # Backpropagates beta gradients.
        e_beta = torch.exp(kt - (-w + ln_beta_prev))
        grad_wb = grad_ln_beta / (1 + e_beta)
        grad_w += grad_wb.flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += grad_ln_beta / (1 + (1 / e_beta))

        # Compute gradients for log alpha and log beta.
        grad_ln_alpha_p = grad_wa_p + grad_ln_alpha_wkv_p
        grad_ln_beta = grad_wb + grad_ln_beta_wkv

    return grad_w, grad_u, grad_k, grad_v, torch.stack((grad_ln_alpha_p, grad_ln_beta), dim=1)


class WkvMyLogSpace(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        w: Tensor,
        u: Tensor,
        k: Tensor,
        v: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        wkv, state_out = wkv_mylog_space_forward(w, u, k, v, state)
        ctx.save_for_backward(w, u, k, v, state_out[:, :, :-1])
        return wkv, state_out[:, :, -1:]

    @staticmethod
    @once_differentiable
    def backward(
        ctx: FunctionCtx,
        grad_wkv: Tensor,
        grad_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        w, u, k, v, state = cast(tuple[Tensor, ...], ctx.saved_tensors)
        return wkv_mylog_space_backward(w, u, k, v, state, grad_wkv, grad_state)


def initial_state_log_space(emb_dim: int) -> Tensor:
    return torch.full((1, 3, 1, emb_dim), float("-inf"))


def wkv_mylog_space(w: Tensor, u: Tensor, k: Tensor, v: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    """Runs the core WKV computation.

    Args:
        w: The decay tensor, with shape (D)
        u: The output multiplier tensor, with shape (D)
        k: The K tensor, with shape (B, T, D)
        v: The V tensor, with shape (B, T, D)
        state: The state tensor, with shape (B, 3, D), consisting of the
            alpha plus, alpha minus and beta tensors, each with shape (B, 1, D)

    Returns:
        The WKV tensor, with shape (B, T, D), and the next state, with shape
        (B, 2, D), consisting of the next alpha plus, alpha minus and beta
        tensors, each with shape (B, 1, D)
    """
    return WkvMyLogSpace.apply(w, u, k, v, state)