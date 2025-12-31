# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os
import warnings

import torch

from fla.modules.l2norm import l2norm_bwd_from_x
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum, solve_tril
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

_NAN_TRACE = os.environ.get("FLA_NAN_TRACE", "").lower() in ("1", "true", "yes")
_TRACE_STATS = os.environ.get("FLA_TRACE_STATS", "").lower() in ("1", "true", "yes")
_TRACE_THRESHOLD = float(os.environ.get("FLA_TRACE_STATS_THRESHOLD", "0") or "0")


def _sample_tensor(tensor: torch.Tensor, max_elems: int = 262_144) -> torch.Tensor:
    flat = tensor.detach().flatten()
    if flat.numel() <= max_elems:
        return flat
    stride = max(1, flat.numel() // max_elems)
    return flat[::stride][:max_elems]


def _format_stats(label: str, tensor: torch.Tensor) -> str:
    with torch.no_grad():
        nan_count = int(torch.isnan(tensor).sum().item())
        inf_count = int(torch.isinf(tensor).sum().item())
        finite_count = int(torch.isfinite(tensor).sum().item())
        sample = _sample_tensor(tensor).float()
        finite_sample = sample[torch.isfinite(sample)]
        if finite_sample.numel() > 0:
            min_val = float(finite_sample.min().item())
            max_val = float(finite_sample.max().item())
            mean_val = float(finite_sample.mean().item())
            std_val = float(finite_sample.std(unbiased=False).item())
            abs_max = float(finite_sample.abs().max().item())
        else:
            min_val = max_val = mean_val = std_val = abs_max = float("nan")
        return (
            f"{label}=numel:{tensor.numel()} sample:{sample.numel()} "
            f"nan:{nan_count} inf:{inf_count} finite:{finite_count} "
            f"min:{min_val:.3e} max:{max_val:.3e} mean:{mean_val:.3e} "
            f"std:{std_val:.3e} abs_max:{abs_max:.3e}"
        )


def _check_finite(label: str, tensor: torch.Tensor | None) -> None:
    if tensor is None:
        return
    if not _NAN_TRACE:
        return
    if torch.isfinite(tensor).all().item():
        return
    print(f"[nan-trace] {label} non-finite")
    print(
        f"[nan-trace] {label} shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}"
    )
    print(f"[nan-trace] {_format_stats(label, tensor)}")
    raise RuntimeError(f"Non-finite tensor detected in {label}")


def _log_abs_max(label: str, tensor: torch.Tensor | None) -> None:
    if tensor is None or not _TRACE_STATS:
        return
    with torch.no_grad():
        max_abs = float(tensor.detach().float().abs().max().item())
    if _TRACE_THRESHOLD <= 0 or max_abs >= _TRACE_THRESHOLD:
        print(f"[nan-trace] {label} abs_max={max_abs:.3e}")


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    q_rstd: torch.Tensor | None = None,
    k_rstd: torch.Tensor | None = None,
    l2_norm_eps: float = 1e-6,
):
    _check_finite("gated_delta_rule.q", q)
    _check_finite("gated_delta_rule.k", k)
    _check_finite("gated_delta_rule.v", v)
    _check_finite("gated_delta_rule.g_in", g)
    _check_finite("gated_delta_rule.beta", beta)
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
    _check_finite("gated_delta_rule.g_cumsum", g)
    _log_abs_max("gated_delta_rule.g_cumsum", g)
    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
        k_rstd=k_rstd,
        l2_norm_eps=l2_norm_eps,
    )
    _check_finite("gated_delta_rule.A_kkt", A)
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        output_dtype=k.dtype,
    )
    _check_finite("gated_delta_rule.A_solve", A)
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        k_rstd=k_rstd,
        cu_seqlens=cu_seqlens,
    )
    _check_finite("gated_delta_rule.w", w)
    _check_finite("gated_delta_rule.u", u)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        k_rstd=k_rstd,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    _check_finite("gated_delta_rule.h", h)
    _check_finite("gated_delta_rule.v_new", v_new)
    _check_finite("gated_delta_rule.final_state", final_state)
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        q_rstd=q_rstd,
        k_rstd=k_rstd,
        l2_norm_eps=l2_norm_eps,
        cu_seqlens=cu_seqlens,
    )
    _check_finite("gated_delta_rule.o", o)
    _log_abs_max("gated_delta_rule.o", o)
    return g, o, A, final_state


def chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    q_rstd: torch.Tensor | None = None,
    k_rstd: torch.Tensor | None = None,
):
    _check_finite("gated_delta_rule.bwd.q", q)
    _check_finite("gated_delta_rule.bwd.k", k)
    _check_finite("gated_delta_rule.bwd.v", v)
    _check_finite("gated_delta_rule.bwd.g", g)
    _check_finite("gated_delta_rule.bwd.beta", beta)
    _check_finite("gated_delta_rule.bwd.A", A)
    _check_finite("gated_delta_rule.bwd.do", do)
    _log_abs_max("gated_delta_rule.bwd.do", do)
    _check_finite("gated_delta_rule.bwd.dht", dht)
    _log_abs_max("gated_delta_rule.bwd.dht", dht)
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        k_rstd=k_rstd,
        cu_seqlens=cu_seqlens,
    )
    _check_finite("gated_delta_rule.bwd.w", w)
    _check_finite("gated_delta_rule.bwd.u", u)
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        k_rstd=k_rstd,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
    )
    _check_finite("gated_delta_rule.bwd.h", h)
    _check_finite("gated_delta_rule.bwd.v_new", v_new)
    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        g=g,
        do=do,
        scale=scale,
        q_rstd=q_rstd,
        k_rstd=k_rstd,
        cu_seqlens=cu_seqlens,
    )
    _check_finite("gated_delta_rule.bwd.dv_local", dv)
    _log_abs_max("gated_delta_rule.bwd.dv_local", dv)
    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g,
        q_rstd=q_rstd,
        k_rstd=k_rstd,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    _check_finite("gated_delta_rule.bwd.dh", dh)
    _log_abs_max("gated_delta_rule.bwd.dh", dh)
    _check_finite("gated_delta_rule.bwd.dh0", dh0)
    _log_abs_max("gated_delta_rule.bwd.dh0", dh0)
    _check_finite("gated_delta_rule.bwd.dv", dv)
    _log_abs_max("gated_delta_rule.bwd.dv", dv)
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
        q_rstd=q_rstd,
        k_rstd=k_rstd,
        cu_seqlens=cu_seqlens,
    )
    _check_finite("gated_delta_rule.bwd.dq", dq)
    _log_abs_max("gated_delta_rule.bwd.dq", dq)
    _check_finite("gated_delta_rule.bwd.dk", dk)
    _log_abs_max("gated_delta_rule.bwd.dk", dk)
    _check_finite("gated_delta_rule.bwd.dw", dw)
    _log_abs_max("gated_delta_rule.bwd.dw", dw)
    _check_finite("gated_delta_rule.bwd.dg", dg)
    _log_abs_max("gated_delta_rule.bwd.dg", dg)
    dk2, dv, db, dg2 = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=dv,
        k_rstd=k_rstd,
        cu_seqlens=cu_seqlens,
    )
    _check_finite("gated_delta_rule.bwd.dk2", dk2)
    _log_abs_max("gated_delta_rule.bwd.dk2", dk2)
    _check_finite("gated_delta_rule.bwd.dv2", dv)
    _log_abs_max("gated_delta_rule.bwd.dv2", dv)
    _check_finite("gated_delta_rule.bwd.db", db)
    _log_abs_max("gated_delta_rule.bwd.db", db)
    _check_finite("gated_delta_rule.bwd.dg2", dg2)
    _log_abs_max("gated_delta_rule.bwd.dg2", dg2)
    dk.add_(dk2)
    dg.add_(dg2)
    dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens)
    _check_finite("gated_delta_rule.bwd.dg_cumsum", dg)
    _log_abs_max("gated_delta_rule.bwd.dg_cumsum", dg)
    return dq, dk, dv, db, dg, dh0


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
    ):
        q_rstd, k_rstd = None, None
        l2_norm_eps = 1e-6
        if use_qk_l2norm_in_kernel:
            q_rstd = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
            k_rstd = torch.empty(k.shape[:-1], dtype=torch.float32, device=k.device)

        g, o, A, final_state = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            q_rstd=q_rstd,
            k_rstd=k_rstd,
            l2_norm_eps=l2_norm_eps,
        )
        _check_finite("gated_delta_rule.autograd.g", g)
        _check_finite("gated_delta_rule.autograd.o", o)
        _check_finite("gated_delta_rule.autograd.A", A)
        _check_finite("gated_delta_rule.autograd.final_state", final_state)
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor,
    ):
        q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens = ctx.saved_tensors
        _check_finite("gated_delta_rule.autograd.do", do)
        _check_finite("gated_delta_rule.autograd.dht", dht)
        dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            q_rstd=q_rstd,
            k_rstd=k_rstd,
        )
        _check_finite("gated_delta_rule.autograd.dq", dq)
        _check_finite("gated_delta_rule.autograd.dk", dk)
        _check_finite("gated_delta_rule.autograd.dv", dv)
        _check_finite("gated_delta_rule.autograd.db", db)
        _check_finite("gated_delta_rule.autograd.dg", dg)
        _check_finite("gated_delta_rule.autograd.dh0", dh0)
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd_from_x(q, q_rstd, dq)
            dk = l2norm_bwd_from_x(k, k_rstd, dk)
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta), None, dh0, None, None, None


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
):
    g_scale = float(os.environ.get("FLA_G_SCALE", "1.0") or "1.0")
    if g_scale != 1.0:
        g = g * g_scale
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to the q/k tensor internally. Default: `True`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    if 'head_first' in kwargs:
        warnings.warn(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead.",
        )

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
    )
    return o, final_state
