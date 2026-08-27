# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch
import transformer_engine as te
import transformer_engine_torch as tex

from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.misc import get_local_tensor_if_DTensor
from cosmos_framework.utils.generator.optimizer import require_fp32_param_group


class FusedAdam(torch.optim.Optimizer):
    """Implements Adam algorithm.


    This version of fused Adam implements 2 fusions.

      * Fusion of the Adam update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to all the model's parameters
        into one or a few kernel launches.

    :class:`FusedAdam` may be used as a drop-in replacement for ``torch.optim.AdamW``,
    or ``torch.optim.Adam`` with ``adam_w_mode=False``::

        opt = FusedAdam(model.parameters(), lr = ....)
        ...
        opt.step()


    .. warning::
        A previous version of :class:`FusedAdam` allowed a number of additional arguments to ``step``.
        These additional arguments are now deprecated and unnecessary.

    Adam was been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        adam_w_mode (boolean, optional): Apply L2 regularization or weight decay
            True for decoupled weight decay(also known as AdamW) (default: True)
        capturable (bool, optional): must be ``True`` (the default); passing ``False``
            raises. Only the CUDA-graph-capturable update is implemented: ``lr`` and the
            per-group ``step`` are kept as device tensors and dispatched to Transformer
            Engine's ``multi_tensor_adam_capturable`` (or its master-weight variant). The
            keyword is retained because the factory in ``utils/optimizer.py`` passes it
            explicitly. (default: True)
        master_weights (bool, optional): whether to maintain FP32 master weights
           in the optimizer with FP16 mixed precision training. (default: False)
           When left disabled, every parameter must be FP32; :meth:`add_param_group`
           rejects anything lower, since without a master weight there would be nothing
           left holding the weights at full precision.

    .. _Adam - A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        adam_w_mode=True,
        weight_decay=0.0,
        amsgrad=False,
        capturable=True,
        master_weights=False,
    ):
        if amsgrad:
            raise RuntimeError("FusedAdam does not support the AMSGrad variant.")
        # Only the capturable update is implemented: lr and the per-group step are held as
        # device tensors and dispatched to TE's multi_tensor_adam_capturable[_master]. The
        # factory in utils/optimizer.py always passes True, so the non-capturable branches
        # were dead and have been removed.
        if not capturable:
            raise ValueError("FusedAdam only supports capturable=True.")

        log.info(f"FusedAdam master_weights: {master_weights} capturable: {capturable}")

        # The capturable kernels read the LR from device memory, so it is a tensor rather
        # than a float (an LR scheduler then updates it in place instead of rebinding it).
        lr = torch.tensor(lr, dtype=torch.float32)
        defaults = dict(lr=lr, bias_correction=bias_correction, betas=betas, eps=eps, weight_decay=weight_decay)
        # Assigned before super().__init__ because that funnels the initial param groups
        # through add_param_group, which reads the flag to decide whether a low precision
        # parameter is acceptable.
        self.master_weights = master_weights
        super(FusedAdam, self).__init__(params, defaults)
        self.adam_w_mode = 1 if adam_w_mode else 0

        # ``capturable`` is always True (enforced above) and is kept as an attribute because
        # external code duck-types on it to detect the group-level step convention (see
        # ``DistillationTrainer._uses_group_step``).
        self.capturable = capturable

        self.param_groups_master = None

        # Move each group's LR onto its params' device, where the capturable kernels read it.
        for idx, group in enumerate(self.param_groups):
            if len(group["params"]) == 0:
                continue
            device = group["params"][0].device
            if isinstance(group["lr"], float):
                group["lr"] = torch.tensor(group["lr"], dtype=torch.float32)
            self.param_groups[idx]["lr"] = group["lr"].to(device=device)

        self._step_supports_amp_scaling = True

        self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")
        self.multi_tensor_adam_capturable = tex.multi_tensor_adam_capturable
        self.multi_tensor_adam_capturable_master = tex.multi_tensor_adam_capturable_master

    def add_param_group(self, param_group: dict) -> None:
        """Register a param group, rejecting non-FP32 params when there is no master weight.

        ``torch.optim.Optimizer.__init__`` funnels every group through here, so this doubles
        as the construction-time check.

        With ``master_weights`` the FP32 master is the accumulator and the parameter is only
        a rounded copy of it, so the parameter's dtype is free. Without one, the parameter
        *is* the accumulator: every step rounds the update into it, and for a BF16 parameter
        (~3 decimal digits) any update finer than its own resolution rounds away to nothing
        -- the underflow gets worse as the weights grow and the LR decays, and it is
        invisible in the loss curve until the model has stopped learning.

        Validates before calling ``super()`` so a rejected group is never registered in
        ``self.param_groups`` -- see :func:`require_fp32_param_group`.
        """
        require_fp32_param_group(
            param_group,
            "FusedAdam",
            "without master weights it applies the update to the parameter itself",
            master_weights=self.master_weights,
        )
        super().add_param_group(param_group)

    def step(self, closure=None, grads=None, output_params=None, scale=None, grad_norms=None, grad_scaler=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        The remaining arguments are deprecated, and are only retained (for the moment) for error-checking purposes.
        """
        if any(p is not None for p in [grads, output_params, scale, grad_norms]):
            raise RuntimeError(
                "FusedAdam has been updated. "
                "Simply initialize it identically to torch.optim.Adam, and call step() with no arguments."
            )
        loss = None
        if closure is not None:
            loss = closure()

        if self.param_groups_master is None:
            # Create full precision master weights
            self.param_groups_master = []
            for i, pg in enumerate(self.param_groups):
                param_list = pg["params"]
                self.param_groups_master.append(
                    {
                        # Change related to master weights
                        "params": [p.clone().detach().float() if self.master_weights else None for p in param_list],
                    }
                )

        for group, group_master in zip(self.param_groups, self.param_groups_master):
            if len(group["params"]) == 0:
                continue
            device = group["params"][0].device
            bias_correction = 1 if "bias_correction" in group and group["bias_correction"] else 0
            beta1, beta2 = group["betas"]

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if "step" in group:
                group["step"] = (
                    group["step"].to(device=device)
                    if isinstance(group["step"], torch.Tensor)
                    else torch.tensor(group["step"], dtype=torch.int32, device=device)
                )
                group["step"] += (self._dummy_overflow_buf != 1).to(torch.int)
            else:
                group["step"] = torch.tensor([1], dtype=torch.int, device=device)

            group["lr"] = (
                group["lr"].to(device=device)
                if isinstance(group["lr"], torch.Tensor)
                else torch.tensor(group["lr"], dtype=torch.float32, device=device)
            )

            # create lists for multi-tensor apply
            g_16, p_16, m_16, v_16 = [], [], [], []
            g_bf, p_bf, m_bf, v_bf = [], [], [], []
            g_32, p_32, m_32, v_32 = [], [], [], []
            p_16_master = []
            p_32_master = []
            bf16_master = []

            for p, p_master in zip(group["params"], group_master["params"]):
                if p.grad is None:
                    continue
                # Unwrap DTensor grads to their local shard before checking
                # sparsity. Touching ``p.grad.data`` directly on a DTensor
                # dispatches ``aten.detach`` through ``__torch_dispatch__``,
                # which operates on a partially-built DTensor shell and raises
                # ``'DTensor' object has no attribute '_local_tensor'``.
                if get_local_tensor_if_DTensor(p.grad).is_sparse:
                    raise RuntimeError(
                        "FusedAdam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p).float()
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p).float()

                if p.dtype == torch.float16:
                    if self.master_weights:
                        p_16_master.append(get_local_tensor_if_DTensor(p_master))
                    g_16.append(get_local_tensor_if_DTensor(p.grad))
                    p_16.append(get_local_tensor_if_DTensor(p))
                    m_16.append(get_local_tensor_if_DTensor(state["exp_avg"]))
                    v_16.append(get_local_tensor_if_DTensor(state["exp_avg_sq"]))
                elif p.dtype == torch.bfloat16:
                    if self.master_weights:
                        bf16_master.append(get_local_tensor_if_DTensor(p_master))
                    g_bf.append(get_local_tensor_if_DTensor(p.grad))
                    p_bf.append(get_local_tensor_if_DTensor(p))
                    m_bf.append(get_local_tensor_if_DTensor(state["exp_avg"]))
                    v_bf.append(get_local_tensor_if_DTensor(state["exp_avg_sq"]))
                elif p.dtype == torch.float32:
                    if self.master_weights:
                        p_32_master.append(get_local_tensor_if_DTensor(p_master))
                    g_32.append(get_local_tensor_if_DTensor(p.grad))
                    p_32.append(get_local_tensor_if_DTensor(p))
                    m_32.append(get_local_tensor_if_DTensor(state["exp_avg"]))
                    v_32.append(get_local_tensor_if_DTensor(state["exp_avg_sq"]))
                else:
                    raise RuntimeError("FusedAdam only support fp16, bf16 and fp32.")

            # A grad scaler, when there is one, works on the GPU: both the overflow flag and
            # the inverse scale are device tensors the capturable kernel consumes.

            # overflow check of gradients
            found_inf = (
                grad_scaler._check_inf_per_device(self)[device]
                if grad_scaler is not None
                else torch.zeros((1,), device=device)
            )
            self._dummy_overflow_buf.copy_(found_inf)

            # get unscale scale factor
            scale, inv_scale = None, None
            if grad_scaler:
                scale = grad_scaler._get_scale_async()
                inv_scale = scale.double().reciprocal().float()
            else:
                scale = torch.ones((1,), device=device, dtype=torch.float32)
                inv_scale = torch.ones((1,), device=device, dtype=torch.float32)

            if len(g_16) > 0:
                te.pytorch.optimizers.multi_tensor_applier(
                    (
                        self.multi_tensor_adam_capturable_master
                        if self.master_weights
                        else self.multi_tensor_adam_capturable
                    ),
                    self._dummy_overflow_buf,
                    [g_16, p_16, m_16, v_16, p_16_master] if self.master_weights else [g_16, p_16, m_16, v_16],
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    group["step"],
                    self.adam_w_mode,
                    bias_correction,
                    group["weight_decay"],
                    inv_scale,
                )

            if len(g_bf) > 0:
                te.pytorch.optimizers.multi_tensor_applier(
                    (
                        self.multi_tensor_adam_capturable_master
                        if self.master_weights
                        else self.multi_tensor_adam_capturable
                    ),
                    self._dummy_overflow_buf,
                    [g_bf, p_bf, m_bf, v_bf, bf16_master] if self.master_weights else [g_bf, p_bf, m_bf, v_bf],
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    group["step"],
                    self.adam_w_mode,
                    bias_correction,
                    group["weight_decay"],
                    inv_scale,
                )

            if len(g_32) > 0:
                te.pytorch.optimizers.multi_tensor_applier(
                    (
                        self.multi_tensor_adam_capturable_master
                        if self.master_weights
                        else self.multi_tensor_adam_capturable
                    ),
                    self._dummy_overflow_buf,
                    [g_32, p_32, m_32, v_32, p_32_master] if self.master_weights else [g_32, p_32, m_32, v_32],
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    group["step"],
                    self.adam_w_mode,
                    bias_correction,
                    group["weight_decay"],
                    inv_scale,
                )

        return loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            # load_state_dict copies param_groups from the checkpoint verbatim, so restore
            # the device tensors the capturable kernels require. Target the params' own
            # device rather than ``.cuda()``, which resolves to the *current* device and so
            # can land the LR/step on a different GPU than the params the kernel reads.
            device = group["params"][0].device if group["params"] else "cuda"
            group["lr"] = (
                group["lr"].to(device=device)
                if isinstance(group["lr"], torch.Tensor)
                else torch.tensor(group["lr"], dtype=torch.float32, device=device)
            )

            if "step" in group:
                if distributed.get_rank() == 0:
                    step = (
                        group["step"].to(device=device)
                        if isinstance(group["step"], torch.Tensor)
                        else torch.tensor([group["step"]], dtype=torch.int32, device=device)
                    )
                else:
                    step = torch.zeros(1, dtype=torch.int32, device=device)
                # make it compatible with FSDP optimizer
                distributed.broadcast(step, 0)
                group["step"] = step
            for p in group["params"]:
                state = self.state[p]
                if "exp_avg" in state:
                    state["exp_avg"] = state["exp_avg"].float()
                    state["exp_avg_sq"] = state["exp_avg_sq"].float()
