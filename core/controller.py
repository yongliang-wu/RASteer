from __future__ import annotations  # Python <3.11 compat for `SteeringVectors | None` annotations

import warnings
import torch
import abc
import typing as tp
from collections import defaultdict
from typing import Any

EPS = 1e-6


class VectorControl(abc.ABC):
    def __init__(self, num_layers: int = None):
        self._active = True
        self._diffusion_step = 0
        self._current_attn_layer = 0
        self._current_position = defaultdict(int)
        self.num_attn_layers = num_layers

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, value: bool):
        self._active = value

    def reset(self):
        self._diffusion_step = 0
        self._current_attn_layer = 0
        self._current_position = defaultdict(int)

    @abc.abstractmethod
    def forward(self, vector: torch.Tensor, diffusion_step: int, place_in_unet: str, block_index: int):
        raise NotImplementedError

    def __call__(self, vector: torch.Tensor, place_in_unet: str):
        if not self.active:
            return vector

        block_index = self._current_position[place_in_unet]
        input_shape = vector.shape
        vector = self.forward(vector, self._diffusion_step, place_in_unet, block_index)
        assert vector.shape == input_shape
        self._current_position[place_in_unet] += 1

        self._current_attn_layer += 1
        if self._current_attn_layer == self.num_attn_layers:
            self._current_attn_layer = 0
            self._current_position = defaultdict(int)
            self._diffusion_step += 1
        return vector

# For each diffusion step,
# for each place in the network represented as string key,
# for each layer position, we store steering vector
SteeringVectors = tp.NewType('SteeringVectors', dict[int, dict[str, list[torch.Tensor]]])


class CrossAttentionOutputSteering(VectorControl):
    """
    Cross-attention output steering for concept erasure.

    Each cross-attention output c is steered along the unit erasure direction d:
        alpha = max(beta * <d, c>, 0)      (gate, with intermediate clipping)
        c_new = c - alpha * d

    Without retain concepts, d is the concept's steering vector (CASteer).
    With retain concepts, d is made retain-orthogonal per layer and step (RASteer).
    """

    def __init__(
        self,
        *,
        source_concepts: list[SteeringVectors],
        strength: float,
        device: Any,
        num_layers: int = None,
        intermediate_clipping: bool = True,
        use_first_diffusion_step: bool = False,
        retain_concepts: list[SteeringVectors] | None = None,
        retain_mode: str = 'fixed',
        layer_scope: set[tuple[str, int]] | None = None,
    ):
        super().__init__(num_layers=num_layers)
        self.device = device
        self.intermediate_clipping = intermediate_clipping
        self.strength = strength
        self.use_first_diffusion_step = use_first_diffusion_step
        # Retain subspace: forget direction is orthogonalized against these concepts'
        # directions so that content lying in the retain span is not affected (avoids collateral damage).
        self.retain_concepts = retain_concepts
        # retain_mode: 'fixed' = hard projection; 'auto_layer' = tuning-free projection strength
        # derived per layer from the forget-retain overlap.
        self.retain_mode = retain_mode
        # Optional ablation mask over (U-Net stage, cross-attention index).
        # None preserves the normal behavior of steering every layer.
        self.layer_scope = layer_scope

        if self.strength < 0:
            raise ValueError('Negative values of strength are not supported')

        self.casteer_vectors = []
        for source_concept in source_concepts:
            casteer_concept_transforms = defaultdict(lambda: defaultdict(list))
            for num_steer in source_concept:
                for place_in_unet in source_concept[num_steer]:
                    for block_idx in range(len(source_concept[num_steer][place_in_unet])):
                        steering_vector = source_concept[num_steer][place_in_unet][block_idx]

                        if len(steering_vector.shape) == 1:
                            steering_vector = steering_vector.unsqueeze(0)
                        steering_vector = steering_vector.to(self.device)

                        # Retain-orthogonal projection: remove the component of the forget direction
                        # that lies in the retain subspace -> f_dis = f - R R^T f.
                        if self.retain_concepts:
                            retain_vecs = []
                            for rc in self.retain_concepts:
                                rv = rc[num_steer][place_in_unet][block_idx]
                                if len(rv.shape) == 1:
                                    rv = rv.unsqueeze(0)
                                retain_vecs.append(rv.to(self.device).reshape(1, -1))
                            steering_vector = self._discriminative_direction(steering_vector, retain_vecs, self.retain_mode)

                        casteer_concept_transforms[num_steer][place_in_unet].append(steering_vector)
            self.casteer_vectors.append(casteer_concept_transforms)

    def _discriminative_direction(self, f: torch.Tensor, retain_vecs: list[torch.Tensor],
                                  mode: str = 'fixed') -> torch.Tensor:
        """Retain-orthogonal direction: d = (I - lambda P_r) f.

        Retain SVD: R^T = U S V^T -> U (orthonormal retain basis).
        c_i = <u_i, f_hat> = forget's coordinate on retain mode i; overlap o = sum c_i^2 = ||P_r f_hat||^2.
        Removal weight lambda depends on `mode`:
          'fixed'      : lambda = 1 (hard projection).
          'auto_layer' : data-driven, tuning-free. lambda = 1 - o -> remove the retain component
                         aggressively where the forget barely overlaps retain (cheap, safe), and
                         preserve erasure where overlap is high. Per-layer adaptive (o varies by layer).
        Returns unit (1, H) direction; zeros if it vanishes entirely in the retain span.
        """
        fv = f.reshape(-1).float()                                   # (H,)
        fhat = fv / fv.norm().clamp(min=EPS)
        R = torch.cat([r.reshape(1, -1) for r in retain_vecs], dim=0).float()  # (k, H)
        U, S, _ = torch.linalg.svd(R.t(), full_matrices=False)       # U: (H, k') orthonormal retain basis
        keep = S > (S.max() * 1e-6)
        U, S = U[:, keep], S[keep]
        c = U.t() @ fhat                                             # (k',) forget coords on retain modes

        if mode == 'auto_layer':
            o = (c ** 2).sum().clamp(0.0, 1.0)                       # overlap energy ||P_r f_hat||^2
            lam = (1.0 - o) * torch.ones_like(S)                    # tuning-free, per-layer adaptive
        else:                                                        # fixed hard projection
            lam = torch.ones_like(S)

        proj = U @ (lam * c)                                         # scaled retain projection of f_hat
        d = fhat - proj                                              # (I - P_r) f_hat
        n = torch.linalg.norm(d)
        d = torch.zeros_like(fv) if n < EPS else d / n
        return d.reshape(1, -1).to(f.dtype)

    def steer_with_clipping(self, vector: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Apply steering with intermediate clipping:
            alpha = max(beta * <d, c>, 0)
            c_new = c - alpha * d
        """
        assert len(vector.shape) == 4

        batch_size = vector.shape[0]
        num_heads = vector.shape[2]
        hidden_dim = vector.shape[3]

        b_norm = b / torch.linalg.norm(b, dim=-1, keepdim=True).clamp(min=EPS)  # clamp: retain-ortho may zero out b

        vector_reshaped = vector.to(self.device).reshape(-1, num_heads, hidden_dim).transpose(0, 1)
        # dtype-compat: steering vectors are fp32, model activations are fp16 -> match activation dtype
        b_norm = b_norm.to(vector_reshaped.dtype)
        b_norm_reshaped = b_norm.unsqueeze(-1)

        # Compute dot products between vector components and steering vector
        projection_scores = (
            vector_reshaped @ b_norm_reshaped
        ).transpose(0, 1).reshape(batch_size, -1, num_heads, 1)

        # Clip: only steer when dot product is positive (concept is present)
        if self.intermediate_clipping:
            projection_scores = torch.where(projection_scores > 0, projection_scores, 0)

        steering_delta = -self.strength * projection_scores.to(vector.device) * b_norm.to(vector.device)

        return vector + steering_delta

    # [batch_size, sequence_length, num_heads, head_dim]
    def forward(self, vector: torch.Tensor, diffusion_step: int, place_in_unet: str, block_index: int):
        if self.layer_scope is not None and (place_in_unet, block_index) not in self.layer_scope:
            return vector.half()

        batch_size = vector.shape[0]
        if batch_size > 1:
            # Steer only the prompt part of classifier-free guidance
            batch_slice = slice(batch_size // 2, None)
            warnings.warn('Steering only the prompt part of classifier-free guidance (assumed the batch_idx=0 is not conditioned on the prompt)')
        else:
            batch_slice = slice(None, None)

        vector = vector.detach().clone()

        # First-step vectors for all steps (CASteer) or per-step vectors (RASteer)
        num_steer = 0 if self.use_first_diffusion_step else diffusion_step

        for casteer_vectors in self.casteer_vectors:
            vector[batch_slice, ...] = self.steer_with_clipping(
                vector[batch_slice, ...],
                casteer_vectors[num_steer][place_in_unet][block_index]
            )
        return vector.half()
