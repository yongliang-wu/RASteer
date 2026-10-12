import warnings
from core.controller import VectorControl
from collections import defaultdict
import torch


class CrossAttentionOutputStatsCollector(VectorControl):
    """Collects token-averaged mean activations of cross-attention outputs."""

    def __init__(self):
        super().__init__()

        self._cnt = defaultdict(lambda: defaultdict(list))
        self._m = defaultdict(lambda: defaultdict(list))  # running sum

    def _update_statistics(self, vector: torch.Tensor, diffusion_step, place_in_unet, block_index):
        # vector shape: [num_heads, num_samples, hidden_size]
        stat_count = vector.shape[1]
        stat_m = torch.sum(vector, dim=1)

        if len(self._cnt[diffusion_step][place_in_unet]) <= block_index:
            self._cnt[diffusion_step][place_in_unet].append(stat_count)
            self._m[diffusion_step][place_in_unet].append(stat_m)
        else:
            self._cnt[diffusion_step][place_in_unet][block_index] += stat_count
            self._m[diffusion_step][place_in_unet][block_index] += stat_m

    # [batch_size, sequence_length, num_heads, head_dim]
    def forward(self, vector: torch.Tensor, diffusion_step, place_in_unet, block_index):
        batch_size = vector.shape[0]
        if batch_size > 1:
            # Collect stats only for the prompt part of classifier-free guidance
            batch_slice = slice(batch_size // 2, None)
            warnings.warn('Collecting stats only for the prompt part of classifier-free guidance')
        else:
            batch_slice = slice(None, None)

        num_heads = vector.shape[-2]
        hidden_size = vector.shape[-1]

        vector_slices = vector[batch_slice, ...]

        vector_permuted = vector_slices.permute(2, 0, 1, 3)  # [num_heads, batch_size, sequence_length, head_dim]
        vec = vector_permuted.view(num_heads, -1, hidden_size).float()
        vec = torch.mean(vec, dim=1, keepdim=True)  # average over tokens

        self._update_statistics(vec, diffusion_step, place_in_unet, block_index)

        return vector

    @property
    def means(self):
        result = {}
        for diffusion_step in self._m:
            result[diffusion_step] = {}
            for place_in_unet in self._m[diffusion_step]:
                result[diffusion_step][place_in_unet] = []
                for block_idx in range(len(self._m[diffusion_step][place_in_unet])):
                    count = self._cnt[diffusion_step][place_in_unet][block_idx]
                    m = self._m[diffusion_step][place_in_unet][block_idx] / count
                    result[diffusion_step][place_in_unet].append(m)
        return result
