from .controller import VectorControl


class HookManager:
    """Manages registration and cleanup of hooks for cross-attention output steering."""

    def __init__(self):
        self.hooks = []
        self.controls = []

    def register_vector_controls_with_hooks(self, model, *controls: VectorControl):
        """Register vector controls using PyTorch hooks on cross-attention outputs."""
        self.controls = list(controls)
        self._clear_hooks()

        block_count = self._register_hooks_recursive(model)

        for control in self.controls:
            control.num_attn_layers = block_count

        return self.hooks

    def _register_hooks_recursive(self, model) -> int:
        """Find the U-Net's down/mid/up stages and register hooks in their transformer blocks."""
        block_count = 0

        for name, module in model.named_children():
            if "down" in name:
                block_count += self._register_hooks_in_submodule(module, "down")
            elif "up" in name:
                block_count += self._register_hooks_in_submodule(module, "up")
            elif "mid" in name:
                block_count += self._register_hooks_in_submodule(module, "mid")

        return block_count

    def _register_hooks_in_submodule(self, module, place_in_unet: str) -> int:
        """Register hooks in a submodule for a specific place in the U-Net."""
        block_count = 0

        for name, submodule in module.named_modules():
            if submodule.__class__.__name__ == 'BasicTransformerBlock':
                self._register_hooks_for_sd_block(submodule, place_in_unet)
                block_count += 1

        return block_count

    def _register_hooks_for_sd_block(self, block, place_in_unet: str):
        """Register cross-attention output hook for a Stable Diffusion BasicTransformerBlock."""
        for control in self.controls:
            # Hook into the cross-attention output (attn2)
            if hasattr(block, 'attn2') and block.attn2 is not None:
                hook = block.attn2.register_forward_hook(
                    self._create_attn_output_hook(control, place_in_unet)
                )
                self.hooks.append(hook)

    def _create_attn_output_hook(self, control: VectorControl, place_in_unet: str):
        """Create a forward hook for SD cross-attention output."""
        def hook_fn(module, input, output):
            if not control.active:
                return output

            output_expanded = output[..., None, :]
            controlled_output = control(output_expanded, place_in_unet)
            return controlled_output[..., 0, :]

        return hook_fn

    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def remove_hooks(self):
        """Public method to remove all hooks."""
        self._clear_hooks()


def diffusion_register_vector_controls_with_hooks(model, *controls: VectorControl) -> HookManager:
    """
    Register vector controls using PyTorch hooks on cross-attention outputs.

    Args:
        model: The U-Net to register controls on
        *controls: VectorControl instances to register

    Returns:
        HookManager: Manager object that can be used to remove hooks later
    """
    manager = HookManager()
    manager.register_vector_controls_with_hooks(model, *controls)
    return manager
