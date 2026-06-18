from comfy_api.latest import io
import torch


class SigmaInsert(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SigmaInsert",
            display_name="🐧 Sigma Insert",
            category="SuperNodes/Scheduling",
            description="Inserts a sigma value preceding the specified index. The inserted value must be smaller than the sigma value before it. Index -1 has special behavior: inserts before the first 0.0, or appends at the end if 0.0 is not present.",
            inputs=[
                io.Custom("SIGMAS").Input(
                    "sigmas", tooltip="Input sigma schedule."
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=-1,
                    max=10_000,
                    step=1,
                    tooltip="Index of the sigma to insert preceding. 0 = first, -1 = before 0.0 (or at the end if 0.0 not present).",
                ),
                io.Float.Input(
                    "value",
                    default=1.0,
                    min=0.0,
                    max=10_000.0,
                    step=0.01,
                    tooltip="Sigma value to insert.",
                ),
            ],
            outputs=[
                io.Custom("SIGMAS").Output(
                    tooltip="Sigma schedule with the value inserted."
                ),
            ],
        )

    @classmethod
    def execute(cls, sigmas, index, value) -> io.NodeOutput:
        s = sigmas.clone()
        length = s.shape[0]

        # Handle special index -1
        if index == -1:
            # Check if 0.0 is present in the schedule
            zero_indices = (s == 0.0).nonzero(as_tuple=True)[0]
            if zero_indices.numel() > 0:
                target_idx = int(zero_indices[0].item())
            else:
                target_idx = length
        else:
            target_idx = index

        # Bounds check
        if target_idx < 0 or target_idx > length:
            raise IndexError(
                f"Sigma index out of range: {index} (resolved to target index: {target_idx}, length of sigmas: {length})"
            )

        # Validate that the inserted value is smaller than the sigma value before it
        if target_idx > 0:
            prev_sigma = s[target_idx - 1]
            if value >= prev_sigma:
                raise ValueError(
                    f"Invalid sigma schedule: inserted value {value} at index {target_idx} "
                    f"must be smaller than the preceding sigma[{target_idx - 1}] = {prev_sigma}"
                )

        # Insert value preceding target_idx
        val_tensor = torch.tensor([value], dtype=s.dtype, device=s.device)
        new_sigmas = torch.cat([s[:target_idx], val_tensor, s[target_idx:]])

        return io.NodeOutput(new_sigmas)


NODE = [SigmaInsert]
