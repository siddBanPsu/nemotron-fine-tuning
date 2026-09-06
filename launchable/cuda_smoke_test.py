from pathlib import Path

import torch

entrypoint = Path("/workspace/launchable/launchable/container-entrypoint.sh")
if not entrypoint.is_file():
    raise RuntimeError(f"Repository bind mount is unreadable: {entrypoint}")
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot initialize CUDA inside the NeMo container.")
for index in range(torch.cuda.device_count()):
    device = torch.device(f"cuda:{index}")
    value = (torch.ones(1, device=device) + 1).item()
    if value != 2:
        raise RuntimeError(f"CUDA arithmetic smoke test failed on {device}: {value}")
    torch.cuda.synchronize(device)
print(
    f"CUDA smoke test passed on {torch.cuda.device_count()} GPU(s): "
    f"{torch.cuda.get_device_name(0)}"
)
