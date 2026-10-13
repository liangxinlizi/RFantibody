import hydra
import random
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig

import rfantibody.rf2.modules.pose_util as pu
import rfantibody.rf2.modules.util as util
from rfantibody.rf2.modules.model_runner import AbPredictor
from rfantibody.rf2.modules.preprocess import Preprocess, pose_to_inference_RFinput
from rfantibody.util.device import get_device, import_torch_npu


@hydra.main(version_base=None, config_path='../src/rfantibody/rf2/config', config_name='base')
def main(conf: HydraConfig) -> None:
    """
    Main function
    """
    print(f'Running RF2 with the following configs: {conf}')
    
    # Set up deterministic mode if seed is provided
    if hasattr(conf.inference, 'seed') and conf.inference.seed is not None:
        seed = conf.inference.seed
        print(f"Setting up deterministic mode with seed {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # CUDA seeding is a silent no-op on a CPU-only torch build; keep it for
        # genuine CUDA hosts and also seed the Ascend NPU when torch_npu is present.
        import_torch_npu()
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    done_list=util.get_done_list(conf)
    device = get_device()
    if device.type == "cpu":
        # A silent CPU fallback for real model inference is forbidden by this
        # migration; fail loudly instead of computing on CPU.
        raise RuntimeError(
            "No NPU or CUDA device available for RF2; CPU fallback is not "
            "permitted by this migration"
        )
    print(f"Running RF2 on device: {device}")
    preprocessor=Preprocess(pose_to_inference_RFinput, conf)
    predictor=AbPredictor(conf, preprocess_fn=preprocessor, device=device)
    for pose, tag in pu.pose_generator(conf):
        if tag in done_list and conf.inference.cautious:
            print(f'Skipping {tag} as output already exists')
            continue
        predictor(pose, tag)

if __name__ == '__main__':
    main()
