import torch 
import numpy as np 
from .wandb_logging import log_image, time_to_log_images

import wandb 

def generate_adversarial_image(self, x, y, batch_idx):
    
        #assert self.params.diffusion_noise==0.0, "Can't have any diffusion noise while adversarial training. Should pretrain model first."

        if self.params.use_auto_attack:

            from autoattack.autopgd_base import APGDAttack
            from autoattack.other_utils import Logger

            apgd = APGDAttack(self, n_restarts=1, n_iter=100, verbose=False,
                eps=self.params.adversarial_max_distortion, norm="Linf", eot_iter=1, rho=.75, 
                    device=self.params.device, logger=Logger(None))

            # apgd on cross-entropy loss
            apgd.loss = 'ce'
            #apgd.seed = self.get_seed()
            x = apgd.perturb(x, y) #cheap=True
                
        else: 
            # using cleverhans
            from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
            from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

            torch.set_grad_enabled(True)
            x.requires_grad = True
            clip_min, clip_max = None, None

            if self.params.use_pgd: 
                x = projected_gradient_descent(self, x, self.params.adversarial_max_distortion, self.params.pgd_step_size,self.params.pgd_step_iters, self.params.adversarial_attack_norm, clip_min=clip_min, clip_max=clip_max)
            else: 
                x = fast_gradient_method(self, x, self.params.adversarial_max_distortion, self.params.adversarial_attack_norm, clip_min=clip_min, clip_max=clip_max)

        x = x.detach()

        if self.params.img_dim is not None and time_to_log_images(self, batch_idx) and not self.training: 
            # then it will log the adversarial example. But only on eval. Avoids duplicates. 
            log_image(self, x, "adversarial_attack_example", "Image", num_images = self.params.num_adversarial_attack_example_imgs)

        return x 