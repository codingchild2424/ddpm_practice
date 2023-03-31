
import torch
import torch.nn.functional as F


class ForwardDiffusion():
    def __init__(self, T, betas, device):
        self.T = T
        self.betas = betas
        self.alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.device = device

        """
        아래에 각 수식 설명 추가하기
        # # betas = tensor([0.001, ...., 0.02])
        # # |betas| = T = 300
        # alphas = 1. - betas
        # # alphas = tensor([0.999, ...., 0.98])
        # # |alphas| = T = 300
        # alphas_cumprod = torch.cumprod(alphas, dim=0)
        # # alphas_cumprod = tensor([0.999, ...., 0.0481])
        # # |alphas_cumprod| = T = 300
        # # note that torch.cumprod() returns returns the cumulative product of elements of input in the dimension dim.
        # alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        # # 가장 오른쪽(마지막)의 값을 잘라내고, 가장 왼쪽(첫번째)에 1.0을 추가한다.
        # # alphas_cumprod_prev = tensor([1.0000, 0.9999, ...., 0.0490])
        # # |alphas_cumprod_prev| = T = 300
        # sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        # # sqrt_recip_alphas = tensor([1.0000, 1.0001, ...., 1.0102])
        # # |sqrt_recip_alphas| = T = 300
        # sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        # # sqrt_alphas_cumprod = tensor([0.9999, ...., 0.2192])
        # # |sqrt_alphas_cumprod| = T = 300
        # sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        # # sqrt_one_minus_alphas_cumprod = tensor([0.0100, ...., 0.9757])
        # # |sqrt_one_minus_alphas_cumprod| = T = 300
        # posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # # posterior_variance = tensor([0.0000, ...., 0.01997])
        """


    def get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


    def forward_diffusion_sample(self, x_0, t, device="cpu"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_alphas_cumprod, t, x_0.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        # Reparameterization trick
        # x_t = sqrt(alphas_cumprod_t) * x_0 + sqrt(1 - alphas_cumprod_t) * noise
        # mean + variance
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
            + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), \
                noise.to(device)
        
    def get_loss(self, model, x_0, t):
        x_noisy, noise = self.forward_diffusion_sample(x_0, t, self.device)
        noise_pred = model(x_noisy, t)
        return F.l1_loss(noise, noise_pred)