import torch
import pywt
import numpy as np
from tqdm import tqdm


def split_dwt(z_tensor_cpu, wavelet_name, dwt_level):
    all_clow_np = []
    all_chigh_list = []
    z_tensor_cpu = z_tensor_cpu.cpu().float()
    
    for i in range(z_tensor_cpu.shape[0]): 
        z_numpy_ch = z_tensor_cpu[i].numpy()
        
        coeffs_ch = pywt.wavedec2(z_numpy_ch, wavelet_name, level=dwt_level, mode='symmetric', axes=(-2, -1))
        
        clow_np = coeffs_ch[0]
        chigh_list = coeffs_ch[1:]
        
        all_clow_np.append(clow_np)
        all_chigh_list.append(chigh_list)
        
    all_clow_tensor = torch.from_numpy(np.stack(all_clow_np, axis=0))
    return all_clow_tensor, all_chigh_list


def reconstruct_dwt(c_low_tensor_cpu, c_high_coeffs, wavelet_name, original_shape):
    H_high, W_high = original_shape
    c_low_tensor_cpu = c_low_tensor_cpu.cpu().float()
    
    clow_np = c_low_tensor_cpu.numpy()
    
    if clow_np.ndim == 4 and clow_np.shape[0] == 1:
        clow_np = clow_np[0]

    coeffs_combined = [clow_np] + c_high_coeffs
    z_recon_np = pywt.waverec2(coeffs_combined, wavelet_name, mode='symmetric', axes=(-2, -1))
    if z_recon_np.shape[-2] != H_high or z_recon_np.shape[-1] != W_high:
        z_recon_np = z_recon_np[..., :H_high, :W_high]
    z_recon_tensor = torch.from_numpy(z_recon_np)
    if z_recon_tensor.ndim == 3:
        z_recon_tensor = z_recon_tensor.unsqueeze(0)
    return z_recon_tensor


def ses_search(
    base_latents,
    objective_reward_fn,
    total_eval_budget=30,
    popsize=10,
    k_elites=5,
    wavelet_name="db1",
    dwt_level=4,
):
    latent_h, latent_w = base_latents.shape[-2], base_latents.shape[-1]
    c_low_init, c_high_fixed_batch = split_dwt(base_latents, wavelet_name, dwt_level)
    c_high_fixed = c_high_fixed_batch[0]    
    c_low_shape = c_low_init.shape[1:]
    mu = torch.zeros_like(c_low_init.view(-1).cpu()) 
    sigma_sq = torch.ones_like(mu) * 1.0 
    
    best_overall = {"fitness": -float('inf'), "score": -float('inf'), "c_low": c_low_init[0]}
    eval_count = 0
    
    elite_db = []    
    n_generations = (total_eval_budget // popsize) + 5
    pbar = tqdm(total=total_eval_budget, desc="[SES] Searching", unit="img")

    for gen in range(n_generations):
        if eval_count >= total_eval_budget: break
        
        std = torch.sqrt(torch.clamp(sigma_sq, min=1e-9))
        z_noise = torch.randn(popsize, mu.shape[0])
        samples_flat = mu + z_noise * std
        samples_reshaped = samples_flat.view(popsize, *c_low_shape) 
        
        batch_results = []
        
        for i in range(popsize):
            if eval_count >= total_eval_budget: break
            
            c_low_sample = samples_reshaped[i].unsqueeze(0) 
            z_recon = reconstruct_dwt(c_low_sample, c_high_fixed, wavelet_name, (latent_h, latent_w))
            z_recon = z_recon.to(base_latents.device, dtype=base_latents.dtype)  
            # img = pipeline_callback(z_recon)

            # score = scorer.get_score(img, prompt)
            score = objective_reward_fn(z_recon)
            res = {
                "score": score, 
                "c_low": c_low_sample.cpu()
            }
            batch_results.append(res)
            if score > best_overall['score']:
                best_overall = res
                
            eval_count += 1
            pbar.update(1)
            
        if not batch_results: break
        elite_db.extend(batch_results)        
        elite_db.sort(key=lambda x: x['score'], reverse=True)        
        elite_db = elite_db[:k_elites]        
        elites_flat = torch.stack([x['c_low'].view(-1) for x in elite_db])
        mu_new = torch.mean(elites_flat, dim=0)
        
        if len(elite_db) > 1:
            sigma_sq_new = torch.var(elites_flat, dim=0, unbiased=True) + 1e-7
        else:
            sigma_sq_new = sigma_sq
        mu = mu_new
        sigma_sq = sigma_sq_new
    pbar.close()
    best_c_low = best_overall['c_low']
    final_latents = reconstruct_dwt(best_c_low, c_high_fixed, wavelet_name, (latent_h, latent_w))
    
    return final_latents.to(base_latents.device, dtype=base_latents.dtype)
