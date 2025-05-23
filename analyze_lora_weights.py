import torch
import safetensors
from safetensors import safe_open
import numpy as np
import re
from collections import defaultdict

def analyze_lora_weights(adapter_path="happy_to_sad_lora_v2/adapter_model.safetensors"):
    """Analyze LoRA weight distribution across layers"""
    
    print("🔍 Analyzing LoRA weight distribution...")
    print(f"📁 Loading from: {adapter_path}")
    
    # Load the LoRA weights
    with safe_open(adapter_path, framework="pt", device="cpu") as f:
        keys = f.keys()
        
        # Group weights by layer number and type
        layer_stats = defaultdict(lambda: {'lora_A': [], 'lora_B': [], 'layer_num': None})
        
        print(f"\n📊 Found {len(keys)} LoRA parameter tensors")
        print("\n🔍 Layer-wise analysis:")
        print("=" * 80)
        
        for key in keys:
            tensor = f.get_tensor(key)
            
            # Extract layer number from key (e.g., "model.layers.0.self_attn.q_proj.lora_A.weight")
            layer_match = re.search(r'layers\.(\d+)', key)
            if layer_match:
                layer_num = int(layer_match.group(1))
                
                # Determine if it's lora_A or lora_B
                if 'lora_A' in key:
                    lora_type = 'lora_A'
                elif 'lora_B' in key:
                    lora_type = 'lora_B'
                else:
                    continue
                
                layer_stats[layer_num]['layer_num'] = layer_num
                layer_stats[layer_num][lora_type].append({
                    'key': key,
                    'shape': tensor.shape,
                    'mean_abs': torch.abs(tensor).mean().item(),
                    'std': tensor.std().item(),
                    'max_abs': torch.abs(tensor).max().item(),
                    'norm': torch.norm(tensor).item()
                })
        
        # Sort by layer number and analyze
        sorted_layers = sorted(layer_stats.keys())
        
        print(f"Layer | LoRA_A (mean_abs) | LoRA_B (mean_abs) | Combined Norm | Components")
        print("-" * 80)
        
        layer_summary = []
        
        for layer_num in sorted_layers:
            stats = layer_stats[layer_num]
            
            # Calculate aggregate stats for this layer
            lora_a_stats = stats['lora_A']
            lora_b_stats = stats['lora_B']
            
            if lora_a_stats and lora_b_stats:
                # Average across all LoRA_A components in this layer
                avg_lora_a_mean = np.mean([s['mean_abs'] for s in lora_a_stats])
                avg_lora_b_mean = np.mean([s['mean_abs'] for s in lora_b_stats])
                
                # Combined norm for the layer
                combined_norm = np.sqrt(
                    sum(s['norm']**2 for s in lora_a_stats) + 
                    sum(s['norm']**2 for s in lora_b_stats)
                )
                
                num_components = len(lora_a_stats) + len(lora_b_stats)
                
                print(f"{layer_num:5d} | {avg_lora_a_mean:12.6f} | {avg_lora_b_mean:12.6f} | {combined_norm:11.4f} | {num_components:9d}")
                
                layer_summary.append({
                    'layer': layer_num,
                    'lora_a_mean': avg_lora_a_mean,
                    'lora_b_mean': avg_lora_b_mean,
                    'combined_norm': combined_norm,
                    'components': num_components
                })
        
        print("\n" + "=" * 80)
        
        # Overall analysis
        if layer_summary:
            layer_nums = [s['layer'] for s in layer_summary]
            combined_norms = [s['combined_norm'] for s in layer_summary]
            
            print(f"\n📈 SUMMARY ANALYSIS:")
            print(f"   Total layers analyzed: {len(layer_summary)}")
            print(f"   Layer range: {min(layer_nums)} - {max(layer_nums)}")
            print(f"   Average norm across all layers: {np.mean(combined_norms):.4f}")
            
            # Check for trends
            early_layers = layer_summary[:len(layer_summary)//3]  # First third
            late_layers = layer_summary[-len(layer_summary)//3:]   # Last third
            
            early_avg_norm = np.mean([s['combined_norm'] for s in early_layers])
            late_avg_norm = np.mean([s['combined_norm'] for s in late_layers])
            
            print(f"   Early layers (first 1/3) avg norm: {early_avg_norm:.4f}")
            print(f"   Late layers (last 1/3) avg norm:  {late_avg_norm:.4f}")
            print(f"   Late/Early ratio: {late_avg_norm/early_avg_norm:.2f}x")
            
            if late_avg_norm > early_avg_norm * 1.2:
                print(f"   🎯 FINDING: Later layers have significantly larger LoRA weights!")
            elif early_avg_norm > late_avg_norm * 1.2:
                print(f"   🎯 FINDING: Earlier layers have significantly larger LoRA weights!")
            else:
                print(f"   🎯 FINDING: LoRA weights are relatively uniform across layers")
                
        print(f"\n🔬 DETAILED BREAKDOWN:")
        print(f"   This shows where the happy→sad learning happened!")
        print(f"   Larger norms = more significant adaptations in those layers")

if __name__ == "__main__":
    analyze_lora_weights() 