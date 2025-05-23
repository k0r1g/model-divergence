import torch
import safetensors
from safetensors import safe_open
import numpy as np
import re
from collections import defaultdict

def visualize_lora_trend():
    """Create a simple ASCII visualization of LoRA weight trends"""
    
    adapter_path = "happy_to_sad_lora_v2/adapter_model.safetensors"
    
    with safe_open(adapter_path, framework="pt", device="cpu") as f:
        keys = f.keys()
        layer_stats = defaultdict(lambda: {'lora_A': [], 'lora_B': []})
        
        for key in keys:
            tensor = f.get_tensor(key)
            layer_match = re.search(r'layers\.(\d+)', key)
            if layer_match:
                layer_num = int(layer_match.group(1))
                if 'lora_A' in key:
                    layer_stats[layer_num]['lora_A'].append(torch.norm(tensor).item())
                elif 'lora_B' in key:
                    layer_stats[layer_num]['lora_B'].append(torch.norm(tensor).item())
        
        # Calculate combined norms per layer
        sorted_layers = sorted(layer_stats.keys())
        layer_norms = []
        
        for layer_num in sorted_layers:
            stats = layer_stats[layer_num]
            combined_norm = np.sqrt(
                sum([x**2 for x in stats['lora_A']]) + 
                sum([x**2 for x in stats['lora_B']])
            )
            layer_norms.append((layer_num, combined_norm))
        
        print("🎨 LoRA Weight Progression (ASCII Visualization)")
        print("=" * 60)
        print("Layer |" + " " * 20 + "Norm" + " " * 20 + "| Value")
        print("-" * 60)
        
        # Normalize for visualization
        min_norm = min(norm for _, norm in layer_norms)
        max_norm = max(norm for _, norm in layer_norms)
        norm_range = max_norm - min_norm
        
        for layer, norm in layer_norms:
            # Create bar visualization
            normalized = (norm - min_norm) / norm_range
            bar_length = int(normalized * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            
            print(f"{layer:5d} |{bar}| {norm:6.3f}")
        
        print("-" * 60)
        print(f"Range: {min_norm:.3f} → {max_norm:.3f} (Δ = {max_norm-min_norm:.3f})")
        
        # Show trend
        first_third = layer_norms[:len(layer_norms)//3]
        last_third = layer_norms[-len(layer_norms)//3:]
        
        early_avg = np.mean([norm for _, norm in first_third])
        late_avg = np.mean([norm for _, norm in last_third])
        
        print(f"\n📊 TREND ANALYSIS:")
        print(f"   Early layers (0-{first_third[-1][0]}): {early_avg:.3f}")
        print(f"   Late layers ({last_third[0][0]}-{last_third[-1][0]}):  {late_avg:.3f}")
        print(f"   Growth: {((late_avg/early_avg - 1) * 100):+.1f}%")
        
        if late_avg > early_avg:
            print(f"   📈 Trend: Increasing toward output layers")
            print(f"   🎯 Interpretation: Semantic/output processing most affected")
        else:
            print(f"   📉 Trend: Decreasing toward output layers")

if __name__ == "__main__":
    visualize_lora_trend() 