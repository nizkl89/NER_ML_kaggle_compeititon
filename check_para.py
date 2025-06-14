import os
import argparse
import torch
import tensorflow as tf
from safetensors.torch import load_file
import h5py

def get_pytorch_model_size(model_path):
    if model_path.endswith('.bin'):
        state_dict = torch.load(model_path)
    elif model_path.endswith('.safetensors'):
        state_dict = load_file(model_path)
    
    total_params = sum(p.numel() for p in state_dict.values())
    return total_params

def get_tensorflow_model_size(model_path):
    if not model_path.endswith('.h5'):
        raise ValueError("Only .h5 files are supported")

    try:
        # 尝试加载完整模型
        model = tf.keras.models.load_model(model_path)
        return model.count_params()
    except:
        # 失败时直接统计权重参数总数
        try:
            total_params = 0
            with h5py.File(model_path, 'r') as f:
                # 遍历所有层权重
                def count_weights(name, obj):
                    nonlocal total_params
                    if isinstance(obj, h5py.Dataset):
                        total_params += obj.size
                f.visititems(count_weights)
            return total_params
        except Exception as e:
            raise ValueError(f"Invalid .h5 file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Check model parameters for different frameworks')
    parser.add_argument('--framework', choices=['pytorch', 'tensorflow'], 
                      required=True, help='Framework to use (pytorch or tensorflow)')
    args = parser.parse_args()

    # Find model files in current directory based on framework
    if args.framework == 'pytorch':
        model_files = [f for f in os.listdir('.') if f.endswith(('.bin', '.safetensors'))]
        get_size_func = get_pytorch_model_size
    else:  # tensorflow
        model_files = [f for f in os.listdir('.') if f.endswith(('.h5'))]
        get_size_func = get_tensorflow_model_size
    
    if not model_files:
        print(f"No {args.framework} model files found in current directory.")
        return
    
    print(f"\n{args.framework.capitalize()} Model Parameter Analysis:")
    print("=" * 50)
    
    total_params = 0
    for model_file in model_files:
        try:
            params = get_size_func(model_file)
            total_params += params
            print(f"\nModel: {model_file}")
            print(f"Total parameters: {params:,}")
            print(f"Size in millions: {params/1e6:.2f}M")
            print(f"Size in billions: {params/1e9:.2f}B")
        except Exception as e:
            print(f"Error processing {model_file}: {str(e)}")
    
    if len(model_files) > 1:
        print("\n" + "=" * 50)
        print(f"Total across all models:")
        print(f"Total parameters: {total_params:,}")
        print(f"Size in millions: {total_params/1e6:.2f}M")
        print(f"Size in billions: {total_params/1e9:.2f}B")

    # Check if total params exceeds 1 billion
    print("=" * 50)
    if total_params < 1e9:
        print("PASS Parameter Check")
    else:
        print("FAIL Parameter Check")
if __name__ == "__main__":
    main()