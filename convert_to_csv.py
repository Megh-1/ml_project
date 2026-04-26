import torch
import pandas as pd
import numpy as np
import os

def convert_pt_to_csv(pt_path, csv_path):
    print(f"Loading {pt_path}...")
    data = torch.load(pt_path)
    
    if isinstance(data, torch.Tensor):
        # Move to CPU if it's on GPU
        data = data.cpu().numpy()
    
    print(f"Converting to DataFrame and saving to {csv_path}...")
    # For very large files, we might want to save in chunks if it's a 2D array
    if len(data.shape) == 1:
        df = pd.DataFrame(data, columns=['label'])
        df.to_csv(csv_path, index=False)
    elif len(data.shape) == 2:
        # Saving 410k x 788 might take a lot of memory
        # We'll save it using numpy which can be more memory efficient for large arrays
        np.savetxt(csv_path, data, delimiter=",")
    else:
        print(f"Unsupported shape: {data.shape}")

if __name__ == "__main__":
    # Convert labels
    convert_pt_to_csv('data/MGStBot-large/labels_bot.pt', 'data/MGStBot-large/labels_bot.csv')
    
    # Convert features
    convert_pt_to_csv('data/MGStBot-large/large_features.pt', 'data/MGStBot-large/large_features.csv')
