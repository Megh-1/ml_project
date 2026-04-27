import pandas as pd
import json
import numpy as np

def extract_features():
    print("Loading raw node.json...", flush=True)
    with open('dataset2/node.json', 'r') as f:
        data = json.load(f)

    print("Extracting features...", flush=True)
    features = []
    for user in data:
        row = {}
        row['id'] = user.get('id')
        
        # Public Metrics
        metrics = user.get('public_metrics')
        if metrics:
            row['followers_count'] = metrics.get('followers_count', 0)
            row['following_count'] = metrics.get('following_count', 0)
            row['tweet_count'] = metrics.get('tweet_count', 0)
            row['listed_count'] = metrics.get('listed_count', 0)
        else:
            row['followers_count'] = 0
            row['following_count'] = 0
            row['tweet_count'] = 0
            row['listed_count'] = 0

        # String lengths
        row['name_length'] = len(user.get('name', '')) if user.get('name') else 0
        row['username_length'] = len(user.get('username', '')) if user.get('username') else 0
        row['description_length'] = len(user.get('description', '')) if user.get('description') else 0
        
        # Booleans
        row['verified'] = int(bool(user.get('verified')))
        row['protected'] = int(bool(user.get('protected')))
        
        # Ratios (safe division)
        follow = row['following_count']
        follower = row['followers_count']
        row['ff_ratio'] = follower / follow if follow > 0 else follower
        
        features.append(row)

    df_features = pd.DataFrame(features)
    print("DataFrame shape:", df_features.shape, flush=True)
    
    print("Loading labels and splits...")
    df_labels = pd.read_csv('dataset2/label.csv')
    df_split = pd.read_csv('dataset2/split.csv')

    # Fix label 'bot' -> 1, human -> 0. Wait, earlier dataset had 1=bot, 0=human. Let's map it.
    df_labels['label'] = df_labels['label'].map({'bot': 1, 'human': 0})
    
    # Merge
    merged = df_features.merge(df_labels, on='id', how='inner')
    merged = merged.merge(df_split, on='id', how='inner')
    
    print("Merged shape:", merged.shape, flush=True)

    # Export
    train = merged[merged['split'] == 'train'].drop(columns=['split'])
    val = merged[merged['split'] == 'val'].drop(columns=['split'])
    test = merged[merged['split'] == 'test'].drop(columns=['split'])
    
    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}", flush=True)

    train.to_csv('dataset2/d2_train.csv', index=False)
    val.to_csv('dataset2/d2_val.csv', index=False)
    test.to_csv('dataset2/d2_test.csv', index=False)
    print("Successfully built d2 features!")

if __name__ == '__main__':
    extract_features()
