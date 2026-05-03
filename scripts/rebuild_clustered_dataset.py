import os
import glob
import shutil
import random
from PIL import Image
import imagehash
from collections import defaultdict
import networkx as nx

SOURCE_DIR = r'D:\Code\Dataset'
TARGET_DIR = r'D:\Code\Dataset_clustered'
CLASSES = ['bacterial_blight', 'blast', 'brown_spot', 'healthy', 'hispa']
HASH_THRESHOLD = 5 # Hamming distance below this means images are "visually similar"

def get_image_paths(source_dir):
    paths = []
    for cls in CLASSES:
        paths.extend(glob.glob(os.path.join(source_dir, '**', cls, '*.jpg'), recursive=True))
        paths.extend(glob.glob(os.path.join(source_dir, '**', cls, '*.png'), recursive=True))
    return list(set(paths)) 

def compute_hashes(image_paths):
    hashes = {}
    print(f"Computing hashes for {len(image_paths)} images...")
    for idx, path in enumerate(image_paths):
        try:
            with Image.open(path) as img:
                hashes[path] = imagehash.phash(img)
        except Exception as e:
            pass
        if idx > 0 and idx % 2000 == 0:
            print(f"Computed {idx}/{len(image_paths)} hashes...")
    return hashes

def build_clusters(hashes):
    print("Building similarity graph to find visually identical clusters...")
    G = nx.Graph()
    paths = list(hashes.keys())
    G.add_nodes_from(paths)
    
    # Simple O(N^2) comparison for clustering. 
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if hashes[paths[i]] - hashes[paths[j]] <= HASH_THRESHOLD:
                G.add_edge(paths[i], paths[j])
                
    clusters = list(nx.connected_components(G))
    print(f"Formed {len(clusters)} unique visual clusters from {len(paths)} images.")
    return clusters

def distribute_clusters(clusters):
    class_clusters = defaultdict(list)
    for cluster in clusters:
        class_counts = defaultdict(int)
        for path in cluster:
            folder_name = os.path.basename(os.path.dirname(path))
            class_counts[folder_name] += 1
        
        if not class_counts:
            continue
        majority_class = max(class_counts, key=class_counts.get)
        class_clusters[majority_class].append(list(cluster))
        
    splits = {'train': [], 'validation': [], 'test': []}
    
    for cls, c_list in class_clusters.items():
        random.shuffle(c_list)
        n = len(c_list)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        
        # 70/15/15 split of ENTIRE CLUSTERS, ensuring NO LEAKAGE.
        splits['train'].extend(c_list[:train_end])
        splits['validation'].extend(c_list[train_end:val_end])
        splits['test'].extend(c_list[val_end:])
        
    return splits

def copy_to_target(splits, target_dir):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        
    total_files = 0
    for split_name, clusters in splits.items():
        for cluster in clusters:
            for path in cluster:
                # determine class based on folder
                cls_name = os.path.basename(os.path.dirname(path))
                dest_dir = os.path.join(target_dir, split_name, cls_name)
                os.makedirs(dest_dir, exist_ok=True)
                
                filename = os.path.basename(path)
                dest_path = os.path.join(dest_dir, filename)
                counter = 1
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(filename)
                    dest_path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
                    counter += 1
                    
                shutil.copy2(path, dest_path)
                total_files += 1
    return total_files

if __name__ == "__main__":
    print("Starting Phase 1: Cluster-Based Decontamination")
    random.seed(42)
    paths = get_image_paths(SOURCE_DIR)
    
    hashes = compute_hashes(paths)
    clusters = build_clusters(hashes)
    splits = distribute_clusters(clusters)
    
    print("\nDataset Split Summary (Images):")
    for s in ['train', 'validation', 'test']:
        print(f"{s.capitalize()}: {sum(len(c) for c in splits[s])} images from {len(splits[s])} unique clusters")
    
    print("\nCopying to partitioned directory...")
    total = copy_to_target(splits, TARGET_DIR)
    print(f"Done! Copied {total} images. Zero perceptual leakage across splits guaranteed.")