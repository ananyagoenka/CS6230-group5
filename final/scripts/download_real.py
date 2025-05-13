#!/usr/bin/env python3
"""
Script to download and prepare graph datasets from SNAP (Stanford Network Analysis Platform)
"""

import os
import argparse
import requests
import tarfile
import zipfile
import gzip
import shutil
from pathlib import Path
import pandas as pd
import networkx as nx
import time

# Common SNAP datasets with their URLs and types
SNAP_DATASETS = {
    # Social networks
    'facebook': {
        'url': 'https://snap.stanford.edu/data/facebook_combined.txt.gz',
        'description': 'Facebook social network: nodes represent users, edges represent friendships',
        'directed': False,
        'nodes': 4039,
        'edges': 88234
    },
    'twitter': {
        'url': 'https://snap.stanford.edu/data/twitter_combined.txt.gz',
        'description': 'Twitter social network: nodes represent users, edges represent follows',
        'directed': True,
        'nodes': 81306,
        'edges': 1768149
    },
    'youtube': {
        'url': 'https://snap.stanford.edu/data/com-Youtube.txt.gz',
        'description': 'YouTube social network: nodes represent users, edges represent friendships',
        'directed': False,
        'nodes': 1134890,
        'edges': 2987624
    },
    'orkut': {
        'url': 'https://snap.stanford.edu/data/com-Orkut.txt.gz',
        'description': 'Orkut social network: nodes represent users, edges represent friendships',
        'directed': False,
        'nodes': 3072441,
        'edges': 117185083
    },
    
    # Web graphs
    'web-Google': {
        'url': 'https://snap.stanford.edu/data/web-Google.txt.gz',
        'description': 'Google web graph: nodes represent pages, edges represent hyperlinks',
        'directed': True,
        'nodes': 875713,
        'edges': 5105039
    },
    'web-Stanford': {
        'url': 'https://snap.stanford.edu/data/web-Stanford.txt.gz',
        'description': 'Stanford web graph: nodes represent pages, edges represent hyperlinks',
        'directed': True,
        'nodes': 281903,
        'edges': 2312497
    },
    'web-BerkStan': {
        'url': 'https://snap.stanford.edu/data/web-BerkStan.txt.gz',
        'description': 'Berkeley-Stanford web graph: nodes represent pages, edges represent hyperlinks',
        'directed': True,
        'nodes': 685230,
        'edges': 7600595
    },
    
    # Road networks
    'roadNet-CA': {
        'url': 'https://snap.stanford.edu/data/roadNet-CA.txt.gz',
        'description': 'California road network: nodes represent intersections, edges represent roads',
        'directed': False,
        'nodes': 1965206,
        'edges': 5533214
    },
    'roadNet-PA': {
        'url': 'https://snap.stanford.edu/data/roadNet-PA.txt.gz',
        'description': 'Pennsylvania road network: nodes represent intersections, edges represent roads',
        'directed': False,
        'nodes': 1088092,
        'edges': 3083796
    },
    'roadNet-TX': {
        'url': 'https://snap.stanford.edu/data/roadNet-TX.txt.gz',
        'description': 'Texas road network: nodes represent intersections, edges represent roads',
        'directed': False,
        'nodes': 1379917,
        'edges': 3843320
    },
    
    # Citation networks
    'cit-Patents': {
        'url': 'https://snap.stanford.edu/data/cit-Patents.txt.gz',
        'description': 'Patent citation network: nodes represent patents, edges represent citations',
        'directed': True,
        'nodes': 3774768,
        'edges': 16518948
    },
    'cit-HepPh': {
        'url': 'https://snap.stanford.edu/data/cit-HepPh.txt.gz',
        'description': 'High Energy Physics paper citation network',
        'directed': True,
        'nodes': 34546,
        'edges': 421578
    },
    
    # Amazon product networks
    'amazon0601': {
        'url': 'https://snap.stanford.edu/data/amazon0601.txt.gz',
        'description': 'Amazon product network: nodes represent products, edges represent co-purchasing',
        'directed': True,
        'nodes': 403394,
        'edges': 3387388
    },
    
    # Medium-sized datasets
    'ego-Facebook': {
        'url': 'https://snap.stanford.edu/data/facebook.tar.gz',
        'description': 'Facebook ego networks: 10 ego networks of Facebook users',
        'directed': False,
        'nodes': 4039,
        'edges': 88234,
        'is_archive': True
    },
    'wiki-Vote': {
        'url': 'https://snap.stanford.edu/data/wiki-Vote.txt.gz',
        'description': 'Wikipedia voting network: nodes represent users, edges represent votes',
        'directed': True,
        'nodes': 7115,
        'edges': 103689
    },
    'email-Enron': {
        'url': 'https://snap.stanford.edu/data/email-Enron.txt.gz',
        'description': 'Enron email network: nodes represent email addresses, edges represent emails',
        'directed': True,
        'nodes': 36692,
        'edges': 183831
    },
    'soc-Epinions1': {
        'url': 'https://snap.stanford.edu/data/soc-Epinions1.txt.gz',
        'description': 'Epinions trust network: nodes represent users, edges represent trust',
        'directed': True,
        'nodes': 75879,
        'edges': 508837
    }
}

def download_file(url, save_path):
    """Download a file from URL to save_path with progress reporting"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        print(f"Downloading {url} to {save_path}")
        print(f"Total size: {total_size / (1024*1024):.2f} MB")
        
        with open(save_path, 'wb') as f:
            start_time = time.time()
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Report progress
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        speed = downloaded / (1024 * elapsed)
                        percent = (downloaded / total_size) * 100 if total_size > 0 else 0
                        print(f"\rProgress: {percent:.1f}% ({downloaded/(1024*1024):.2f} MB) - {speed:.2f} KB/s", end='')
            
            print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def extract_file(file_path, extract_dir):
    """Extract downloaded archive file"""
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Handle different compression formats
        if file_ext == '.gz':
            # Check if it's a tar.gz file
            if file_path.endswith('.tar.gz'):
                print(f"Extracting tar.gz archive: {file_path}")
                with tarfile.open(file_path, 'r:gz') as tar:
                    tar.extractall(path=extract_dir)
            else:
                # Regular .gz file (likely a text file)
                print(f"Extracting gz file: {file_path}")
                output_file = os.path.join(extract_dir, os.path.splitext(os.path.basename(file_path))[0])
                with gzip.open(file_path, 'rb') as f_in:
                    with open(output_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
        elif file_ext == '.zip':
            print(f"Extracting zip archive: {file_path}")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif file_path.endswith('.tar.bz2'):
            print(f"Extracting tar.bz2 archive: {file_path}")
            with tarfile.open(file_path, 'r:bz2') as tar:
                tar.extractall(path=extract_dir)
        else:
            print(f"No extraction needed for file: {file_path}")
            return False
        
        print("Extraction complete!")
        return True
    except Exception as e:
        print(f"Error extracting file: {e}")
        return False

def process_dataset(dataset_name, output_dir, keep_original=False):
    """Download and process a SNAP dataset"""
    if dataset_name not in SNAP_DATASETS:
        print(f"Dataset '{dataset_name}' not found in known SNAP datasets")
        return False
    
    dataset_info = SNAP_DATASETS[dataset_name]
    url = dataset_info['url']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download file
    file_name = os.path.basename(url)
    download_path = os.path.join(output_dir, file_name)
    if not download_file(url, download_path):
        return False
    
    # Extract file if needed
    extract_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(extract_dir, exist_ok=True)
    
    is_archive = dataset_info.get('is_archive', False)
    if is_archive or any(file_name.endswith(ext) for ext in ['.gz', '.zip', '.tar.gz', '.tar.bz2']):
        if not extract_file(download_path, extract_dir):
            return False
    else:
        # Just copy the file
        shutil.copy(download_path, os.path.join(extract_dir, os.path.basename(file_name)))
    
    # Clean up original download if not keeping it
    if not keep_original:
        os.remove(download_path)
    
    print(f"Dataset '{dataset_name}' processed successfully!")
    return True

def list_available_datasets():
    """List all available SNAP datasets with details"""
    print("\nAvailable SNAP Datasets:")
    print("=" * 100)
    print(f"{'Name':<20} {'Nodes':<10} {'Edges':<12} {'Directed':<10} Description")
    print("-" * 100)
    
    for name, info in sorted(SNAP_DATASETS.items()):
        print(f"{name:<20} {info.get('nodes', 'N/A'):<10} {info.get('edges', 'N/A'):<12} "
              f"{'Yes' if info.get('directed', False) else 'No':<10} {info.get('description', '')}")
    
    print("\nUse the dataset name with the --datasets parameter to download specific datasets.")
    print("Example: python download_snap.py --datasets facebook twitter")

def main():
    parser = argparse.ArgumentParser(description='Download and prepare SNAP graph datasets')
    parser.add_argument('--datasets', type=str, nargs='+', help='Names of datasets to download')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    parser.add_argument('--output-dir', type=str, default='./data', help='Output directory for datasets')
    parser.add_argument('--keep-original', action='store_true', help='Keep original downloaded files')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_datasets()
        return
    
    if not args.datasets:
        print("No datasets specified. Use --datasets or --list to see available options.")
        return
    
    success_count = 0
    for dataset_name in args.datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        if process_dataset(dataset_name, args.output_dir, args.keep_original):
            success_count += 1
    
    print(f"\nProcessed {success_count}/{len(args.datasets)} datasets successfully.")

if __name__ == '__main__':
    main()