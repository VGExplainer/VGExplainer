import json
import os
from tqdm import tqdm
from collections import defaultdict

def load_jsonl(file_path):
    data = []
    if not os.path.exists(file_path):
        print(f"Warning: File not found {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def process_and_collect_urls():
    # 1. Configure paths
    base_dir = '/home/Dataset/primevul/0-src'
    save_path_0 = os.path.join(base_dir, '0_novul')
    save_path_1 = os.path.join(base_dir, '1_vul')
    url_list_path = os.path.join(base_dir, 'diff_urls.json')

    for p in [save_path_0, save_path_1]:
        os.makedirs(p, exist_ok=True)

    # 2. Load raw datasets
    print("Loading datasets...")
    files = [
        '/home/Dataset/primevul/primevul_test_paired.jsonl',
        '/home/Dataset/primevul/primevul_valid_paired.jsonl',
        '/home/Dataset/primevul/primevul_train_paired.jsonl'
    ]
    
    all_items = []
    for f in files:
        all_items.extend(load_jsonl(f))

    # 3. Cluster pairs by commit
    groups = defaultdict(dict)
    for item in all_items:
        # Use project + cve + commit_id to ensure uniqueness
        key = f"{item['project']}_{item['cve']}_{item['commit_id']}"
        groups[key][int(item['target'])] = item

    print(f"Found {len(groups)} unique groups. Matching pairs...")

    # 4. Iterate and execute: Save local code + Collect URLs
    diff_url_list = []
    paired_count = 0

    for key, pair in tqdm(groups.items()):
        # Strict check: Must contain both target 0 and target 1
        if 0 in pair and 1 in pair:
            vul_item = pair[1]
            novul_item = pair[0]
            
            # Generate filename (ensure corresponding filenames in 0/1 folders)
            file_base = f"{vul_item['cve']}_{vul_item['project']}_{vul_item['commit_id']}.c"
            
            # Save vulnerable version (target 1)
            with open(os.path.join(save_path_1, f"1_{file_base}"), 'w', encoding='utf-8') as f:
                f.write(vul_item['func'])
            
            # Save fixed version (target 0)
            with open(os.path.join(save_path_0, f"0_{file_base}"), 'w', encoding='utf-8') as f:
                f.write(novul_item['func'])

            # Extract Diff links and organize into a list of dictionaries
            # Appending '.diff' to a GitHub commit URL allows direct patch download
            diff_link = vul_item['commit_url'].rstrip('/') + ".diff"
            
            diff_url_list.append({
                "file_id": file_base.replace('.c', ''),
                "cve": vul_item['cve'],
                "project": vul_item['project'],
                "diff_url": diff_link
            })
            
            paired_count += 1

    # 5. Save the URL list as a JSON file
    with open(url_list_path, 'w', encoding='utf-8') as f_json:
        json.dump(diff_url_list, f_json, indent=4, ensure_ascii=False)

    print("-" * 30)
    print(f"Processing complete!")
    print(f"1. Paired and saved code files: {paired_count} pairs (total {paired_count * 2} files)")
    print(f"2. Diff URL list saved to: {url_list_path}")

if __name__ == '__main__':
    process_and_collect_urls()