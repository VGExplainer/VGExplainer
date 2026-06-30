import os
import json

class ExplainerEvaluator:
    def __init__(self, base_res_dir, gt_dir):
        """
        Args:
            base_res_dir: Root directory containing 20, 15, 10, 5 folders (Prediction results)
            gt_dir: Directory containing Ground Truth JSON files (True results)
        """
        self.base_res_dir = base_res_dir
        self.gt_dir = gt_dir
        
        # Evaluate in order from most nodes to fewest nodes
        self.ratio_folders = ['20', '15', '10', '5']

    def evaluate(self):
        print(f"{'='*50}")
        print(f"🚀 Starting cross-reference with Ground Truth to evaluate explainer performance")
        print(f"{'='*50}")

        for folder in self.ratio_folders:
            folder_path = os.path.join(self.base_res_dir, folder)
            
            if not os.path.exists(folder_path):
                print(f"⚠️ Folder {folder} not found, skipping.")
                continue

            json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            if not json_files:
                continue

            total_precision = 0.0
            total_recall = 0.0
            total_iou = 0.0
            valid_file_count = 0

            for file in json_files:
                # 1. Join paths for both prediction and ground truth files
                pred_file_path = os.path.join(folder_path, file)
                # Assuming GT filename matches prediction filename without the '1_' or '0_' prefix
                gt_file_path = os.path.join(self.gt_dir, file[2:])
                
                # Skip if the corresponding file does not exist in the Ground Truth folder
                if not os.path.exists(gt_file_path):
                    continue
                
                # 2. Read prediction results (predicted important line numbers)
                with open(pred_file_path, 'r', encoding='utf-8') as f:
                    try:
                        pred_data = json.load(f)
                        pred_lines = set(pred_data.get('important_lines', []))
                    except json.JSONDecodeError:
                        continue
                
                # 3. Read true results (Ground Truth line numbers)
                with open(gt_file_path, 'r', encoding='utf-8') as f:
                    try:
                        gt_data = json.load(f)
                        gt_lines = set(gt_data.get('ground_truth', []))
                    except json.JSONDecodeError:
                        continue

                # If there are no true vulnerability lines, skip this file in statistics
                if not gt_lines:
                    continue

                # 4. Calculate intersection and union
                intersection = pred_lines.intersection(gt_lines)
                union = pred_lines.union(gt_lines)

                # 5. Calculate metrics
                # Precision: Out of the predicted lines, how many are true vulnerabilities? (Intersection / Predictions)
                precision = len(intersection) / len(pred_lines) if pred_lines else 0.0

                # Recall: Out of the true vulnerability lines, how many were successfully found? (Intersection / Ground Truth)
                recall = len(intersection) / len(gt_lines)

                # IoU: Intersection over Union (Intersection / Union)
                iou = len(intersection) / len(union) if union else 0.0

                total_precision += precision
                total_recall += recall
                total_iou += iou
                valid_file_count += 1

            # Calculate average metrics for the current ratio
            if valid_file_count > 0:
                avg_precision = total_precision / valid_file_count
                avg_recall = total_recall / valid_file_count
                avg_iou = total_iou / valid_file_count

                print(f"📂 Folder/Ratio: [{folder}] (Successfully matched and evaluated {valid_file_count} samples)")
                print(f"   🎯 Precision: {avg_precision:.4f} ({avg_precision*100:.2f}%)")
                print(f"   🔍 Recall:    {avg_recall:.4f} ({avg_recall*100:.2f}%)")
                print(f"   📊 IoU:       {avg_iou:.4f} ({avg_iou*100:.2f}%)")
                print("-" * 50)
            else:
                print(f"📂 Folder/Ratio: [{folder}] - No valid files matching Ground Truth were found.")

# ================= Execution Example =================
if __name__ == "__main__":
    evaluator = ExplainerEvaluator(
        # The directory containing the 20, 15, 10, 5 folders generated previously
        base_res_dir="/home/VGExplainer/res_primevul/res_line/deepwukong",
        
        # The directory where Ground Truth files are located
        gt_dir="/home/Dataset/primevul/6_ground_truth/gt_1"
    )
    evaluator.evaluate()