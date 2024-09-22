import argparse
import torch
import json
from utils import process_data, load_checkpoint, predict

def main():
    parser = argparse.ArgumentParser(description="Predict flower name from an image.")
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to category names mapping')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()
    
    # Load the checkpoint and model
    model = load_checkpoint(args.checkpoint)

    # Make prediction
    probs, classes = predict(model,topk=args.top_k, gpu=args.gpu,image_path=args.image_path)
    cat_to_name = "cat_to_name.json"
    with open(cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
    class_names = [cat_to_name[str(cls)] for cls in classes]
     # Print the results
    print(f"{args.image_path}")
    print(f"Top {args.top_k} Predictions:")
    for i in range(args.top_k):
        print(f"{classes[i]}: {probs[i]:.4f} flower name: {class_names[i]}")
if __name__ == "__main__":
    main()
