import os
import sys
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

def get_text(item, ds_name):
    ds_name_lower = ds_name.lower()
    if "fineweb" in ds_name_lower:
        return item.get("text", "")
    elif "hermes" in ds_name_lower:
        convs = item.get("conversations", [])
        return "\n".join([f"{c.get('from', 'User')}: {c.get('value', '')}" for c in convs])
    elif "stack-exchange" in ds_name_lower:
        # H4/stack-exchange-preferences usually has 'question' and 'answers'
        q = item.get("question", "")
        a = item.get("answers", [{"text": ""}])[0].get("text", "")
        return f"Q: {q}\nA: {a}" if q else item.get("text", "")
    return item.get("text", "")

def download_and_tokenize(dataset_name, subset, out_file, tokenizer, max_samples=None):
    print(f"Loading {dataset_name} ({subset if subset else 'default'}) ...")
    try:
        ds = load_dataset(dataset_name, subset, split="train", streaming=True)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
        
    with open(out_file, 'wb') as f:
        count = 0
        for item in ds:
            text = get_text(item, dataset_name)
            if not text:
                continue
            
            # Tokenize
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=2048,
                add_special_tokens=True
            )['input_ids']
            arr = np.array(tokens, dtype=np.uint16)
            f.write(arr.tobytes())
            
            count += 1
            if count % 10000 == 0:
                print(f"Processed {count} docs from {dataset_name}")
            if max_samples and count >= max_samples:
                break
                
    print(f"Saved {count} documents to {out_file}")

if __name__ == '__main__':
    test_mode = '--test' in sys.argv
    max_samples = 10 if test_mode else None
    
    os.makedirs('baka/data/cache', exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    print("Loaded Llama tokenizer.")
        
    # User requirements
    download_and_tokenize("HuggingFaceFW/fineweb-edu", "sample-10BT", "baka/data/cache/fineweb.bin", tokenizer, max_samples)
    download_and_tokenize("teknium/OpenHermes-2.5", None, "baka/data/cache/openhermes.bin", tokenizer, max_samples)
    download_and_tokenize("HuggingFaceH4/stack-exchange-preferences", None, "baka/data/cache/stackexchange.bin", tokenizer, max_samples)
    
    print("PASS")
