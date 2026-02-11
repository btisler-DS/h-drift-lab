from datasets import load_dataset

def load_webgpt():
    ds = load_dataset("openai/webgpt_comparisons", split="train")
    print(ds)
    return ds

if __name__ == "__main__":
    load_webgpt()
