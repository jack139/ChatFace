import argparse
import time
import readline
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=".", type=str)
parser.add_argument('--max_length', default=200, type=int)
parser.add_argument('--no_compile', action='store_true', help='do not compile model (PyTorch < 2.0)')
args = parser.parse_args()

if torch.cuda.is_available():
    torch.set_default_device("cuda")
    load_in_4bit = True
else:
    torch.set_default_device("cpu")
    load_in_4bit = False

print(f"load_in_4bit = {load_in_4bit}")

start = time.time()
print("Load model QWen-1.8B-Chat ...")
model = AutoModelForCausalLM.from_pretrained(
    f"{args.model_path}/Qwen-1_8B-Chat", 
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    device_map='auto',
    load_in_4bit=load_in_4bit,
    trust_remote_code=True,
)
model.generation_config = GenerationConfig.from_pretrained(
    f"{args.model_path}/Qwen-1_8B-Chat", trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(
    f"{args.model_path}/Qwen-1_8B-Chat", trust_remote_code=True)

if not args.no_compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model) # requires PyTorch 2.0 (optional)
print(f"Time elapsed: {(time.time() - start):.3f} sec.")


with torch.no_grad():
    print("Start inference mode.")
    print('=' * 85)
    history = None

    while True:
        raw_input_text = input("(## to clear history; !! to quit.) Prompt:")
        raw_input_text = str(raw_input_text)
        if raw_input_text.strip() == "##":
            history = None
            print("History cleared.")
            continue
        if raw_input_text.strip() == "!!":
            print("Bye!")
            break
        if len(raw_input_text.strip()) == 0:
            continue

        start = time.time()
        response, history = model.chat(tokenizer, raw_input_text, history=history)
        print("Response: ", response)
        print("\nHistory: ", history)
        print(f">>>>>>> Time elapsed: {(time.time() - start):.3f} sec.\n\n")
