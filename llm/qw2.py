import argparse
import time
import readline
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


model_path = "../lm_model"
device = "cuda"

start = time.time()

print("Load model Qwen1.5-0.5B-Chat ...")
model = AutoModelForCausalLM.from_pretrained(
    f"{model_path}/Qwen1.5-0.5B-Chat", 
    #load_in_8bit=False,
    #load_in_4bit=False,
    device_map='auto',
    torch_dtype="auto",
)
#model.generation_config = GenerationConfig.from_pretrained(
#    f"{model_path}/Qwen1.5-0.5B-Chat", trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/Qwen1.5-0.5B-Chat")

print("compiling the model... (takes a ~minute)")
model = torch.compile(model) # requires PyTorch 2.0 (optional)

print(f"Time elapsed: {(time.time() - start):.3f} sec.")


with torch.no_grad():
    print("Start inference mode.")
    print('=' * 85)

    while True:
        raw_input_text = input("(!! to quit.) 请输入您的问题：")
        raw_input_text = str(raw_input_text)
        if raw_input_text.strip() == "!!":
            print("Bye!")
            break
        if len(raw_input_text.strip()) == 0:
            continue

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": raw_input_text}
        ]

        start = time.time()

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print("Response: ", response)
        print(f">>>>>>> Time elapsed: {(time.time() - start):.3f} sec.\n\n")
