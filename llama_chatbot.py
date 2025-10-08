from torch import bfloat16
import torch
import transformers
from transformers import AutoTokenizer
import os
from huggingface_hub import login

#MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

class LLAMA_Chat:
    def __init__(self):
        if "HF_ACCESS_TOKEN" in os.environ:
            login(token=os.environ["HF_ACCESS_TOKEN"])

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer

        bnb_config = transformers.BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=bfloat16,
        )

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map=device,
        )

    def ask(self, input_data):
        system_message = "You are a cognitive neurologist."
        formatted_chat = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"{input_data}"},
        ]

        tokenizer = self.tokenizer
        tokenized_prompt = tokenizer.apply_chat_template(formatted_chat, add_generation_prompt=True, return_tensors="pt", max_length=1000, return_dict=True).to("cuda") 

        if 'llama' in MODEL_ID.lower():
            if 'token_type_ids' in tokenized_prompt:  
                del tokenized_prompt['token_type_ids']
        
        outputs = self.model.generate(
                **tokenized_prompt,
                max_new_tokens=1000,
                do_sample = False
            )
        generated_sequence = outputs[0]
        full_answer = self.tokenizer.decode(
            generated_sequence, skip_special_tokens=True)

        return full_answer


if __name__ == "__main__":
    bot = LLAMA_Chat()

    print("\n LLaMA-3 Chat is ready! Type your question (type 'exit' to quit):\n")

    while True:
        user_input = input("Doc Path: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        with open(user_input) as fin:
            all_text_list = fin.readlines()
        all_text = '\n'.join(all_text_list)
        

        print(all_text)
        print("Model answer from here:-----------------------")
        answer = bot.ask(all_text)
        print(answer)
