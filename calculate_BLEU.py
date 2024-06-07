# from transformers import BleuForRefGenScorer
import torch
from transformers import EncoderDecoderConfig, EncoderDecoderModel, AutoTokenizer
from main import create_model, Arguments
from safetensors.torch import load_file
from datasets import load_metric

from datasets import load_dataset

from tqdm import tqdm

# scorer = BleuForRefGenScorer()
total_bleu = 0.0

# def load_tokenize_data(args):
#     # Example code to load and process code_x_glue_ct_code_to_text python dataset for code summarization task
#     test_datasets  = load_dataset("code_x_glue_ct_code_to_text", 'python', 'test')

    
#     encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
#     decoder_tokenizer = AutoTokenizer.from_pretrained(args.decoder_model_name)
    
#     decoder_tokenizer.pad_token_id = decoder_tokenizer.eos_token_id

#     def preprocess_function(examples):
#         source = [' '.join(ex) for ex in examples["code_tokens"]]
#         target = [' '.join(ex) for ex in examples["docstring_tokens"]]

#         model_inputs = encoder_tokenizer(source, max_length=args.max_source_len, padding="max_length", truncation=True)
#         labels       = decoder_tokenizer(target, max_length=args.max_target_len, padding="max_length", truncation=True)

#         model_inputs["labels"] = labels["input_ids"].copy()
#         model_inputs["labels"] = [
#             [(l if l != decoder_tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
#         ]
#         return model_inputs

#     test_data  = test_datasets.map(preprocess_function, batched=True, remove_columns=test_datasets.column_names, num_proc=64, load_from_cache_file=False,)
    
#     return test_data

def calculate_bleu(model, args, valid_data):
    tokenizer     = AutoTokenizer.from_pretrained(args.encoder_model_name)
    dec_tokenizer = AutoTokenizer.from_pretrained(args.decoder_model_name)
    
    # Load the BLEU metric
    bleu_metric = load_metric("bleu")

    predictions = []
    for sample in tqdm(valid_data):
        print(sample.keys())
        print()
        input  = ' '.join(sample["code_tokens"])
    
        inputs = tokenizer(input, max_length=128, padding="max_length", truncation=True)
        labels = ' '.join(sample["docstring_tokens"])

        outputs_ids   = model.generate(torch.tensor(inputs['input_ids']).unsqueeze(dim=0), attention_mask=torch.tensor(inputs['attention_mask']).unsqueeze(dim=0))
        output        = dec_tokenizer.decode(outputs_ids[0], skip_special_tokens=True)
    
        # Compute BLEU score
        bleu_score = bleu_metric.compute(predictions=output, references=[labels])  
        print("bleu_score", bleu_score)     

    # avg_bleu = total_bleu / len(valid_data)
    # print(f"Average BLEU Score: {avg_bleu}")

if __name__ == '__main__':
    model_path = '/nfs/stak/users/buddhacs/hpc-share/CodeT5_to_GPT/saved_models_without_linear_layer_train_all/checkpoint-102297/model.safetensors'
    args = Arguments()

    model      = create_model()
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict, strict=False)
    test_data  = load_dataset("code_x_glue_ct_code_to_text", 'python', split='test')

    calculate_bleu(model, args, test_data)


