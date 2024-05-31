import wandb
import torch.nn as nn
import torch

from transformers import AutoTokenizer, T5EncoderModel, EncoderDecoderModel, AutoConfig, AutoModelForCausalLM,\
                        TrainingArguments, Trainer

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions



from datasets import load_dataset


class T5WithDenseLayer(T5EncoderModel):
    def __init__(self, t5encoder, output_dim):
        config = t5encoder.config
        super().__init__(config)                                # Add Configuration
        super().load_state_dict(t5_encoder.state_dict())        # Add weigths   
        self.dense = nn.Linear(config.hidden_size, output_dim)
        self.activation = nn.ReLU()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs.last_hidden_state
        dense_output    = self.activation(self.dense(sequence_output))

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=dense_output,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions
        )

def load_tokenize_data(args):
    # Example code to load and process code_x_glue_ct_code_to_text python dataset for code summarization task
    datasets  = load_dataset("code_x_glue_ct_code_to_text", 'python')
    train_datasets      = datasets['train']
    validation_datasets = datasets['validation']
    test_datasets       = datasets['test']
    
    encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
    decoder_tokenizer = AutoTokenizer.from_pretrained(args.decoder_model_name)
    
    decoder_tokenizer.pad_token_id = decoder_tokenizer.eos_token_id

    def preprocess_function(examples):
        source = [' '.join(ex) for ex in examples["code_tokens"]]
        target = [' '.join(ex) for ex in examples["docstring_tokens"]]

        model_inputs = encoder_tokenizer(source, max_length=args.max_source_len, padding="max_length", truncation=True)
        labels       = decoder_tokenizer(target, max_length=args.max_target_len, padding="max_length", truncation=True)

        model_inputs["labels"] = labels["input_ids"].copy()
        model_inputs["labels"] = [
            [(l if l != decoder_tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
        ]
        return model_inputs

    train_data = train_datasets.map(preprocess_function, batched=True, remove_columns=train_datasets.column_names, num_proc=64, load_from_cache_file=False,)
    valid_data = validation_datasets.map(preprocess_function, batched=True, remove_columns=train_datasets.column_names, num_proc=64, load_from_cache_file=False,)
    test_data  = test_datasets.map(preprocess_function, batched=True, remove_columns=train_datasets.column_names, num_proc=64, load_from_cache_file=False,)
    
    
    return train_data, valid_data, test_data
    
def run_training(args, model, train_data, valid_data):
    print(f"Starting main loop")

    training_args = TrainingArguments(
        report_to='wandb',
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        do_train=True,
        save_strategy='epoch',

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        per_device_eval_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,
        evaluation_strategy="epoch",

        learning_rate=args.lr,
        weight_decay=0.05,
        warmup_steps=args.lr_warmup_steps,

        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_total_limit=1,

        dataloader_drop_last=True,
        dataloader_num_workers=4,

        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset = valid_data
    )

    trainer.train()

class Arguments:
    def __init__(self):
        self.max_source_len         = 320
        self.max_target_len         = 128
        self.encoder_model_name     = 'Salesforce/codet5p-220m'
        self.decoder_model_name     = 'gpt2'
        
        # Training parameters
        self.epochs                 = 100
        self.lr                     = 5e-5
        self.lr_warmup_steps        = 200
        self.batch_size_per_replica = 16
        self.grad_acc_steps         = 4
        self.local_rank             = -1
        self.deepspeed              = None
        self.fp16                   = False
        
        #Logging
        self.save_dir               = 'saved_models/'
        self.log_freq               = 10
        self.save_freq              = 500
        
        
    
    
args = Arguments()

if __name__ == '__main__':

    wandb.login(key = '9eee79adec4354d94d81dc178c3f9ce537b4802d')
    wandb.init(project='CodeT52GPT', name = 'Experiement')

    args = Arguments()

    decoder_config, kwargs_decoder = AutoConfig.from_pretrained(args.decoder_model_name, return_unused_kwargs=True)
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True
    kwargs_decoder["config"] = decoder_config
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer      = AutoTokenizer.from_pretrained(args.encoder_model_name)
    gpt_tokenizer  = AutoTokenizer.from_pretrained(args.decoder_model_name)
    t5_encoder     = T5EncoderModel.from_pretrained(args.encoder_model_name)
    gpt2_decoder   = AutoModelForCausalLM.from_pretrained(args.decoder_model_name, **kwargs_decoder)
    t5encoderDense = T5WithDenseLayer(t5_encoder, gpt2_decoder.config.n_embd)

    model = EncoderDecoderModel(encoder=t5encoderDense, decoder =gpt2_decoder)

   # training
    model.config.decoder_start_token_id = gpt_tokenizer.bos_token_id
    model.config.eos_token_id           = gpt_tokenizer.eos_token_id
    model.config.pad_token_id           = gpt_tokenizer.eos_token_id
    model.config.vocab_size             = model.config.decoder.vocab_size

    train_data, valid_data, test_data = load_tokenize_data(args)
    
    # To Device Cuda
    model.to(device)

    run_training(args, model, train_data, valid_data)
