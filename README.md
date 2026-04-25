# InstFree
Instruction-Free Tuning of Large Vision Language Models for Medical Instruction Following

## Running fine-tuning and test recipes

```
source activate instfree

python

import os
GPU='0'
llama_dir='./models/llama3/Llama-3.2-11B-Vision-Instruct'
dataset='wbcatt'  # skincon, wbcatt, cbis
data_dir=f'./datasets/concept/{dataset}/'
result_dir=f'./results_instfree'
question='\"Describe this medical scan.\"'
epochs=5
seed_list=[1,]
test_csv_list=['image_text_test_att1.csv','image_text_val_att1.csv']
name='report'
proxy_instruction=True
proxy_instruction_type='momentum'
proxy_instruction_momentum=True
proxy_instruction_momentum_value=0.999
num_proxy_instruction=8
response_shuffling='comma'  # no, comma
filename_list=['image_text']
for seed in seed_list:
    for filename in filename_list:
        postfix_result = f'_{dataset}_{filename}'
        if proxy_instruction:
            postfix_result += f'_dim{num_proxy_instruction}'
            postfix_result += f'_momentum_{proxy_instruction_momentum_value}'
        if response_shuffling != 'no':
            postfix_result += f'_RS_{response_shuffling}'
        postfix_result += f'_s{seed}'
        if proxy_instruction:
            cmd=f"PYTHONPATH=./ CUDA_VISIBLE_DEVICES={GPU} python torchtune/_cli/tune.py run full_finetune_single_device --config recipes/configs/llama3_2_vision/11B_full_single_device.yaml \
                seed={seed} \
                epochs=1 \
                batch_size=2 \
                gradient_accumulation_steps=1 \
                optimizer._component_=torch.optim.AdamW \
                optimizer.fused=True \
                optimizer.lr=2e-5 \
                dataset._component_=torchtune.datasets.multimodal.the_cauldron_dataset \
                dataset.subset={name} \
                dataset.data_dir={data_dir} \
                dataset.csv_filename={filename}.csv \
                dataset.question={question} \
                dataset.image_size=560 \
                dataset.proxy_instruction=True \
                dataset.proxy_instruction_warmup=True \
                dataset.num_proxy_instruction={num_proxy_instruction} \
                dataset.response_shuffling={response_shuffling} \
                model.image_size=560 \
                tokenizer.image_size=560 \
                tokenizer.max_seq_len=8192 \
                tokenizer.path={llama_dir}/original/tokenizer.model \
                checkpointer.checkpoint_dir={llama_dir} \
                output_dir={result_dir}/{name}{postfix_result}/warmup"
            print(cmd)
            os.system(cmd)
            proxy_instruction_resume=f'{result_dir}/{name}{postfix_result}/warmup/proxy_instruction_0.pt'
        else:
            proxy_instruction_resume='False'
        cmd=f"PYTHONPATH=./ CUDA_VISIBLE_DEVICES={GPU} python torchtune/_cli/tune.py run full_finetune_single_device --config recipes/configs/llama3_2_vision/11B_full_single_device.yaml \
            seed={seed} \
            epochs={epochs} \
            batch_size=2 \
            gradient_accumulation_steps=1 \
            optimizer._component_=torch.optim.AdamW \
            optimizer.fused=True \
            optimizer.lr=2e-5 \
            dataset._component_=torchtune.datasets.multimodal.the_cauldron_dataset \
            dataset.subset={name} \
            dataset.data_dir={data_dir} \
            dataset.csv_filename={filename}.csv \
            dataset.question={question} \
            dataset.image_size=560 \
            dataset.proxy_instruction={proxy_instruction} \
            dataset.proxy_instruction_momentum={proxy_instruction_momentum} \
            dataset.proxy_instruction_momentum_value={proxy_instruction_momentum_value} \
            dataset.num_proxy_instruction={num_proxy_instruction} \
            dataset.proxy_instruction_resume={proxy_instruction_resume} \
            dataset.response_shuffling={response_shuffling} \
            model.image_size=560 \
            tokenizer.image_size=560 \
            tokenizer.max_seq_len=8192 \
            tokenizer.path={llama_dir}/original/tokenizer.model \
            checkpointer.checkpoint_dir={llama_dir} \
            output_dir={result_dir}/{name}{postfix_result}"
        print(cmd)
        os.system(cmd)
        for test_csv in test_csv_list:
            for epoch in range(epochs):
                cmd=f"python tools/eval.py \
                    --gpu {GPU} \
                    --dataset {dataset} \
                    --filename {test_csv} \
                    --data_dir {data_dir} \
                    --model_id {result_dir}/{name}{postfix_result}/epoch_{epoch}"
                print(cmd)
                os.system(cmd)
```

## Requirements

```
conda create -n instfree python=3.10 -y
source activate instfree

# CUDA 12.1
pip install pip==24.0 setuptools==69.5.1 packaging==24.0 numpy==1.26.2
pip install torch==2.5.1 torchvision==0.20.1 torchao==0.7.0 --index-url https://download.pytorch.org/whl/cu121

pip install torchtune==0.6.1
pip uninstall torchtune -y
pip install bitsandbytes

pip install transformers==4.47.1
pip install accelerate
pip install pandas nltk
pip install "git+https://github.com/salaniz/pycocoevalcap.git"
```

## Acknowledgements

Thanks to works below for their implementations which were useful for this work.
[torchtune](https://github.com/pytorch/torchtune)