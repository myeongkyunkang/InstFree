import argparse
import os
import random

import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
from PIL import Image
from tqdm import tqdm

from eval_utils import *


def main(model, processor, args):
    csv_path = os.path.join(args.data_dir, args.filename)

    if args.dataset == 'wbcatt':
        option_dict = wbcatt_option_dict
    elif args.dataset == 'cbis':
        option_dict = cbis_option_dict
    elif args.dataset == 'skincon':
        option_dict = skin_option_dict
    else:
        raise ValueError()

    # read csv
    readData = pd.read_csv(csv_path)

    out_dict = {'image': [], 'gt': [], 'generated': [], 'predict': [], 'question': []}
    for index, row in tqdm(list(readData.iterrows())):
        image = Image.open(os.path.join(args.data_dir, row['image'])).convert('RGB').resize((560, 560))
        if args.dataset == 'skincon':
            gt_data = [('dermatologic feature', row['text'].strip()), ]
        else:
            gt_data = [(l.split(':')[0].strip(), l.split(':')[1].strip()) for l in row['text'].strip().split('\n')]

        for key, gt in gt_data:
            if ((args.dataset == 'cbis') and (key in is_cbis_keys)) or (args.dataset == 'skincon'):
                option_copy = list(option_dict[key])
                try:
                    option_copy.remove(gt)
                except:
                    pass
                options = [gt] + random.sample(option_copy, k=np.min([len(option_copy), 4]))
                random.shuffle(options)
            else:
                options = option_dict[key]

            if key == 'mass margins':  # TODO:
                key = 'mass margin'  # TODO:

            formatted_options_str, formatted_alphabet_str = format_options(options)
            question = f'What is the {key}?\n{formatted_options_str}\nResponse with only the letter of the correct choice, starting with "The correct answer is [{formatted_alphabet_str}].".'

            # run
            if args.model == 'llama':
                output = run_llama(question, image, model, processor, args.max_new_tokens)

            try:
                generated = output.split(f'The correct answer is')[-1]
                generated = generated.replace('<|eot_id|>', '').replace('.', '').replace('*', '').replace('[', '').replace(']', '').strip()  # clean
                predict = int(ALPHABET_LIST.index(generated.upper()) == options.index(gt))
            except:
                generated = output
                predict = 0

            out_dict['image'].append(row['image'])
            out_dict['gt'].append(gt)
            out_dict['generated'].append(generated)
            out_dict['predict'].append(predict)
            out_dict['question'].append(question)

    name = os.path.splitext(args.filename)[0] + '_' + os.path.splitext(os.path.basename(args.model_id))[0]

    # save response
    acc = np.mean(out_dict['predict'])
    out_path = os.path.join(args.result_dir, f'{name}_{round(acc, 4)}.csv')
    pd.DataFrame(out_dict).to_csv(out_path, index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='wbcatt')
    parser.add_argument('--model_id', default='./models/llama3/Llama-3.2-11B-Vision-Instruct')
    parser.add_argument('--data_dir', default='./datasets/concept/wbcatt')
    parser.add_argument('--model', default='llama')
    parser.add_argument('--result_dir', default='')
    parser.add_argument('--filename', default='image_text_test_att1.csv')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    if args.result_dir == '':
        args.result_dir = os.path.dirname(args.model_id)
    os.makedirs(args.result_dir, exist_ok=True)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    import torch

    # create model and processor
    if args.model == 'llama':
        from transformers import MllamaForConditionalGeneration, AutoProcessor

        model = MllamaForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map='auto')
        processor = AutoProcessor.from_pretrained(args.model_id)
    else:
        raise ValueError()

    main(model, processor, args)
