import os
import random

import pandas as pd
from PIL import Image
from tqdm import tqdm


def main_wbcatt():
    random.seed(0)

    wbcatt_dir = './datasets/concept/wbcatt/'
    wbcatt_train_csv_path = os.path.join(wbcatt_dir, 'pbc_attr_v1_train.csv')
    wbcatt_val_csv_path = os.path.join(wbcatt_dir, 'pbc_attr_v1_val.csv')
    wbcatt_test_csv_path = os.path.join(wbcatt_dir, 'pbc_attr_v1_test.csv')

    wbcatt_att_key_list = ['cell_size', 'cell_shape',
                           'nucleus_shape', 'nuclear_cytoplasmic_ratio', 'chromatin_density',
                           'cytoplasm_vacuole', 'cytoplasm_texture', 'cytoplasm_colour',
                           'granularity', 'granule_type', 'granule_colour',
                           'label']
    wbcatt_id_key = 'path'

    # img_name,label,cell_size,cell_shape,nucleus_shape,nuclear_cytoplasmic_ratio,chromatin_density,cytoplasm_vacuole,cytoplasm_texture,cytoplasm_colour,granule_type,granule_colour,granularity,path
    # BNE_422954.jpg,Neutrophil,small,round,unsegmented-band,low,densely,no,clear,light blue,small,pink,yes,PBC_dataset_normal_DIB/neutrophil/BNE_422954.jpg

    def process_wbcatt(csv_path, save_filename, test_val=False):
        out_dict = {'image': [], 'text': []}
        for index, row in list(pd.read_csv(csv_path).iterrows()):
            id = row[wbcatt_id_key]
            _att_key_list = list(wbcatt_att_key_list)
            _att_key_list = [key for key in _att_key_list if row[key] != 'nil']  # for 'no' granularity
            if test_val:
                att_num = 1  # TODO:
                random.shuffle(_att_key_list)
                _att_key_list = _att_key_list[:att_num]
                text = '\n'.join(f"{key.replace('_', ' ')}: {row[key]}" for key in _att_key_list)
            else:
                _text_list = []
                for key in _att_key_list:
                    _key, _value = key.replace('_', ' '), row[key]
                    if _value == 'no':
                        continue
                    elif _value == 'yes':
                        _text_list.append(f"{_key}")
                    elif _key == 'label':
                        _text_list.append(f"{_value}")
                    else:
                        _text_list.append(f"{_value} {_key}")
                text = ', '.join(_text_list)
            out_dict['image'].append(id)
            out_dict['text'].append(text)
        pd.DataFrame(out_dict).to_csv(os.path.join(wbcatt_dir, save_filename), index=False, encoding='utf-8-sig')

    # build val and test
    process_wbcatt(wbcatt_val_csv_path, 'image_text_val_att1.csv', test_val=True)
    process_wbcatt(wbcatt_test_csv_path, 'image_text_test_att1.csv', test_val=True)

    # build train
    process_wbcatt(wbcatt_train_csv_path, 'image_text.csv')
    process_wbcatt(wbcatt_val_csv_path, 'image_text_val.csv')
    process_wbcatt(wbcatt_test_csv_path, 'image_text_test.csv')


def main_cbis():
    random.seed(0)

    cbis_dir = './datasets/concept/cbis/'
    train_mass_csv_path = os.path.join(cbis_dir, 'csv', 'mass_case_description_train_set.csv')
    test_mass_csv_path = os.path.join(cbis_dir, 'csv', 'mass_case_description_test_set.csv')
    train_calc_csv_path = os.path.join(cbis_dir, 'csv', 'calc_case_description_train_set.csv')
    test_calc_csv_path = os.path.join(cbis_dir, 'csv', 'calc_case_description_test_set.csv')
    dicom_csv_path = os.path.join(cbis_dir, 'csv', 'dicom_info.csv')

    convert_image = False
    val_rate = 0.1

    image_type = 'cropped images'  # 'full mammogram images' is invalid.

    if image_type == 'cropped images':
        image_file_path_key = 'cropped image file path'
    elif image_type == 'full mammogram images':
        image_file_path_key = 'image file path'
    else:
        raise ValueError()

    # label list
    subtlety_label = ['very obvious', 'obvious', 'moderate', 'subtle', 'very subtle']
    assessment_label = ['incomplete', 'negative', 'benign finding', 'probably benign', 'suspicious abnormality', 'highly suggestive of malignancy']
    density_label = ['incomplete', 'almost entirely fatty', 'scattered areas of fibroglandular density', 'heterogeneously dense', 'extremely dense']

    # key list
    mass_keys = ['mass shape', 'mass margins']
    calc_keys = ['calc type', 'calc distribution']
    if image_type == 'cropped images':
        general_keys = ['abnormality type', 'pathology']  # 'breast density', 'assessment', 'subtlety'
    elif image_type == 'full mammogram images':
        general_keys = ['image view', 'abnormality type', 'pathology']  # 'breast density','assessment', 'subtlety'

    test_key_dict = {
        'mass shape': [
            'architectural distortion',  # 23 (Test)
            'asymmetric breast tissue',  # 5
            'focal asymmetric density',  # 6
            'irregular',  # 113
            'lobulated',  # 79
            'lymph node',  # 9
            'oval',  # 91
            'round',  # 41
        ],
        'mass margins': [
            'circumscribed',  # 87
            'ill defined',  # 92
            'microlobulated',  # 21
            'obscured',  # 50
            'spiculated',  # 82
        ],
        'calc type': [
            'amorphous',  # 43
            'coarse',  # 4
            'eggshell',  # 6
            'fine linear branching',  # 25
            'lucent center',  # 17
            'pleomorphic',  # 149
            'punctate',  # 26
            'round and regular',  # 10
            'vascular',  # 8
        ],
        'calc distribution': [
            'clustered',  # 195
            'diffusely scattered',  # 3
            'linear',  # 22
            'regional',  # 3
            'segmental',  # 34
        ],
    }

    # read csv
    train_mass_list = list(pd.read_csv(train_mass_csv_path).iterrows())
    test_mass_list = list(pd.read_csv(test_mass_csv_path).iterrows())
    train_calc_list = list(pd.read_csv(train_calc_csv_path).iterrows())
    test_calc_list = list(pd.read_csv(test_calc_csv_path).iterrows())

    # read images
    image_path_dict = {}
    for index, row in pd.read_csv(dicom_csv_path).iterrows():
        PatientID = row['PatientID']
        SeriesDescription = row['SeriesDescription']
        image_path = row['image_path']
        if SeriesDescription != image_type:
            continue
        if PatientID in image_path_dict:
            raise ValueError()
        image_path_dict[PatientID] = image_path.replace('CBIS-DDSM/', '')

    if image_type == 'full mammogram images':
        # select images with one abnormality
        exclude_list = []
        for index, row in train_mass_list + test_mass_list + train_calc_list + test_calc_list:
            if row['abnormality id'] > 1:
                image_key = row['image file path'].split('/')[0]
                exclude_list.append(image_key)

        # exclude images
        for key in exclude_list:
            image_path_dict.pop(key, None)

    # convert image
    if convert_image:
        os.makedirs(os.path.join(cbis_dir, 'jpeg560'), exist_ok=True)
    _image_path_dict = {}
    for k, image_filename in tqdm(list(image_path_dict.items())):
        new_image_filename = f'jpeg560/{k}.jpg'
        if convert_image:
            image = Image.open(os.path.join(cbis_dir, image_filename)).convert('RGB')
            target_size = 560
            w, h = image.size
            ratio = min(target_size / w, target_size / h)
            new_size = (int(w * ratio), int(h * ratio))
            image = image.resize(new_size, Image.LANCZOS)
            new_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
            new_image.paste(image, ((target_size - image.size[0]) // 2, (target_size - image.size[1]) // 2))
            new_image.save(os.path.join(cbis_dir, new_image_filename))
        _image_path_dict[k] = new_image_filename
    image_path_dict = _image_path_dict

    # make validation
    train_val_patient_id_list = []
    for key in image_path_dict.keys():
        if key.startswith('Mass-Test') or key.startswith('Calc-Test'):  # Mass-Training,Mass-Test,Calc-Training,Calc-Test
            continue
        train_val_patient_id_list.append('P_' + key.split('_')[2])
    train_val_patient_id_list = sorted(list(set(train_val_patient_id_list)))  # unique
    random.Random(0).shuffle(train_val_patient_id_list)
    val_patient_id_list = train_val_patient_id_list[:int(len(train_val_patient_id_list) * val_rate)]
    train_patient_id_list = train_val_patient_id_list[int(len(train_val_patient_id_list) * val_rate):]

    train_list, val_list, test_list = [], [], []
    for index, row in train_mass_list + test_mass_list:
        patient_id = row['patient_id']
        image_view = row['image view']
        image_key = row[image_file_path_key].split('/')[0]
        if image_key not in image_path_dict:
            continue

        image_path = image_path_dict[image_key]

        text_dict = {}
        if image_type == 'full mammogram images':
            text_dict['image view'] = image_view
        text_dict['abnormality type'] = row['abnormality type'].lower().replace('_', ' ')
        text_dict['pathology'] = row['pathology'].lower().replace('_', ' ').replace(' without callback', '')  # TODO:
        if row['breast_density'] != 0:
            text_dict['breast density'] = density_label[row['breast_density']]  # 1~4,0(=incomplete)
        text_dict['assessment'] = assessment_label[row['assessment']]  # BI-RADS assessment from 0 to 5
        text_dict['subtlety'] = subtlety_label[row['subtlety'] - 1]  # 1~5

        if type(row['mass shape']) is str:
            mass_shape = row['mass shape'].lower().replace('_', ' ').strip()
        else:
            mass_shape = None
        text_dict['mass shape'] = mass_shape

        if type(row['mass margins']) is str:
            mass_margins = row['mass margins'].lower().replace('_', ' ').strip()
        else:
            mass_margins = None
        text_dict['mass margins'] = mass_margins

        if image_key.startswith('Mass-Training_'):
            if patient_id in val_patient_id_list:
                val_list.append((image_path, text_dict))
            elif patient_id in train_patient_id_list:
                train_list.append((image_path, text_dict))
            else:
                raise ValueError('')
        elif image_key.startswith('Mass-Test_'):
            test_list.append((image_path, text_dict))
        else:
            raise ValueError('')

    for index, row in train_calc_list + test_calc_list:
        patient_id = row['patient_id']
        image_view = row['image view']
        image_key = row[image_file_path_key].split('/')[0]
        if image_key not in image_path_dict:
            continue

        image_path = image_path_dict[image_key]

        text_dict = {}
        if image_type == 'full mammogram images':
            text_dict['image view'] = image_view
        text_dict['abnormality type'] = row['abnormality type'].lower().replace('_', ' ')
        text_dict['pathology'] = row['pathology'].lower().replace('_', ' ').replace(' without callback', '')  # TODO:
        if row['breast density'] != 0:
            text_dict['breast density'] = density_label[row['breast density']]  # 1~4,0(=incomplete)
        text_dict['assessment'] = assessment_label[row['assessment']]  # BI-RADS assessment from 0 to 5
        text_dict['subtlety'] = subtlety_label[row['subtlety'] - 1]  # 1~5

        if type(row['calc type']) is str:
            calc_type = row['calc type'].lower().replace('_', ' ').strip()
        else:
            calc_type = None
        text_dict['calc type'] = calc_type

        if type(row['calc distribution']) is str:
            calc_distribution = row['calc distribution'].lower().replace('_', ' ').strip()
        else:
            calc_distribution = None
        text_dict['calc distribution'] = calc_distribution

        if image_key.startswith('Calc-Training_'):
            if patient_id in val_patient_id_list:
                val_list.append((image_path, text_dict))
            elif patient_id in train_patient_id_list:
                train_list.append((image_path, text_dict))
            else:
                raise ValueError('')
        elif image_key.startswith('Calc-Test_'):
            test_list.append((image_path, text_dict))
        else:
            raise ValueError('')

    data_dict = {}
    for image_path, text_dict in train_list + val_list + test_list:
        for k, v in text_dict.items():
            if k not in data_dict:
                data_dict[k] = []
            data_dict[k].append(v)

    print('cbis_option_dict = {')
    for k, v in data_dict.items():
        # print(f"'{k}' :", sorted(set([_ for _ in v if _ is not None])), ',')
        from collections import Counter
        print(f"'{k}' :", Counter(v))
    print('}')

    print(len(train_list), len(val_list), len(test_list))

    def build_csv(data_list, save_filename):
        out_dict = {'image': [], 'text': []}
        for image, text_dict in data_list:
            out_dict['image'].append(image)
            _text_list = []
            for key in mass_keys + calc_keys:
                if key not in text_dict:
                    continue
                value = text_dict[key]
                if value is None:
                    continue
                _text_list.append(f'{value} {key}')
            for key in general_keys:
                _text_list.append(text_dict[key])  # only value
            out_dict['text'].append(', '.join(_text_list))
        pd.DataFrame(out_dict).to_csv(os.path.join(cbis_dir, save_filename), index=False, encoding='utf-8-sig')

    def build_att1_csv(data_list, save_filename):
        out_dict = {'image': [], 'text': []}
        for image, text_dict in data_list:
            abnormality_keys = mass_keys if text_dict['abnormality type'] == 'mass' else calc_keys
            for key in abnormality_keys:
                if key not in text_dict:
                    continue
                value = text_dict[key]
                if value is None:
                    continue
                if value not in test_key_dict[key]:
                    # print('skip:', key, value)
                    continue
                out_dict['image'].append(image)
                out_dict['text'].append(f'{key}: {value}')
            for key in general_keys:
                if key not in text_dict:
                    continue
                value = text_dict[key]
                if value is None:
                    continue
                out_dict['image'].append(image)
                out_dict['text'].append(f'{key}: {value}')
        pd.DataFrame(out_dict).to_csv(os.path.join(cbis_dir, save_filename), index=False, encoding='utf-8-sig')

    build_att1_csv(val_list, os.path.join(cbis_dir, 'image_text_val_att1.csv'))
    build_att1_csv(test_list, os.path.join(cbis_dir, 'image_text_test_att1.csv'))

    # TODO: we increase the size of training to match the scale to match (x3)
    build_csv(train_list * 3, os.path.join(cbis_dir, 'image_text.csv'))
    build_csv(val_list, os.path.join(cbis_dir, 'image_text_val.csv'))
    build_csv(test_list, os.path.join(cbis_dir, 'image_text_test.csv'))


def main_skincon():
    random.seed(0)

    skin_dir = './datasets/concept/skincon/'
    image_dir = os.path.join(skin_dir, 'fitzpatrick17k')
    anno_csv_path = os.path.join(skin_dir, 'fitzpatrick17k_anno.csv')

    image_560_dirname = 'fitzpatrick17k_560'
    os.makedirs(os.path.join(skin_dir, image_560_dirname), exist_ok=True)
    convert_image = False
    test_rate = 0.1

    data_dict = {}

    # read csv
    df = pd.read_csv(anno_csv_path)
    for _, row in df.iterrows():
        image_id = row['ImageID']
        labels = row[row == 1].index.tolist()
        labels = [label for label in labels if label != 'ImageID']  # exclude 'ImageID' from the list of labels
        if 'Do not consider this image' in labels:
            continue
        labels = [l.lower().strip() for l in labels]
        data_dict[image_id] = labels
    print([l.lower() for l in row.index.tolist()])

    # read images
    image_path_dict = {f: os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))}

    inter_keys = set(image_path_dict.keys()) & set(data_dict.keys())
    image_path_dict = {k: image_path_dict[k] for k in inter_keys}
    data_dict = {k: data_dict[k] for k in inter_keys}

    _data_dict = {}
    for k, text_list in tqdm(list(data_dict.items())):
        image_path = image_path_dict[k]
        new_k = f'{image_560_dirname}/{k}'
        new_image_path = os.path.join(skin_dir, new_k)
        if convert_image:
            image = Image.open(image_path).convert('RGB')
            target_size = 560
            w, h = image.size
            ratio = min(target_size / w, target_size / h)
            new_size = (int(w * ratio), int(h * ratio))
            image = image.resize(new_size, Image.LANCZOS)
            new_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
            new_image.paste(image, ((target_size - image.size[0]) // 2, (target_size - image.size[1]) // 2))
            new_image.save(new_image_path)
        _data_dict[new_k] = text_list
    data_dict = _data_dict

    print(sorted(list(set([', '.join(v) for v in data_dict.values()]))))

    train_data_list = sorted([(k, v) for k, v in data_dict.items()])
    random.Random(0).shuffle(train_data_list)
    test_val_size = int(len(train_data_list) * test_rate)
    val_data_list = train_data_list[:test_val_size]
    test_data_list = train_data_list[test_val_size:test_val_size * 2]
    train_data_list = train_data_list[test_val_size * 2:]

    def build_csv(data_list, save_filename):
        out_dict = {'image': [], 'text': []}
        for image, text_list in data_list:
            out_dict['image'].append(image)
            out_dict['text'].append(', '.join(text_list))
        pd.DataFrame(out_dict).to_csv(os.path.join(skin_dir, save_filename), index=False, encoding='utf-8-sig')

    build_csv(val_data_list, os.path.join(skin_dir, 'image_text_val_att1.csv'))
    build_csv(test_data_list, os.path.join(skin_dir, 'image_text_test_att1.csv'))

    # TODO: we increase the size of training to match the scale to match (x3)
    build_csv(train_data_list + train_data_list + train_data_list, os.path.join(skin_dir, 'image_text.csv'))
    build_csv(val_data_list, os.path.join(skin_dir, 'image_text_val.csv'))
    build_csv(test_data_list, os.path.join(skin_dir, 'image_text_test.csv'))


if __name__ == '__main__':
    dataset = ''
    if dataset == 'wbcatt':
        main_wbcatt()
    elif dataset == 'cbis':
        main_cbis()
    elif dataset == 'skincon':
        main_skincon()
