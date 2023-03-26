import os.path
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
import argparse

from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from SENN.aggregators import additive_scalar_aggregator
from SENN.arglist import get_senn_parser
from SENN.conceptizers import input_conceptizer, image_cnn_conceptizer, image_fcc_conceptizer, image_resnet_conceptizer
from SENN.models import GSENN
from SENN.parametrizers import image_parametrizer, vgg_parametrizer, torchvision_parametrizer
from SENN.trainers import VanillaClassTrainer, GradPenaltyTrainer, CLPenaltyTrainer
from SENN.utils import generate_dir_names, concept_grid

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns


def analyze_concepts(concept_batches, num_concepts):
    concept_batches_to_2d = concept_batches.view(-1, num_concepts)
    concept_batches_to_2d = concept_batches_to_2d.detach().numpy()
    df = pd.DataFrame(concept_batches_to_2d)
    sns.heatmap(df.corr().round(2), annot=True)


def parse_args():
    senn_parser = get_senn_parser()

    ### Local ones
    parser = argparse.ArgumentParser(parents=[senn_parser], add_help=False,
                                     description='Interpteratbility robustness evaluation on MNIST')

    # #setup
    parser.add_argument('-d', '--datasets', nargs='+',
                        default=['heart', 'ionosphere', 'breast-cancer', 'wine', 'heart',
                                 'glass', 'diabetes', 'yeast', 'leukemia', 'abalone'], help='<Required> Set flag')
    parser.add_argument('--lip_calls', type=int, default=10,
                        help='ncalls for bayes opt gp method in Lipschitz estimation')
    parser.add_argument('--lip_eps', type=float, default=0.01,
                        help='eps for Lipschitz estimation')
    parser.add_argument('--lip_points', type=int, default=100,
                        help='sample size for dataset Lipschitz estimation')
    parser.add_argument('--optim', type=str, default='gp',
                        help='black-box optimization method')

    #####

    args = parser.parse_args()

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    return args


def get_duplicates(x, df):
    unique_list = list(df['lesion_id'])
    if x in unique_list:
        return 'unduplicated'
    else:
        return 'duplicated'


def get_val_rows(x, df):
    # create a list of all the lesion_id's in the val set
    val_list = list(df['image_id'])
    if str(x) in val_list:
        return 'val'
    else:
        return 'train'


class ham10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __getitem__(self, index):
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform is not None:
            X = self.transform(X)

        return X, y

    def __len__(self):
        return len(self.df)


# https://www.kaggle.com/code/xinruizhuang/skin-lesion-classification-acc-90-pytorch/notebook
def load_ham10000_data(valid_size=0.1, shuffle=True, resize=None, random_seed=2008, batch_size=64, num_workers=1):
    # create ham10000 dataset
    ham10000_dir = os.path.abspath(os.path.join('/data', 'HAM10000', 'dataverse_files'))
    # my file structure
    # - data/HAM10000/dataverse_files
    #   - HAM10000_images_part_1
    #       - ISIC_*.jpg
    #   - HAM10000_images_part_2
    #       - ISIC_*.jpg
    #   - HAM10000_metadata
    all_image_path = glob(os.path.join(ham10000_dir, '*', '*.jpg'))
    # image_id_path_dict will be like this
    # {ISIC_* : 'data/HAM10000/dataverse_files/*/*.jpg', ...}
    image_id_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}

    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'dermatofibroma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    # original columns in HAM10000_metadata
    # [lesion_id, image_id, dx, dx_type, age, sex, localization, dataset] (8 columns)
    df_original = pd.read_csv(os.path.join(ham10000_dir, 'HAM10000_metadata'))
    # add new columns
    df_original['path'] = df_original['image_id'].map(image_id_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes

    # drop duplicated lesion, get unique lesion_id
    df_unduplicated = df_original.groupby('lesion_id').count()
    df_unduplicated = df_unduplicated[df_unduplicated['image_id'] == 1]
    df_unduplicated.reset_index(inplace=True)

    # create a new colum that is a copy of the lesion_id column
    df_original['duplicates'] = df_original['lesion_id']
    # apply the function to this new column
    df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates, df=df_unduplicated)

    # now we filter out images that don't have duplicates
    df_unduplicated = df_original[df_original['duplicates'] == 'unduplicated']

    # check df_unduplicated
    # print(f'shape: {df_unduplicated.shape}')

    y = df_unduplicated['cell_type_idx']
    # use stratify to keep ratio the same
    _, df_test = train_test_split(df_unduplicated, test_size=0.2, random_state=random_seed, stratify=y)

    # check test set
    # print(df_test['cell_type_idx'].value_counts())

    '''
    check df_test label distribution
    4    883
    2     88
    6     46
    1     35
    0     30
    5     13
    3      8
    '''
    # print(df_test['cell_type_idx'].value_counts())

    # create a new colum that is a copy of the image_id column
    df_original['train_or_val'] = df_original['image_id']
    # apply the function to this new column
    df_original['train_or_val'] = df_original['train_or_val'].apply(get_val_rows, df=df_test)
    # filter out train rows
    df_train = df_original[df_original['train_or_val'] == 'train']

    # balance the data by resampling
    data_aug_rate = [15, 10, 5, 50, 0, 40, 5]
    for i in range(7):
        if data_aug_rate[i]:
            df_train = df_train.append([df_train.loc[df_train['cell_type_idx'] == i, :]] * (data_aug_rate[i] - 1),
                                       ignore_index=True)

    '''
    check df_train label distribution
    Melanocytic nevi                  5822
    Dermatofibroma                    5350
    dermatofibroma                    5335
    Vascular lesions                  5160
    Benign keratosis-like lesions     5055
    Basal cell carcinoma              4790
    Actinic keratoses                 4455
    '''
    # print(df_train['cell_type'].value_counts())

    # split train into train and valid
    df_train, df_valid = train_test_split(df_train, test_size=0.1, random_state=random_seed)
    df_train = df_train.reset_index()
    df_valid = df_valid.reset_index()
    df_test = df_test.reset_index()

    # mean of the dataset (RGB) = [0.7630331, 0.5456457, 0.5700467]
    # std  of the dataset (RGB) = [0.1409281, 0.15261227, 0.16997086]
    transf_seq = [
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7630331, 0.5456457, 0.5700467],
                             std=[0.1409281, 0.15261227, 0.16997086])
    ]

    # if resize and (resize[0] != 32 or resize[1] != 32):
    #     transf_seq.insert(0, transforms.Resize(resize))

    # all transform operations
    transform_train = transforms.Compose(transf_seq)

    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7630331, 0.5456457, 0.5700467],
                             std=[0.1409281, 0.15261227, 0.16997086])
    ])

    # create datasets
    train_dataset = ham10000(df_train, transform=transform_train)
    valid_dataset = ham10000(df_valid, transform=transform_train)
    test_dataset = ham10000(df_test, transform=transform_test)

    # create dataloaders
    dataloader_args = dict(batch_size=batch_size, num_workers=num_workers)
    train_loader = DataLoader(train_dataset, **dataloader_args)
    valid_loader = DataLoader(valid_dataset, **dataloader_args)
    dataloader_args['shuffle'] = False
    test_loader = DataLoader(test_dataset, **dataloader_args)

    return train_loader, valid_loader, test_loader


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.nclasses = 7
    args.theta_dim = args.nclasses

    if (args.theta_arch == 'simple') or ('vgg' in args.theta_arch):
        H, W = 32, 32
    else:
        # Need to resize to have access to torchvision's models
        H, W = 224, 224

    args.input_dim = H * W

    model_path, log_path, results_path = generate_dir_names('ham10000', args)

    train_loader, valid_loader, test_loader = load_ham10000_data(
        batch_size=args.batch_size, num_workers=args.num_workers, resize=(H, W)
    )

    # choose conceptizer
    if args.h_type == 'input':
        conceptizer = input_conceptizer()
        args.nconcepts = args.input_dim + int(not args.nobias)
    elif args.h_type == 'cnn':
        # conceptizer = image_cnn_conceptizer(args.input_dim, args.nconcepts, args.concept_dim,
        #                                     nchannel=3)  # , sparsity = sparsity_l)
        conceptizer = image_resnet_conceptizer(args.input_dim, args.nconcepts, args.nclasses, args.concept_dim,
                                               nchannel=3)
    else:
        conceptizer = image_fcc_conceptizer(args.input_dim, args.nconcepts, args.concept_dim,
                                            nchannel=3)  # , sparsity = sparsity_l)

    # choose Parametrizer architecture
    if args.theta_arch == 'simple':
        parametrizer = image_parametrizer(args.input_dim, args.nconcepts, args.theta_dim, nchannel=3,
                                          only_positive=args.positive_theta)
    elif 'vgg' in args.theta_arch:
        parametrizer = vgg_parametrizer(args.input_dim, args.nconcepts, args.theta_dim, arch=args.theta_arch,
                                        nchannel=3,
                                        only_positive=args.positive_theta)
    else:
        parametrizer = torchvision_parametrizer(args.input_dim, args.nconcepts, args.theta_dim, arch=args.theta_arch,
                                                nchannel=3,
                                                only_positive=args.positive_theta)

    # set aggregator to combine concepts and parameters
    aggregator = additive_scalar_aggregator(args.nconcepts, args.concept_dim, args.nclasses)

    model = GSENN(conceptizer, aggregator)

    # Type of regularization on theta
    if args.theta_reg_type in ['unreg', 'none', None]:
        trainer = VanillaClassTrainer(model, args)
    elif args.theta_reg_type == 'grad1':
        trainer = GradPenaltyTrainer(model, args, typ=1)
    elif args.theta_reg_type == 'grad2':
        trainer = GradPenaltyTrainer(model, args, typ=2)
    elif args.theta_reg_type == 'grad3':
        trainer = GradPenaltyTrainer(model, args, typ=3)
    elif args.theta_reg_type == 'crosslip':
        trainer = CLPenaltyTrainer(model, args)
    else:
        raise ValueError('Unrecoginzed theta_reg_type')

    # if choose train or not model can be found in path
    if args.train or not args.load_model or (not os.path.isfile(os.path.join(model_path, 'model_best.pth.tar'))):
        trainer.train(train_loader, valid_loader, epochs=args.epochs, save_path=model_path)
        trainer.plot_losses(save_path=results_path)
    else:
        checkpoint = torch.load(os.path.join(model_path, 'model_best.pth.tar'),
                                map_location=lambda storage, loc: storage)
        checkpoint.keys()
        model = checkpoint['model']
        trainer = VanillaClassTrainer(model, args)  # arbtrary trained, only need to compuyte val acc

    model.eval()

    # Check accuracy with the best model
    checkpoint = torch.load(os.path.join(model_path, 'model_best.pth.tar'), map_location=lambda storage, loc: storage)
    checkpoint.keys()
    model = checkpoint['model']
    trainer = VanillaClassTrainer(model, args)
    # analyze_concepts(model.concepts, args.nconcepts)
    trainer.evaluate(test_loader, fold='test')

    concept_grid(model, test_loader, top_k=10, cuda=args.cuda, save_path=results_path + '/concept_grid.pdf')


if __name__ == '__main__':
    main()
