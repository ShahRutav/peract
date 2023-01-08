import os
import argparse
from pathlib import Path

import torch
import transformers

from helpers import fit
from helpers.fit import parse_config
import helpers.fit.model as module_arch
from helpers.fit.utils.util import state_dict_data_parallel_fix

def build_model(config_path="config.json"):
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default=Path(os.path.join(fit.__path__[0], 'config.json')), type=str,
                      help='config file path (default: None)')
    args.add_argument('-s', '--sliding_window_stride', default=-1, type=int,
                      help='test time temporal augmentation, repeat samples with different start times.')
    args.add_argument('--save_feats', default=None,
                      help='path to store text & video feats, this is for saving embeddings if you want to do offline retrieval.')
    args.add_argument('--save_type', default='both', choices=['both', 'text', 'video'],
                      help='Whether to save video, text or both feats. If running on inference videos, text is just a placeholder')
    args.add_argument('--vis_token_similarity', action='store_true')
    args.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                      help='split to evaluate on.')
    args.add_argument('--batch_size', default=16, type=int,
                      help='size of batch')
    config = parse_config.ConfigParser(args)
    # hack to get sliding into config
    args = args.parse_args()
    config._config['sliding_window_stride'] = args.sliding_window_stride
    config._config['resume'] = "/home/rutavms/models/frozen-4frames-pretrain.tar" ## TODO: change this behavior
    config._config['data_loader'][0]['args']['split'] = args.split
    config._config['data_loader'][0]['args']['tsfm_split'] = 'test'  # set transform to test split to remove augmentations
    config._config['data_loader'][0]['args']['shuffle'] = False
    config._config['data_loader'][0]['args']['batch_size'] = args.batch_size
    config._config['data_loader'][0]['args']['sliding_window_stride'] = args.sliding_window_stride # Test time augmentatiot, repeat samples with different start times

    text_model_name = config['arch']['args']['text_params']['model']
    if "openai/clip" in text_model_name:
        tokenizer_builder = transformers.CLIPTokenizer
    else:
        tokenizer_builder = transformers.AutoTokenizer
    tokenizer = tokenizer_builder.from_pretrained(
        text_model_name,
        model_max_length=config['arch']['args']['text_params'].get('max_length', 1e6), # If missing, sets to 1e6
        TOKENIZERS_PARALLELISM=False)

    # build model architecture
    model = config.initialize('arch', module_arch)

    # logger.info('Loading checkpoint: {} ...'.format(config.resume))
    assert config._config['resume'] is not None, "Please provide model_checkpoint path using --resume flag"
    checkpoint = torch.load(config._config['resume'])
    state_dict = checkpoint['state_dict']
    new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
    model.load_state_dict(new_state_dict, strict=True)

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()



def main():
    build_model()

if __name__ == '__main__':
    main()

