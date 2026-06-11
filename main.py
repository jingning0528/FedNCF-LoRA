#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
import logging
from datetime import datetime
from framework.utils import load_config, set_logger, print_to_json, print_to_list
from framework.modules.utils import seed_everything
import zoo as model_zoo
import dataloaders as dataload_zoo
import gc
import argparse
import os
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='DeepFM_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    parser.add_argument('--seed', type=int, default=None, help='Override seed from config')
    parser.add_argument('--result_file', type=str, default='', help='Output csv file path')
    parser.add_argument('--save_csv', action='store_true', help='Save result summary to csv')
    args = vars(parser.parse_args())

    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)

    if args['seed'] is not None:
        params['seed'] = args['seed']

    params['device'] = args['gpu']
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    dataload_class = getattr(dataload_zoo, params['dataloader'])
    dataload = dataload_class(**params)

    model_class = getattr(model_zoo, params['model'])
    model = model_class(dataload=dataload, **params)
    test_results = model.fit()

    # only save csv when explicitly requested
    if args['save_csv']:
        if args['result_file']:
            out_path = Path(args['result_file'])
        else:
            result_filename = Path(args['config']).name.replace(".yaml", "") + '.csv'
            out_path = Path(result_filename)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'a+', encoding='utf-8') as fw:
            fw.write(
                ' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[test] {}\n'.format(
                    datetime.now().strftime('%Y%m%d-%H%M%S'),
                    ' '.join(sys.argv), experiment_id, params['dataset_id'],
                    "N.A.", print_to_list(test_results)
                )
            )