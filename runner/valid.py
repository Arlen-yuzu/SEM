import sys
import json
sys.path.append('./')
sys.path.append('../')
import os
import cv2
import tqdm
import torch
import numpy as np
from libs.configs import cfg, setup_config
from libs.model import build_model
from libs.data import create_valid_dataloader
from libs.utils import logger
from libs.utils.cal_f1 import pred_result_to_table, table_to_relations, evaluate_f1
from libs.utils.checkpoint import load_checkpoint
from libs.utils.comm import synchronize, all_gather
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def init():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lrc", type=str, default='/shared/aia/alg/xyl/tsrdataset/model/sem/icdar13/test.lrc')
    parser.add_argument("--cfg", type=str, default='default')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--valid_data_dir", type=str, default='/shared/aia/alg/xyl/tsrdataset/unify/icdar13/image')
    parser.add_argument("--vis_dir", type=str, default='vis_icdar13')
    args = parser.parse_args()
    
    setup_config(args.cfg)
    if args.lrc is not None:
        cfg.valid_lrc_path = args.lrc
    if args.valid_data_dir is not None:
        cfg.valid_data_dir = args.valid_data_dir
    if args.vis_dir is not None:
        cfg.vis_dir = args.vis_dir
    
    if not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir)

    os.environ['LOCAL_RANK'] = str(args.local_rank)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    logger.setup_logger('Line Detect Model', cfg.work_dir, 'valid.log')
    logger.info('Use config: %s' % args.cfg)
    logger.info('Evaluate Dataset: %s' % cfg.valid_lrc_path)


def valid(cfg, dataloader, model):
    model.eval()
    total_label_relations = list()
    total_pred_relations = list()
    total_relations_metric = list()
    
    for it, data_batch in enumerate(tqdm.tqdm(dataloader)):
        ids = data_batch['ids']
        images_size = data_batch['images_size']
        images = data_batch['images'].to(cfg.device)
        tables = data_batch['tables']
        
        pred_result, _ = model(images, images_size)

        # pred
        pred_tables = [
            pred_result_to_table(tables[batch_idx],
                (pred_result[0][batch_idx], pred_result[1][batch_idx], \
                    pred_result[2][batch_idx], pred_result[3][batch_idx])
            ) \
            for batch_idx in range(len(ids))
        ]
        
        pred_bbox = []
        pred_logi = []
        for cell in pred_tables[0]['cells']:
            pred_bbox.append(cell['bbox'])
            pred_logi.append(cell['logi'])
        pred_bbox = np.array(pred_bbox)
        pred_logi = np.array(pred_logi)
        
        # img_name = tables[0]['image_path']
        # img = cv2.imread(os.path.join(cfg.valid_data_dir, img_name))
        
        # for bid, box in enumerate(pred_bbox):
        #     for j in range(0, len(box), 2):
        #         cv2.rectangle(img, (box[j], box[j + 1]), (box[(j + 2) % len(box)], box[(j + 3) % len(box)]), (0, 0, 255), 1) # 红色框为pred
        #     logi = pred_logi[bid]
        #     logi_txt = '{:.0f},{:.0f},{:.0f},{:.0f}'.format(logi[0], logi[1], logi[2], logi[3])
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     cat_size = cv2.getTextSize(logi_txt, font, 0.3, 2)[0]
        #     cv2.rectangle(img, (box[0], box[1]), (box[0] + cat_size[0], box[1] + cat_size[1]), (128,128,128), -1)
        #     cv2.putText(img, logi_txt, (int(box[0]), int(box[1] + cat_size[1])), font, 0.30, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        
        # cv2.imwrite(os.path.join(cfg.vis_dir, img_name), img)
        
        pred_relations = [table_to_relations(table) for table in pred_tables]        
        total_pred_relations.extend(pred_relations)
        
        # label
        label_relations = [table_to_relations(table) for table in tables]
        total_label_relations.extend(label_relations)

    # cal P, R, F1
    total_relations_metric = evaluate_f1(total_label_relations, total_pred_relations, num_workers=40)
    P, R, F1 = np.array(total_relations_metric).mean(0).tolist()
    F1 = 2 * P * R / (P + R)
    logger.info('[Valid] Total Type Mertric: Precision: %s, Recall: %s, F1-Score: %s' % (P, R, F1))

    return (F1, )


def main():
    init()

    valid_dataloader = create_valid_dataloader(
        cfg.vocab,
        cfg.valid_lrc_path,
        cfg.valid_num_workers,
        cfg.valid_batch_size,
        cfg.valid_data_dir
    )
    logger.info(
        'Valid dataset have %d samples, %d batchs with batch_size=%d' % \
            (
                len(valid_dataloader.dataset),
                len(valid_dataloader.batch_sampler),
                valid_dataloader.batch_size
            )
    )
    
    # for i, data_batch in enumerate(valid_dataloader):
    #     print(data_batch['tables'][0].keys())
    #     breakq

    model = build_model(cfg)
    model.cuda()
    
    # load_checkpoint(cfg.eval_checkpoint, model)
    # logger.info('Load checkpoint from: %s' % cfg.eval_checkpoint)

    with torch.no_grad():
        valid(cfg, valid_dataloader, model)


if __name__ == '__main__':
    main()
