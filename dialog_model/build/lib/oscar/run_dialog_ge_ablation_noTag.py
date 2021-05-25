# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.

from __future__ import absolute_import, division, print_function
import argparse
import base64
import os.path as op
import random, time, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from oscar.utils.logger import setup_logger
from oscar.utils.tsv_file import TSVFile
from oscar.utils.tsv_file_ops import tsv_writer
from oscar.utils.misc import (mkdir, set_seed, 
        load_from_yaml_file, find_file_path_in_yaml)
from oscar.utils.caption_evaluate import (evaluate_on_coco_caption, evaluate_on_reddit_caption,
        evaluate_on_nocaps, ScstRewardCriterion)
from oscar.utils.cbs import ConstraintFilter, ConstraintBoxesReader
from oscar.utils.cbs import FiniteStateMachineBuilder
from oscar.modeling.modeling_bert import BertForImageCaptioning
from transformers.pytorch_transformers import BertTokenizer, BertConfig
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
import torch.nn as nn
import sys
sys.setrecursionlimit(1000000)

total_num_boxes = 36
tmp_for_cat = torch.zeros((total_num_boxes, 6))



class CaptionTSVDataset(Dataset):
    def __init__(self, yaml_file, tokenizer=None, add_od_labels=True,
            max_img_seq_length=50, max_seq_length=70, max_seq_a_length=40, max_seq_c_length=40, 
            is_train=True, mask_prob=0.15,mask_response_prob=0.7, max_masked_tokens=3, **kwargs):
        """Constructor.
        Args:
            yaml file with all required data (image feature, caption, labels, etc)
            tokenizer: tokenizer for text processing.
            add_od_labels: whether to add labels from yaml file to BERT. 
            max_img_seq_length: max image sequence length.
            max_seq_length: max text sequence length.
            max_seq_a_length: max caption sequence length.
            is_train: train or test mode.
            mask_prob: probability to mask a input token.
            max_masked_tokens: maximum number of tokens to be masked in one sentence.
            kwargs: other arguments.
        """
        self.yaml_file = yaml_file
        self.cfg = load_from_yaml_file(yaml_file)
        self.root = op.dirname(yaml_file)
        self.tag_and_feat_file = find_file_path_in_yaml(self.cfg['feature'], self.root)
        # self.feat_file = find_file_path_in_yaml(self.cfg['feature'], self.root)
        self.dialog_file = find_file_path_in_yaml(self.cfg.get('dialog'), self.root)
        self.dialog2img_file = find_file_path_in_yaml(self.cfg.get('dialog2img'), self.root)

        assert op.isfile(self.tag_and_feat_file)
        # if add_od_labels: assert op.isfile(self.label_file)
        if is_train: assert op.isfile(self.dialog_file) and tokenizer is not None

        # self.label_tsv = None if not self.label_file else TSVFile(self.label_file)
        self.tag_and_feat_tsv = TSVFile(self.tag_and_feat_file, gen_lineidx=True)
        if self.dialog_file and op.isfile(self.dialog_file):
            self.dialogs = [line.decode().rstrip('\n').split('\t') for line in open(self.dialog_file,'rb')] # [[sent1,sent2,....sentn],...[sent1,sent2,....sentn]]
        if self.dialog2img_file and op.isfile(self.dialog2img_file):
            self.dialog2img = [line.decode().rstrip('\n').split('\t') for line in open(self.dialog2img_file,'rb')] # [[top1,top2,....topn],...[top1,top2,....topn]] 

        # print('len of dialogs', len(self.dialogs))


        self.tokenizer = tokenizer
        self.tensorizer = CaptionTensorizer(self.tokenizer, max_img_seq_length,
                max_seq_length, max_seq_a_length, max_seq_c_length, mask_prob, mask_response_prob, max_masked_tokens,
                is_train=is_train)
        self.add_od_labels = add_od_labels
        self.is_train = is_train
        self.kwargs = kwargs
        # self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        # self.key2captions = self.prepare_image_key_to_captions()

    def get_valid_tsv(self):
        return self.tag_and_feat_tsv

    # def prepare_image_keys(self):
    #     tsv = self.get_valid_tsv()
    #     return [tsv.seek(i)[0] for i in range(tsv.num_rows())]

    def get_dialog_file(self):
        return self.dialog_file

    def prepare_image_key_to_index(self):
        tsv = self.get_valid_tsv()
        # print(tsv.seek(0)[0])
        # print(type(tsv.seek(0)[0]))
        # print(tsv.seek(0)[1])
        return {tsv.seek(i)[0] : i for i in range(tsv.num_rows())}

    def get_image_features(self, img_idx):
        # feat_info = json.loads(self.tag_and_feat_tsv.seek(img_idx)[1])
        feat_info = eval(self.tag_and_feat_tsv.seek(img_idx)[1]) #transfer to dict
        features = np.frombuffer(base64.b64decode(feat_info['features']), np.float32
                ).reshape((total_num_boxes, -1))
                
        fea = torch.Tensor(features)
        
        # fea = torch.cat((fea,tmp_for_cat), 1)
        return fea

    def get_empty_image_features(self):
        
        fea = torch.zeros((total_num_boxes, 2048))
        
        return fea

    def get_empty_od_labels(self):

        od_labels = "."*36 # hack for runable , won't calculate the MTM loss and tag_logit_bias
        
        return od_labels


    def get_od_labels(self, img_idx):
        od_labels = None
        if self.add_od_labels:
            label_info = eval(self.tag_and_feat_tsv.seek(img_idx)[1])
            # If filter the repeat token
            # od_labels = " ".join(list(set(label_info['objects_class'])))
            od_labels = " ".join(label_info['objects_class'])
        return od_labels


    def __getitem__(self, idx):
        
        dialogs = self.dialogs[idx]
        context = '[SEP]'.join(dialogs[:-1])
        # if add tag bias 
        # context += '[TAG]'
        #else when decoding, empty tag input
        # context += '[SEP]'
        response = dialogs[-1]

        
        # self.dialog2img[idx][0] # e.g., top1 image: [xxx.jpg] 
        # self.dialog2img[idx][k] # e.g., topk image: [xxx.jpg,xxx.jpg...,xxx.jpg] 
        # img_key = op.splitext(self.dialog2img[idx][0])[0] # get the img idx from img path: get the xxx from xxx.jpg

        #tmp for sliding windows' dialog2img
        # img_key = op.splitext(self.dialog2img[idx//2][0])[0] # get the img idx from img path: get the xxx from xxx.jpg
        
        img_key = op.splitext(self.dialog2img[idx][0])[0] # get the img idx from img path: get the xxx from xxx.jpg
        img_idx = self.key2index[img_key]

        
        # features = self.get_empty_image_features()
        # od_labels = self.get_od_labels(img_idx)

        # features = self.get_image_features(img_idx)
        # od_labels = self.get_empty_od_labels()

        features = self.get_empty_image_features()
        od_labels = self.get_empty_od_labels()

        # example = self.tensorizer.tensorize_example(context, response, features, text_b=od_labels)
        text_b = context + od_labels
        
        example = self.tensorizer.tensorize_example(response, features, text_b=text_b)
        # return torch.tensor(int(img_key)), example
        return example

    def __len__(self):
        return len(self.dialogs)



class CaptionTensorizer(object):
    def __init__(self, tokenizer, max_img_seq_length=50, max_seq_length=70, 
            max_seq_a_length=40, max_seq_c_length=40, mask_prob=0.15, mask_response_prob=0.7, max_masked_tokens=3,
            is_train=True):
        """Constructor.
        Args:
            tokenizer: tokenizer for text processing.
            max_img_seq_length: max image sequence length.
            max_seq_length: max text sequence length.
            max_seq_a_length: max caption sequence length.
            is_train: train or test mode.
            mask_prob: probability to mask a input token.
            max_masked_tokens: maximum number of tokens to be masked in one sentence.
        """
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_img_seq_len = max_img_seq_length
        self.max_seq_len = max_seq_length
        self.max_seq_a_len = max_seq_a_length
        self.max_seq_c_len = max_seq_c_length
        self.mask_prob = mask_prob
        self.mask_tag_prob = 0.0
        self.mask_response_prob = mask_response_prob
        self.max_masked_tokens = max_masked_tokens
        self._triangle_mask = torch.tril(torch.ones((self.max_seq_len, 
            self.max_seq_len), dtype=torch.long))

    def tensorize_example(self, text_a, img_feat, text_b=None,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1, sequence_tag_segment_id=2): # when using the oscar should only have 2segment id
            # sequence_a_segment_id=0, sequence_b_segment_id=1, sequence_tag_segment_id=1): # when using the oscar should only have 2segment id
        if self.is_train:
            tokens_a = self.tokenizer.tokenize(text_a)
            if len(tokens_a) > self.max_seq_a_len - 2:
                tokens_a = tokens_a[:(self.max_seq_a_len - 2)]
        else:
            # fake tokens to generate masks
            tokens_a = [self.tokenizer.mask_token] * (self.max_seq_a_len - 2)

            gth_tokens_a = self.tokenizer.tokenize(text_a)
            if len(tokens_a) > self.max_seq_a_len - 2:
                tokens_a = tokens_a[:(self.max_seq_a_len - 2)]

            if len(gth_tokens_a) > self.max_seq_a_len:
                gth_tokens_a = gth_tokens_a[:(self.max_seq_a_len)]

            padding_gth_len = self.max_seq_a_len - len(gth_tokens_a)
            gth_tokens_a += [self.tokenizer.pad_token] * padding_gth_len
            gth_ids = self.tokenizer.convert_tokens_to_ids(gth_tokens_a)
            gth_ids = torch.tensor(gth_ids, dtype=torch.long)
        
        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len = len(tokens)
        if text_b:
            # pad text_a to keep it in fixed length for better inference.
            padding_a_len = self.max_seq_a_len - seq_a_len
            tokens += [self.tokenizer.pad_token] * padding_a_len
            segment_ids += ([pad_token_segment_id] * padding_a_len)

            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            
            # tmp_idx = [i for i,x in enumerate(tokens) if x=="[SEP]"]
            tmp_idx = [i for i,x in enumerate(tokens) if x=="[TAG]"]
            if len(tmp_idx) is not 0:#[TAG]
            # if len(tmp_idx) is not 0:#[SEP]
                # tag_start = tmp_idx[-2] #[SEP]
                tag_start = tmp_idx[-1] #[TAG]
                tokens[tag_start] = "[SEP]"
                tag_end = (tag_start + total_num_boxes) if (tag_start + total_num_boxes) < (self.max_seq_len-1) else (self.max_seq_len-1)
            else: # hacked for tmp
                # tag_start = (self.max_seq_len -1 ) -1 
                # print('enter hacked')
                tag_start = (len(tokens) -1 ) -5 
                tag_end = tag_start + 1
            
            
            # segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
            #tmp comment for OSCAR model by jokie
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) - total_num_boxes)
            segment_ids += [sequence_tag_segment_id] * (total_num_boxes + 1)

        seq_len = len(tokens)
        if self.is_train:
            # rand_prob = 11# random.randint(1,12)
            masked_pos = torch.zeros(self.max_seq_len, dtype=torch.int)
            masked_tag_pos = torch.zeros(self.max_seq_len, dtype=torch.int)
            

        #-----------------------------mask response part-----------------------------
        # if rand_prob<=4:
        # if rand_prob>100:
            
            # randomly mask words for prediction, ignore [CLS]
            candidate_masked_idx = list(range(1, seq_a_len)) # only mask text_a
            random.shuffle(candidate_masked_idx)
            num_masked = min(max(round(self.mask_response_prob * seq_a_len), 1), self.max_masked_tokens)
            num_masked = int(num_masked)
            masked_idx = candidate_masked_idx[:num_masked]
            masked_idx = sorted(masked_idx)
            masked_token = [tokens[i] for i in masked_idx]
            for pos in masked_idx:
                if random.random() <= 0.8:
                    # 80% chance to be a ['MASK'] token
                    tokens[pos] = self.tokenizer.mask_token
                elif random.random() <= 0.5:
                    # 10% chance to be a random word ((1-0.8)*0.5)
                    from random import randint
                    i = randint(0, len(self.tokenizer.vocab))
                    self.tokenizer._convert_id_to_token(i)
                    tokens[pos] = self.tokenizer._convert_id_to_token(i)
                else:
                    # 10% chance to remain the same (1-0.8-0.1)
                    pass

            masked_pos[masked_idx] = 1 
            # # pad masked tokens to the same length
            # if num_masked < self.max_masked_tokens:
            #     masked_token = masked_token + ([self.tokenizer.pad_token] *
            #             (self.max_masked_tokens - num_masked))
            # masked_ids = self.tokenizer.convert_tokens_to_ids(masked_token)

        #-----------------------------mask context part-----------------------------
        # else:
            # candidate_masked_idx_text_b = list(range(self.max_seq_a_len, self.max_seq_a_len + len(tokens_b) )) #   mask text_b
            candidate_masked_idx_text_b = list(range(self.max_seq_a_len, tag_start )) #   mask context
            random.shuffle(candidate_masked_idx_text_b)
            # num_masked_tb = min(max(round(self.mask_prob * len(tokens_b)), 1), self.max_masked_tokens)
            num_masked_tb = min(max(round(self.mask_prob * (tag_start - self.max_seq_a_len)), 1), self.max_masked_tokens)
            num_masked_tb = int(num_masked_tb)
            masked_idx_tb = candidate_masked_idx_text_b[:num_masked_tb]
            masked_idx_tb = sorted(masked_idx_tb)
            masked_token_tb = [tokens[i] for i in masked_idx_tb]
            for pos in masked_idx_tb:
                if random.random() <= 0.8:
                    # 80% chance to be a ['MASK'] token
                    tokens[pos] = self.tokenizer.mask_token
                elif random.random() <= 0.5:
                    # 10% chance to be a random word ((1-0.8)*0.5)
                    from random import randint
                    i = randint(0, len(self.tokenizer.vocab))
                    self.tokenizer._convert_id_to_token(i)
                    tokens[pos] = self.tokenizer._convert_id_to_token(i)
                else:
                    # 10% chance to remain the same (1-0.8-0.1)
                    pass

            # way1: if conditioned Masked LM
            # masked_pos[masked_idx_tb] = 1 
            # pad masked tokens to the same length
            # total_num_masked = (num_masked_tb)
            # if total_num_masked < self.max_masked_tokens:
            #     masked_token = masked_token_tb + ([self.tokenizer.pad_token] *
            #             (self.max_masked_tokens - total_num_masked))
            # masked_ids = self.tokenizer.convert_tokens_to_ids(masked_token)

            # way2: if MRM with MLM in the same time
            masked_pos[masked_idx_tb] = 1 
            # pad masked tokens to the same length
            total_num_masked = (num_masked  + num_masked_tb)
            if total_num_masked < self.max_masked_tokens:
                masked_token = masked_token  + masked_token_tb + ([self.tokenizer.pad_token] *
                        (self.max_masked_tokens - total_num_masked))
            masked_ids = self.tokenizer.convert_tokens_to_ids(masked_token)

            #-----------------------------mask object tag  part-----------------------------
        # else:    
            candidate_masked_idx_tag = list(range(tag_start, tag_end )) #   mask text_b
            candidate_masked_idx_tag = [ min(i,len(tokens)-1) for i in candidate_masked_idx_tag]
            random.shuffle(candidate_masked_idx_tag)
            num_masked_tag = min(max(round(self.mask_tag_prob * (tag_end - tag_start)), 1), self.max_masked_tokens)
            num_masked_tag = int(num_masked_tag)
            masked_idx_tag = candidate_masked_idx_tag[:num_masked_tag]
            masked_idx_tag = sorted(masked_idx_tag)


            # print('num_masked_tag', num_masked_tag)
            # print('tokens:', tokens)
            # print('candidate_masked_idx_tag', candidate_masked_idx_tag)
            # print('num_masked_tag', num_masked_tag)
            # print('len tokens', len(tokens))
            # print('masked_idx_tag',masked_idx_tag)
            # print('tag_start', tag_start)
            # print('tag_end', tag_end)
            # print('tag_end - tag_start', (tag_end-tag_start))

            # try:
            masked_token_tag = [tokens[i] for i in masked_idx_tag]
            
            # except Exception:
            #     print('tokens:', tokens)
            #     print('candidate_masked_idx_tag', candidate_masked_idx_tag)
            #     print('num_masked_tag', num_masked_tag)
            #     print('len tokens', len(tokens))
            #     print('masked_idx_tag',masked_idx_tag)
            #     print('tag_start', tag_start)
            #     print('tag_end', tag_end)
            #     print('tag_end - tag_start', (tag_end-tag_start))
            for pos in masked_idx_tag:
                if random.random() <= 0.8:
                    # 80% chance to be a ['MASK'] token
                    tokens[pos] = self.tokenizer.mask_token
                elif random.random() <= 0.5:
                    # 10% chance to be a random word ((1-0.8)*0.5)
                    from random import randint
                    i = randint(0, len(self.tokenizer.vocab))
                    self.tokenizer._convert_id_to_token(i)
                    tokens[pos] = self.tokenizer._convert_id_to_token(i)
                else:
                    # 10% chance to remain the same (1-0.8-0.1)
                    pass

            # way1: if conditioned Masked LM
            masked_tag_pos[masked_idx_tag] = 1 
            # pad masked tokens to the same length
            if num_masked_tag < self.max_masked_tokens:
                masked_tag_token = masked_token_tag + ([self.tokenizer.pad_token] *
                        (self.max_masked_tokens - num_masked_tag))
            masked_tag_ids = self.tokenizer.convert_tokens_to_ids(masked_tag_token)

            # way2: if MRM with MLM in the same time
            # masked_pos[masked_idx_tb] = 1 
            # # pad masked tokens to the same length
            # total_num_masked = (num_masked  + num_masked_tb)
            # if total_num_masked < self.max_masked_tokens:
            #     masked_token = masked_token  + masked_token_tb + ([self.tokenizer.pad_token] *
            #             (self.max_masked_tokens - total_num_masked))
            # masked_ids = self.tokenizer.convert_tokens_to_ids(masked_token)

                
        
        else:
            masked_pos = torch.ones(self.max_seq_len, dtype=torch.int)

        # pad on the right for image captioning
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        segment_ids += ([pad_token_segment_id] * padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0 : self.max_img_seq_len, ]
            img_len = img_feat.shape[0]
        else:
            padding_matrix = torch.zeros((self.max_img_seq_len - img_len,
                                          img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # prepare attention mask:
        # note that there is no attention from caption to image
        # because otherwise it will violate the triangle attention 
        # for caption as caption will have full attention on image. 
        max_len = self.max_seq_len + self.max_img_seq_len
        attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
        # C: caption, L: label, R: image region
        c_start, c_end = 0, seq_a_len
        l_start, l_end = self.max_seq_a_len, seq_len
        r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
        # triangle mask for caption to caption C-C
        attention_mask[c_start : c_end, c_start : c_end].copy_(self._triangle_mask[0 : seq_a_len, 0 : seq_a_len])
        # full attention for L-L, R-R
        attention_mask[l_start : l_end, l_start : l_end] = 1
        attention_mask[r_start : r_end, r_start : r_end] = 1
        # full attention for C-L, C-R
        attention_mask[c_start : c_end, l_start : l_end] = 1
        attention_mask[c_start : c_end, r_start : r_end] = 1
        # full attention for L-R, R-L:
        attention_mask[l_start : l_end, r_start : r_end] = 1
        attention_mask[r_start : r_end, l_start : l_end] = 1
        # no attention for R-C, L-C

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        # used for get tag position
        tag_pos = torch.zeros(self.max_seq_len, dtype=torch.int)
        tag_idx = list(range( tag_start,  tag_end))
        tag_pos[tag_idx] = 1 

        if self.is_train:
            masked_ids = torch.tensor(masked_ids, dtype=torch.long)
            masked_tag_ids = torch.tensor(masked_tag_ids, dtype=torch.long)
            
            return (input_ids, attention_mask, segment_ids, img_feat, masked_pos, masked_ids, tag_pos, masked_tag_pos, masked_tag_ids)
        return (input_ids, attention_mask, segment_ids, img_feat, masked_pos, gth_ids)

def build_dataset(yaml_file, tokenizer, args, is_train=True):
    if not op.isfile(yaml_file):
        yaml_file = op.join(args.data_dir, yaml_file)
        assert op.isfile(yaml_file)

    if is_train:
        return CaptionTSVDataset(yaml_file, tokenizer=tokenizer,
            add_od_labels=args.add_od_labels, max_img_seq_length=args.max_img_seq_length,
            max_seq_length=args.max_seq_length, max_seq_a_length=args.max_seq_a_length,max_seq_c_length=args.max_seq_c_length,
            is_train=True, mask_prob=args.mask_prob, mask_response_prob=args.mask_response_prob, max_masked_tokens=args.max_masked_tokens)
    if args.use_cbs:
        dataset_class = CaptionTSVDatasetWithConstraints
    else:
        dataset_class = CaptionTSVDataset
    return dataset_class(yaml_file, tokenizer=tokenizer,
            add_od_labels=args.add_od_labels, max_img_seq_length=args.max_img_seq_length,
            max_seq_length=args.max_seq_length, max_seq_a_length=args.max_gen_length,
            is_train=False)


def save_checkpoint(model, tokenizer, args, epoch, global_step):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, global_step))
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model 
    save_num = 0
    while (save_num < 10):
        try:
            model_to_save.save_pretrained(checkpoint_dir)
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            save_num += 1
    if save_num == 10:
        logger.info("Failed to save checkpoint after 10 trails.")
    return checkpoint_dir


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, -1)[1].data # argmax
    scores = logits == labels 
    return scores


def train(args, train_dataset, val_dataset, test_dataset, model, tokenizer):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
            batch_size=args.train_batch_size, num_workers=args.num_workers)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // \
                args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps \
                * args.num_train_epochs

    # Prepare optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not \
            any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if \
            any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.scheduler == "constant":
        scheduler = WarmupConstantSchedule(
                optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = WarmupLinearSchedule(
                optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    if args.scst:
        scst_criterion = ScstRewardCriterion()
        logger.info("  SCST training...")

    global_step, global_loss, global_acc =0,  0.0, 0.0
    global_tag_loss, global_tag_acc = 0.0, 0.0
    masked_global_loss, masked_global_acc =  0.0, 0.0
    # lm_global_loss, lm_global_acc =  0.0, 0.0
    model.zero_grad()
    eval_log = []
    eval_testset_log = []
    best_Bleu_1 = 0
    best_Bleu_2 = 0
    best_Bleu_3 = 0
    best_Bleu_4 = 0
    best_ROUGE_L = 0
    best_METEOR = 0
    best_SPICE = 0


    for epoch in range(int(args.num_train_epochs)):
        # for step, (img_keys, batch) in enumerate(train_dataloader):
        for step, ( batch) in enumerate(train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)

            if not args.scst:
                model.train()
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1],
                    'token_type_ids': batch[2], 'img_feats': batch[3], 
                    'masked_pos': batch[4], 'masked_ids': batch[5], 'tag_pos': batch[6],'masked_tag_pos': batch[7], 'masked_tag_ids': batch[8]
                    # 'masked_pos': batch[4], 'masked_ids': batch[5] , 'response_pos': batch[6], 'response_ids': batch[7]
                }

                outputs = model(**inputs)
                # masked_loss, lm_loss, masked_logits, shift_logits, shift_labels = outputs[:5]
                masked_loss, masked_logits, MTP_loss, tag_logits, tag_ids = outputs[:5]
                # loss = masked_loss + lm_loss
                # loss = masked_loss + MTP_loss
                loss = masked_loss 
                masked_ids = inputs['masked_ids']
                masked_ids = masked_ids[masked_ids != 0]
                batch_score = compute_score_with_logits(masked_logits, masked_ids)
                batch_acc = torch.sum(batch_score.float()) / torch.sum(inputs['masked_pos'])
                batch_tag_score = compute_score_with_logits(tag_logits, tag_ids)
                batch_tag_acc = torch.sum(batch_tag_score.float()) / torch.sum(inputs['masked_tag_pos'])
                # lm_batch_score = compute_score_with_logits(shift_logits, shift_labels)
                # lm_batch_acc = torch.sum(lm_batch_score.float()) / torch.sum(shift_labels)
            # else:
            #     loss = scst_train_iter(args, train_dataset, model, scst_criterion, img_keys, batch, tokenizer)
            #     batch_acc = scst_criterion.get_score()

            if args.n_gpu > 1: 
                loss = loss.mean() # mean() to average on multi-gpu parallel training
                masked_loss = masked_loss.mean() # mean() to average on multi-gpu parallel training
                MTP_loss = MTP_loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            global_loss += loss.item()
            global_acc += batch_acc
            global_tag_loss += MTP_loss.item()
            global_tag_acc += batch_tag_acc
            masked_global_loss += masked_loss.item()
            masked_global_acc += batch_acc
            # lm_global_loss += lm_loss.item()
            # lm_global_acc += lm_batch_acc
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                if global_step % args.logging_steps == 0:
                    logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f}), MTM loss: {:.4f} ({:.4f}), MLM(MRM+MCM) loss: {:.4f} ({:.4f}), ppl: {:.4f} ({:.4f}) " \
                        "MLM(MRM+MCM) score: {:.4f} ({:.4f}), MTM score: {:.4f} ({:.4f})".format(epoch, global_step, 
                        optimizer.param_groups[0]["lr"], loss, global_loss / global_step, MTP_loss, global_tag_loss / global_step, masked_loss, masked_global_loss / global_step, torch.exp(masked_loss), torch.exp(torch.tensor(masked_global_loss / global_step)), 
                        batch_acc, global_acc / global_step, batch_tag_acc, global_tag_acc / global_step)
                    )


                    # logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f}), ppl: {:.4f} ({:.4f}) " \
                    #     "score: {:.4f} ({:.4f})".format(epoch, global_step, 
                    #     optimizer.param_groups[0]["lr"], loss, global_loss / global_step, torch.exp(loss), torch.exp(torch.tensor(global_loss / global_step)), 
                    #     batch_acc, global_acc / global_step)
                    # )


                    # logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f}), masked_loss: {:.4f} ({:.4f}), lm_loss: {:.4f} ({:.4f}), " \
                    #     "masked_pred_acc: {:.4f} ({:.4f}), lm_pred_acc: {:.4f} ({:.4f})".format(epoch, global_step, 
                    #     optimizer.param_groups[0]["lr"], loss, global_loss / global_step , masked_loss, masked_global_loss / global_step , lm_loss, lm_global_loss / global_step, 
                    #     batch_acc, masked_global_acc / global_step, lm_batch_acc, lm_global_acc / global_step)
                    # )

                if (args.save_steps > 0 and global_step % args.save_steps == 0) or \
                        global_step == t_total:
                    checkpoint_dir = save_checkpoint(model, tokenizer, args, epoch, global_step) 
                    # evaluation
                    if args.evaluate_during_training: 
                        logger.info("Perform evaluation at step: %d" % (global_step))
                        evaluate_file = evaluate(args, val_dataset, model, tokenizer,
                                checkpoint_dir)
                        with open(evaluate_file, 'r') as f:
                            res = json.load(f)
                        best_Bleu_1 = max(best_Bleu_1, res['Bleu_1'])
                        best_Bleu_2 = max(best_Bleu_2, res['Bleu_2'])
                        best_Bleu_3 = max(best_Bleu_3, res['Bleu_3'])
                        best_Bleu_4 = max(best_Bleu_4, res['Bleu_4'])
                        best_ROUGE_L = max(best_ROUGE_L, res['ROUGE_L'])
                        best_METEOR = max(best_METEOR, res['METEOR'])
                        best_SPICE = max(best_SPICE, res['SPICE'])
                        res['epoch'] = epoch
                        res['global_step'] = step
                        res['best_Bleu_1'] = best_Bleu_1
                        res['best_Bleu_2'] = best_Bleu_2
                        res['best_Bleu_3'] = best_Bleu_3
                        res['best_Bleu_4'] = best_Bleu_4
                        res['best_ROUGE_L'] = best_ROUGE_L
                        res['best_METEOR'] = best_METEOR
                        res['best_SPICE'] = best_SPICE
                        eval_log.append(res)
                        with open(args.output_dir + '/eval_logs.json', 'w') as f:
                            json.dump(eval_log, f)
                    if args.evaluate_testset_during_training: 
                        logger.info("Perform evaluation testset at step: %d" % (global_step))
                        evaluate_file = evaluate(args, test_dataset, model, tokenizer,
                                checkpoint_dir)
                        with open(evaluate_file, 'r') as f:
                            res = json.load(f)
                        best_Bleu_1 = max(best_Bleu_1, res['Bleu_1'])
                        best_Bleu_2 = max(best_Bleu_2, res['Bleu_2'])
                        best_Bleu_3 = max(best_Bleu_3, res['Bleu_3'])
                        best_Bleu_4 = max(best_Bleu_4, res['Bleu_4'])
                        best_ROUGE_L = max(best_ROUGE_L, res['ROUGE_L'])
                        best_METEOR = max(best_METEOR, res['METEOR'])
                        best_SPICE = max(best_SPICE, res['SPICE'])
                        res['epoch'] = epoch
                        res['global_step'] = step
                        res['best_Bleu_1'] = best_Bleu_1
                        res['best_Bleu_2'] = best_Bleu_2
                        res['best_Bleu_3'] = best_Bleu_3
                        res['best_Bleu_4'] = best_Bleu_4
                        res['best_ROUGE_L'] = best_ROUGE_L
                        res['best_METEOR'] = best_METEOR
                        res['best_SPICE'] = best_SPICE                        
                        
                        eval_testset_log.append(res)
                        with open(args.output_dir + '/eval_testset_logs.json', 'w') as f:
                            json.dump(eval_testset_log, f)
    return global_step, global_loss / global_step


def scst_train_iter(args, train_dataset, model, scst_criterion, img_keys, batch, tokenizer):
    cls_token_id, sep_token_id, pad_token_id, mask_token_id = tokenizer.convert_tokens_to_ids(
            [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token,
                tokenizer.mask_token]
            )
    inputs = {'is_decode': True,
        'input_ids': batch[0], 'attention_mask': batch[1],
        'token_type_ids': batch[2], 'img_feats': batch[3],
        'masked_pos': batch[4],
        'do_sample': False,
        'bos_token_id': cls_token_id,
        'pad_token_id': pad_token_id,
        'eos_token_ids': [sep_token_id, pad_token_id],
        'mask_token_id': mask_token_id,
        # for adding od labels
        'add_od_labels': args.add_od_labels, 'od_labels_start_posid': args.max_seq_a_length,

        # hyperparameters of beam search
        'max_length': args.max_seq_a_length,
        'num_beams': 1,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "num_return_sequences": 1,
        "num_keep_best": 1,
    }

    model.eval()
    with torch.no_grad():
        greedy_res_raw, _ = model(**inputs)
        greedy_res_raw.squeeze_(1)  # batch_size * max_len

    model.train()
    inputs['do_sample'] = True
    sample_res_raw, sample_logprobs = model(**inputs)
    sample_res_raw.squeeze_(1)
    sample_logprobs.squeeze_(1)
    assert sample_logprobs.requires_grad == True
    assert sample_res_raw.requires_grad == False

    def _ids_to_captions(all_ids):
        captions = []
        for ids in all_ids:
            c = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
            captions.append(c)
        return captions

    greedy_res = _ids_to_captions(greedy_res_raw)
    sample_res = _ids_to_captions(sample_res_raw)
    gt_res = [train_dataset.get_captions_by_key(k) for k in img_keys]

    loss = scst_criterion(gt_res, greedy_res, sample_res, sample_logprobs)
    return loss


def get_predict_file(output_dir, yaml_file, args):
    cc = ['pred']
    # make sure it works with/without / in end of the path.
    data = op.basename(op.join(args.data_dir, '')[:-1])
    split = op.basename(yaml_file)
    assert split.endswith('.yaml')
    split = split[:-5]
    cc.append(data)
    cc.append(split)
    cc.append('beam{}'.format(args.num_beams))
    cc.append('sample{}'.format(args.do_sample))
    cc.append('top_p{}'.format(args.top_p))
    cc.append('top_k{}'.format(args.top_k))
    cc.append('temperature{}'.format(args.temperature))
    cc.append('repetition_penalty{}'.format(args.repetition_penalty))
    cc.append('max{}'.format(args.max_gen_length))
    if args.add_od_labels:
        cc.append('odlabels')
    if args.num_keep_best != 1:
        cc.append('best{}'.format(args.num_keep_best))
    if args.use_cbs:
        cc.append('cbs{}'.format(args.min_constraints_to_satisfy))
    if args.output_hidden_states:
        cc.append('hidden')
    return op.join(output_dir, '{}.tsv'.format('.'.join(cc)))


def get_evaluate_file(predict_file):
    assert predict_file.endswith('.tsv')
    fpath = op.splitext(predict_file)[0]
    return fpath + '.eval.json'


def get_evaluate_method(predict_file):
    if 'nocaps' in op.basename(predict_file):
        return 'nocaps'
    else:
        return 'coco'


def evaluate(args, val_dataset, model, tokenizer, output_dir):
    assert op.isdir(output_dir)
    predict_file = get_predict_file(output_dir, val_dataset.yaml_file, args)
    if op.isfile(predict_file):
        logger.info('Skip predict. {} already exists'.format(predict_file))
    else:
        test(args, val_dataset, model, tokenizer, predict_file)

    evaluate_file = get_evaluate_file(predict_file)
    if op.isfile(evaluate_file):
        logger.info('Skip evaluation. {} already exists'.format(evaluate_file))
        return evaluate_file

    eval_method = get_evaluate_method(predict_file)
    if eval_method == 'coco':
        # gt_file = val_dataset.get_caption_file_in_coco_format()
        # result = evaluate_on_coco_caption(predict_file, gt_file, outfile=evaluate_file)

        gt_file = val_dataset.get_dialog_file()
        result = evaluate_on_reddit_caption(predict_file, gt_file, outfile=evaluate_file)
        #for evaluating the DialoGPT result
        # result = evaluate_on_reddit_caption(predict_file+'.txt', gt_file, outfile=evaluate_file)
    else:
        split = 'val' if 'val' in op.basename(val_dataset.yaml_file) else 'test'
        result = evaluate_on_nocaps(split, predict_file, 
                    data_dir=args.data_dir, evaluate_file=evaluate_file)
    logger.info("evaluation result: {}".format(str(result)))
    return evaluate_file


def test(args, test_dataset, model, tokenizer, predict_file):
    args.test_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    test_sampler = SequentialSampler(test_dataset)
    cache_file = predict_file

    test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
            batch_size=args.test_batch_size, num_workers=args.num_workers)

    # cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id, tag_sep_token_id, exclamation_token_id = \
    #     tokenizer.convert_tokens_to_ids( [tokenizer.cls_token, 
    #         tokenizer.sep_token, tokenizer.pad_token, tokenizer.mask_token, '.', '[TAG]', '!']
    #     )
    cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id, question_token_id, exclamation_token_id = \
        tokenizer.convert_tokens_to_ids( [tokenizer.cls_token, 
            tokenizer.sep_token, tokenizer.pad_token, tokenizer.mask_token, '.', '?', '!']
        )
    model.eval()

    def gen_rows():
        time_meter = 0
        # restore existing results for long running inference tasks
        # exist_key2pred = {}
        # tmp_file = cache_file + '.tmp.copy'
        # if op.isfile(tmp_file):
        #     with open(tmp_file, 'r') as fp:
        #         for line in fp:
        #             parts = line.strip().split('\t')
        #             if len(parts) == 2:
        #                 exist_key2pred[parts[0]] = parts[1]
        ppl_total = 0.0
        with torch.no_grad():
            # for step, (img_keys, batch) in tqdm(enumerate(test_dataloader)):
            for step, batch in tqdm(enumerate(test_dataloader), ncols=100,desc="Testing Forward: ", total=len(test_dataloader)):
                # is_exist = True
                # for k in img_keys:
                #     if k not in exist_key2pred:
                #         is_exist = False
                #         break
                # if is_exist:
                #     for k in img_keys:
                #         yield k, exist_key2pred[k]
                #     continue
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'is_decode': True,
                    'input_ids': batch[0], 'attention_mask': batch[1],
                    'token_type_ids': batch[2], 'img_feats': batch[3],
                    'masked_pos': batch[4],
                    'gth_ids': batch[5],
                    'do_sample': args.do_sample,
                    'bos_token_id': cls_token_id,
                    'pad_token_id': pad_token_id,
                    # 'eos_token_ids': [sep_token_id, pad_token_id, period_token_id, exclamation_token_id, question_token_id],
                    'eos_token_ids': [sep_token_id, pad_token_id],
                    'mask_token_id': mask_token_id,
                    # for adding od labels
                    'add_od_labels': args.add_od_labels, 'od_labels_start_posid': args.max_seq_a_length,

                    # hyperparameters of beam search
                    'max_length': args.max_gen_length,
                    'num_beams': args.num_beams,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                    "length_penalty": args.length_penalty,
                    "num_return_sequences": args.num_return_sequences,
                    "num_keep_best": args.num_keep_best,
                }
                if args.use_cbs:
                    inputs.update({'use_cbs': True,
                        'fsm': batch[5],
                        'num_constraints': batch[6],
                        'min_constraints_to_satisfy': args.min_constraints_to_satisfy,
                    })
                tic = time.time()
                # captions, logprobs
                outputs = model(**inputs)
                time_meter += time.time() - tic
                all_caps = outputs[0]  # [batch_size, num_keep_best, max_len]
                all_confs = torch.exp(outputs[1])
                # ppl_per_batch = outputs[2]
                # ppl_total += ppl_per_batch

                # # logger.info("ppl_per_batch: {} , ppl_total: {} ".format(ppl_per_batch.item(), ppl_total.item()/(step+1)))
                # print('\nppl_per_batch: ', ppl_per_batch)                
                # print('\nppl_total: ', ppl_total/(step+1))
                # for img_key, caps, confs in zip(img_keys, all_caps, all_confs):
                #     res = []
                #     for cap, conf in zip(caps, confs):
                #         cap = tokenizer.decode(cap.tolist(), skip_special_tokens=True)
                #         res.append({'caption': cap, 'conf': conf.item()})
                #     if isinstance(img_key, torch.Tensor):
                #         img_key = img_key.item()
                #     yield img_key, json.dumps(res)


                # for gthids in batch[5]:
                #     gth_cap = tokenizer.decode(gthids.tolist(), skip_special_tokens=True)
                #     print('gth_cap: ', gth_cap)
                
                for  caps, confs in zip( all_caps, all_confs):
                    res = []
                    for cap, conf in zip(caps, confs):
                        cap = tokenizer.decode(cap.tolist(), skip_special_tokens=True)
                        
                        # print('resp: ',cap)
                        # cap = cap.split('.')[0]
                        # print(cap)
                        res.append({'response': cap, 'conf': conf.item()})
                        
                    # yield json.dumps(res)
                    yield res
                    # return res

        logger.info("Inference model computing time: {} seconds per batch".format(time_meter / (step+1)))

    tsv_writer(gen_rows(), cache_file)

    return predict_file


def restore_training_settings(args):
    assert not args.do_train
    assert args.do_test or args.do_eval
    # restore training settings, check hasattr for backward compatibility
    train_args = torch.load(op.join(args.eval_model_dir, 'training_args.bin'))
    if hasattr(train_args, 'max_seq_a_length'):
        max_od_labels_len = train_args.max_seq_length - train_args.max_seq_a_length
        max_seq_length = args.max_gen_length + max_od_labels_len
        args.max_seq_length = max_seq_length
        logger.warning('Override max_seq_length to {} = max_gen_length:{} + od_labels_len:{}'.format(
                max_seq_length, args.max_gen_length, max_od_labels_len))

    override_params = ['max_seq_a_length', 'do_lower_case', 'add_od_labels',
            'max_img_seq_length', 'img_feature_dim',
            'img_feature_type']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning('Override {} with train args: {} -> {}'.format(param,
                    test_v, train_v))
                setattr(args, param, train_v)
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='/data/t-zujieliang/data_for_oscar/reddit_oi', type=str, required=False,
                        help="The input data dir with all required files.")
    parser.add_argument("--tag_vocab_file", default='/data/t-zujieliang/data_for_oscar/reddit_oi/objects_vocab.txt', type=str, required=False,
                        help="The files for saving the tag vocabulary.")
    parser.add_argument("--train_yaml", default='train.yaml', type=str, required=False, 
                        help="yaml file for training.")
    parser.add_argument("--test_yaml", default='test.yaml', type=str, required=False,
                        help="yaml file for testing.")
    parser.add_argument("--val_yaml", default='test.yaml', type=str, required=False, 
                        help="yaml file used for validation during training.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
                        help="Path to pre-trained model or model type.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--loss_type", default='sfmx', type=str, 
                        help="Loss function types: support kl, x2, sfmx")
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str, 
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--max_seq_length", default=120, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, "
                             "sequences shorter will be padded.")
    parser.add_argument("--max_seq_a_length", default=40, type=int, 
                        help="The maximum sequence length for caption.")
    parser.add_argument("--max_seq_c_length", default=80, type=int, #dump, useless args now
                        help="The maximum sequence length for dialog context.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_sample", action='store_true', help="Whether to do sampling.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run evaluation.")
    parser.add_argument("--do_lower_case", action='store_true', 
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--mask_prob", default=0.15, type=float,
                        help= "Probability to mask input sentence during training.")
    parser.add_argument("--mask_response_prob", default=0.7, type=float,
                        help= "Probability to mask input sentence during training.")
    parser.add_argument("--max_masked_tokens", type=int, default=300,
                        help="The max number of masked tokens per sentence.")
    parser.add_argument("--add_od_labels", default=False, action='store_true', 
                        help="Whether to add object detection labels or not")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
    parser.add_argument("--max_img_seq_length", default=50, type=int, 
                        help="The maximum total input image sequence length.")
    # parser.add_argument("--img_feature_dim", default=2054, type=int, 
    parser.add_argument("--img_feature_dim", default=2048, type=int, 
                        help="The Image Feature Dimension.")
    parser.add_argument("--type_vocab_size", default=3, type=int, 
                        help="segment embedding vocabulary size.")
    parser.add_argument("--img_feature_type", default='frcnn', type=str,
                        help="Image feature type.")
    parser.add_argument("--per_gpu_train_batch_size", default=64, type=int, 
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=6, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--output_mode", default='classification', type=str,
                        help="output mode, support classification or regression.")
    parser.add_argument("--num_labels", default=2, type=int, 
                        help="num_labels is 2 for classification and 1 for regression.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear or")
    parser.add_argument("--num_workers", default=4, type=int, help="Workers in dataloader.")
    parser.add_argument("--num_train_epochs", default=40, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, 
                        help="Total number of training steps. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=10, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=-1, 
                        help="Save checkpoint every X steps. Will also perform evaluatin.")
    parser.add_argument("--evaluate_during_training", action='store_true', 
                        help="Run evaluation during training at each save_steps.")
    parser.add_argument("--evaluate_testset_during_training", action='store_true', 
                        help="Run evaluation testset during training at each save_steps.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA.")
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization.")
    parser.add_argument('--scst', action='store_true', help='Self-critical sequence training')
    # for generation
    parser.add_argument("--eval_model_dir", type=str, default='', 
                        help="Model directory for evaluation.")
    parser.add_argument('--max_gen_length', type=int, default=40,
                        help="max length of generated sentences")
    parser.add_argument('--output_hidden_states', action='store_true',
                        help="Turn on for fast decoding")
    parser.add_argument('--num_return_sequences', type=int, default=1,
                        help="repeating times per image")
    parser.add_argument('--num_beams', type=int, default=5, help="beam search width")
    parser.add_argument('--num_keep_best', type=int, default=1,
                        help="number of hypotheses to keep in beam search")
    parser.add_argument('--temperature', type=float, default=1,
                        help="temperature in softmax for sampling")
    parser.add_argument('--top_k', type=int, default=0,
                        help="filter distribution for sampling")
    parser.add_argument('--top_p', type=float, default=1,
                        help="filter distribution for sampling")
    parser.add_argument('--repetition_penalty', type=int, default=3,
                        help="repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)")
    parser.add_argument('--length_penalty', type=float, default=0.3,#1,
                        help="beam search length penalty")
    # for Constrained Beam Search
    parser.add_argument('--use_cbs', action='store_true',
                        help='Use constrained beam search for decoding')
    parser.add_argument('--min_constraints_to_satisfy', type=int, default=2,
                        help="minimum number of constraints to satisfy")
    args = parser.parse_args()

    global logger

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    
    output_dir = args.output_dir
    mkdir(output_dir)

    logger = setup_logger("vlpretrain", output_dir, 0)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.n_gpu)
    set_seed(args.seed, args.n_gpu)

    # Load pretrained model and tokenizer
    config_class, model_class, tokenizer_class = BertConfig, BertForImageCaptioning, BertTokenizer
    if args.do_train:
        assert args.model_name_or_path is not None
        config = config_class.from_pretrained(args.config_name if args.config_name else \
                args.model_name_or_path, num_labels=args.num_labels, finetuning_task='image_captioning')
        if args.scst:
            # avoid using too much memory
            config.output_hidden_states = True
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name \
                else args.model_name_or_path, do_lower_case=args.do_lower_case)
        
        config.img_feature_dim = args.img_feature_dim
        config.img_feature_type = args.img_feature_type
        config.hidden_dropout_prob = args.drop_out
        config.loss_type = args.loss_type
        config.type_vocab_size = args.type_vocab_size
        model = model_class.from_pretrained(args.model_name_or_path, 
                from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
        print(args.train_yaml)
        #---------------add by Jokie for add object tags to the vocab , should be commented when calculating the ppl------------------------
        if args.train_yaml not in "test.yaml":
            TAGS_TOKENS_LIST = [line.decode().rstrip('\n').split(',')[0] for line in open(args.tag_vocab_file,'rb')]
            print('len(TAGS_TOKENS_LIST):', len(TAGS_TOKENS_LIST))
            num_add_tokens = tokenizer.add_tokens(TAGS_TOKENS_LIST)
            SPECIAL_TOKENS = ['[TAG]']
            SPECIAL_TOKENS_DICT = {'additional_special_tokens':'[TAG]'}
            num_add_sep_tokens = tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
            print('num_add_tokens:', num_add_tokens)
            print('num_add_sep_tokens:', num_add_sep_tokens)
            print('num_tokens:', len(tokenizer))
            model.resize_token_embeddings(len(tokenizer))
            model.bert._resize_token_embeddings(len(tokenizer))
            bert_embedding_weight = model.bert.embeddings.word_embeddings.weight
            print('after resized decoder_shape: ', bert_embedding_weight.size(1))
            print('after resized decoder_shape1: ', bert_embedding_weight.size(0))
            model.decoder = nn.Linear(bert_embedding_weight.size(1),
                                bert_embedding_weight.size(0), bias=False)
            model.vocab_size = bert_embedding_weight.size(0)
        
    else:
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        config = config_class.from_pretrained(checkpoint)
        config.output_hidden_states = args.output_hidden_states
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        model = model_class.from_pretrained(checkpoint, config=config)
        print('num_tokens for decoding when testing:', len(tokenizer))
        print(tokenizer.tokenize('[TAG]'))

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    if args.do_train:
        train_dataset = build_dataset(op.join(args.data_dir, args.train_yaml), tokenizer, args)
        val_dataset = build_dataset(op.join(args.data_dir, args.val_yaml), 
                tokenizer, args, is_train=False)
        test_dataset = build_dataset(op.join(args.data_dir, args.test_yaml), 
                tokenizer, args, is_train=False)
        global_step, avg_loss = train(args, train_dataset, val_dataset, test_dataset, model, tokenizer)
        logger.info("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)

    # inference and evaluation
    if args.do_test or args.do_eval:
        args = restore_training_settings(args)
        test_dataset = build_dataset(op.join(args.data_dir, args.test_yaml), 
                tokenizer, args, is_train=False)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if not args.do_eval:
            predict_file = get_predict_file(checkpoint, test_dataset.yaml_file, args)
            test(args, test_dataset, model, tokenizer, predict_file)
            logger.info("Prediction results saved to: {}".format(predict_file))
        else:
            evaluate_file = evaluate(args, test_dataset, model, tokenizer,
                    checkpoint)
            logger.info("Evaluation results saved to: {}".format(evaluate_file))

if __name__ == "__main__":
    main()

#TODO: 1. add PPL metrics 2. enlarge the vocabulary size with the object distribution set 3. segment different segment id for the response and the object tag