import argparse
import json
import os
import random

import torch
from torch import distributed as dist
from torch.distributed import init_process_group

from config import Config
from models.zsl import transformer_map
from scripts import train_cub, train_visual_bert, train_zest_bert

# ['bert', 'sbert_mean', sbert_cls, 'xlnet', 'distilbert']
embed_models = list(transformer_map.keys())
txt_feats = embed_models + ["vanilla", "similarity", "similarity_VRS"]

parser = argparse.ArgumentParser()

######################################################################################
#                                   General Args                                     #
######################################################################################
parser.add_argument("--gpu", default=Config.gpu, type=str, help="index of GPU to use")
parser.add_argument("--seed", type=int, help="manual seed")
parser.add_argument(
    "--split",
    default=Config.split,
    type=str,
    help="the way to split train/test data: easy/hard",
)
parser.add_argument(
    "--txt_feat",
    default="similarity_VRS",
    type=str,
    choices=txt_feats,
    help="the type of text feature to use:" + str(txt_feats),
)
parser.add_argument("--regularizer_off_epoch", type=int, default=None)
parser.add_argument("--tfidf", action="store_true", default=False)
parser.add_argument("--tfidf_dropout", type=float, default=0)
parser.add_argument("--similarity", action="store_true", default=False)
parser.add_argument("--vrs", action="store_true", default=False)
parser.add_argument(
    "--img_feat",
    default="vpde",
    type=str,
    choices=["vpde", "resnet"],
    help="Image features for each image.",
)
parser.add_argument("--no_bert", action="store_true", default=False)
parser.add_argument("--zero_bert", action="store_true", default=False)
parser.add_argument("--raw_img_feat", action="store_true", default=False)
parser.add_argument(
    "--sentence_extraction",
    type=str,
    default="vrs",
    choices=["vrs_clauses, random_clauses, vrs, random, all"],
)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--nsentences", type=int, default=15)
# Training Params
parser.add_argument("--max_epoch", type=int, default=500)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--after_bert_lr", type=float, default=5e-4)
parser.add_argument("--pooler_lr", type=float, default=5e-4)
parser.add_argument("--lr_discount", type=float, default=0.95)
parser.add_argument(
    "--lr_update_epochs",
    nargs="+",
    default=[50, 80, 150],
    type=int,
)
parser.add_argument(
    "--img_encoding",
    type=str,
    choices=["transformer", "ff", "pad"],
    default="transformer",
)
parser.add_argument("--const_lr", action="store_true", default=False)
parser.add_argument("--reload", action="store_true", default=False)
parser.add_argument("--reload_embeddings", action="store_true", default=False)
parser.add_argument("--weight_decay", type=float, default=0)  # 0.0001
parser.add_argument("-bs", "--batch_size", type=int, default=9)
parser.add_argument("-tbs", "--test_batch_size", type=int, default=30)
parser.add_argument("-nw", "--num_workers", type=int, default=1)
parser.add_argument("-nc", "--ncandidates", type=int, default=100)
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument("--zest", action="store_true", default=False)
# parser.add_argument("--local_rank", type=int)
# parser.add_argument(
#     "--use_env",
#     default=False,
#     action="store_true",
#     help="Use environment variable to pass'local rank'. Must be set to False by default otherwise --local_rank will not be passed",
# )
# Model Params
parser.add_argument("-bn", "--batch_norm", action="store_true", default=False)
# parser.add_argument(
#     "--linear",
#     type=int,
#     default=0,
#     help="Number of linear layers ontop of text processing",
# )
########################################################################################

########################################################################################
#  BERT Args (if txt_feat in ['bert', 'sbert_mean', sbert_cls, 'xlnet', 'distilbert']) #
########################################################################################
parser.add_argument(
    "--nunfrozen",
    default=0,
    type=int,
    help="The number of unfrozen layers in embedding model during training. Defaults to 0/12",
)
# parser.add_argument(
#     "--granularity",
#     default="sentences",
#     type=str,
#     choices=["sentences", "words", "vrs_words"],
#     help="the granularity of each text document to compute an embedding for:",
# )

parser.add_argument(
    "-p",
    "--pooling",
    default="sum",
    type=str,
    choices=[
        "local_cluster",
        "global_cluster",
        "mean",
        "lstm",
        "lstm_sum",
        "cls",
        "mean_cls",
        "sum",
        "concat",
    ],
    # choices=["lstm", "mean", "pointwise_ffn", "none"],
    help="Pooling strategy for embeddings of each sentence.",
)

parser.add_argument(
    "--nclusters",
    default=1,
    type=int,
)

parser.add_argument("--last_two", action="store_true", default=False)
parser.add_argument("--last_two_mean", action="store_true", default=False)
parser.add_argument(
    "--lstm_hidden_dim",
    default=100,
    type=int,
)
parser.add_argument("--lstm_mean_input", action="store_true", default=False)
parser.add_argument(
    "--lstm_input_dim",
    default=None,
    type=int,
)

parser.add_argument(
    "--img_dropout",
    default=0,
    type=float,
)
parser.add_argument(
    "--lstm_dropout",
    default=0,
    type=float,
)
parser.add_argument(
    "--trans_hidden_dim",
    default=2048,
    type=int,
)
parser.add_argument(
    "--trans_dropout",
    default=0,
    type=float,
)
parser.add_argument(
    "--trans_nhead",
    default=8,
    type=float,
)
parser.add_argument(
    "--txt_dropout",
    default=0,
    type=float,
)
parser.add_argument(
    "--ff_dropout",
    default=0,
    type=float,
)
parser.add_argument(
    "--max_txt_len",
    default=None,
    type=int,
)

parser.add_argument(
    "--img_bn",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--txt_bn",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--ff_bn",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--ff_hidden_dim",
    default=0,
    type=int,
)
parser.add_argument(
    "--bert_dropout",
    default=0.1,
    type=float,
)


# parser.add_argument("--lstm_hidden_dim", type=int, default=100)
# parser.add_argument(
#     "-ep",
#     "--token_pooling",
#     default="mean",
#     type=str,
#     choices=["mean", "cls"],
#     help="Pooling strategy for embeddings of each token in a sentence.",
# )
# # parser.add_argument(
# #     "-msl",
# #     "--max_sentence_length",
# #     default=128,
# #     type=int,
# #     help="Maximum sentence length.",
# # )
parser.add_argument("--pretrain", action="store_true", default=False)
parser.add_argument("--img2txt", action="store_true", default=False)
# # Pretrain Args (if pretrain == True)
parser.add_argument(
    "--pretrain_task",
    default="mlm",
    type=str,
    help="[cls, mlm, both] (classification or masked language model or both)",
)
parser.add_argument("--pretrain_mlm_nunfrozen", type=int, default=4)
parser.add_argument("--pretrain_cls_nunfrozen", type=int, default=4)
parser.add_argument("--pretrain_bs", type=int, default=4)
parser.add_argument("--pretrain_mlm_epochs", type=int, default=15)
parser.add_argument("--pretrain_cls_epochs", type=int, default=15)
parser.add_argument("--pretrain_mlm_lr", type=float, default=1e-6)
parser.add_argument("--pretrain_cls_lr", type=float, default=1e-6)
parser.add_argument("--pretrained_path", type=str, default=None)
# parser.add_argument("--pretrain_parallel", action="store_true", default=False)
# parser.add_argument("--pretrain_devices", nargs="+", type=int, default=[2, 3])
# ########################################################################################

# ########################################################################################
# #                            Proto Args (if proto==True)                               #
# ########################################################################################
# parser.add_argument("--encode", action="store_true", default=False)
# parser.add_argument("-np", "--num_proto", type=int, default=100)
# parser.add_argument("-nh", "--nhead", type=int, default=1)
# parser.add_argument("-nd", "--num_decoder_layers", type=int, default=1)
# parser.add_argument("-hd", "--hidden_dim", type=int, default=2048)
# parser.add_argument("--custom_decoder", action="store_true", default=False)
# parser.add_argument("--decoder_only", action="store_true", default=False)
# ########################################################################################

# # Nikkil Args
# parser.add_argument(
#     "-l", "--loss", type=str.lower, choices=["ace", "mse"], default="ace"
# )


args = parser.parse_args()


# if args.seed is None:
#     args.seed = random.randint(1, 10000)

if args.txt_feat in embed_models:
    args.embed = True
else:
    args.embed = False

# args.complete_transformer = not any(
#     (args.custom_decoder, args.decoder_only, args.encode)
# )

# print("Random Seed: ", args.seed)
# random.seed(opt.seed)
# torch.manual_seed(opt.seed)
# torch.cuda.manual_seed_all(opt.seed)

if args.distributed:
    args.gpu = int(os.environ.get("LOCAL_RANK"))  # + 2
    torch.cuda.set_device(args.gpu)
    init_process_group(backend="nccl")
elif not torch.cuda.is_available():
    args.gpu = "cpu"

device = torch.device(
    "cuda:" + str(args.gpu)
    if torch.cuda.is_available() and args.gpu != "cpu"
    else "cpu"
)

if args.gpu == 0 or not dist.is_initialized():
    print("Running parameters:")
    print(json.dumps(vars(args), indent=4, separators=(",", ":")))
if dist.is_initialized() and dist.get_rank() == 0:
    print("world size: ", dist.get_world_size())

args.device = device

if args.gpu != "cpu":
    args.gpu = int(args.gpu)


if __name__ == "__main__":
    if args.embed:
        if args.zest:
            train_zest_bert.train(args)
        else:
            train_visual_bert.train(args)
    else:
        train_cub.train_zest_style(args.loss, args.split, args.batch_size, args.gpu)
