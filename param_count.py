import torch
from model_gru import *
import utils
from config import parse_args


args = parse_args(type=1)
args.input_dim, args.mem_dim, args.word_dim, args.tag_dim, args.rel_dim = 300, 150, 300, 20, 20
criterion = nn.CrossEntropyLoss()
# initialize model, criterion/loss_function, optimizer
model = TreeGRUSentiment(
    args.cuda, args.input_dim,
    args.tag_dim, args.rel_dim,
    args.mem_dim, args.at_hid_dim, 3, criterion
)

print ('-----gru-at-------')
print ('at')
utils.count_param(model.tree_module.gru_at.at)
print ('gru-cell')
utils.count_param(model.tree_module.gru_at.gru_cell)
print ('---')

print ('----gru-cell for leaf ----')
# word + tag
utils.count_param(model.tree_module.gru_cell)
print ('---')