import os
from utils import (read_dataset, sanity_check, prep, load_pretrained_config, preprocess_data, create_vocabulary, get_logger,
prepare_data, tokenize, prepare_dataloader)

import numpy as np


def annotate(config, data_dir, adata, test_adata, model_dir, pretrained_model_dir, output_dir):
    sanity_check(config["input_style"], config["output_style"], config["input_emb_style"])
    if config["preprocess"]["input_emb_style"] == "category":
        mask_value = config["preprocess"]["n_bins"] + 1
        pad_value = config["preprocess"]["n_bins"]  # for padding gene expr values
        n_input_bins = config["preprocess"]["n_bins"] + 2
    else:
        mask_value = -1
        pad_value = -2
        n_input_bins = config["preprocess"]["n_bins"]
    logger = get_logger(output_dir)
    adata, adata_test_raw = read_dataset(os.path.join(data_dir, adata), os.path.join(data_dir, test_adata))
    adata = prep(adata)
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    if pretrained_model_dir is not None:
        adata, vocab = load_pretrained_config(adata,
                                              os.path.join(model_dir, 'args.json'),
                                              os.path.join(model_dir, 'best_model.pt'),
                                              os.path.join(model_dir, 'vocab.json'),
                                              output_dir,
                                              special_tokens,
                                              logger
                                              )
    train_data, valid_data, train_celltype_labels, valid_celltype_labels, train_batch_labels, valid_batch_labels = (
        preprocess_data(adata, config["preprocess"]["n_bins"]))
    if pretrained_model_dir is None:
        vocab = create_vocabulary(adata.var["gene_name"].tolist(), special_tokens)
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(adata.var["gene_name"].tolist()), dtype=int)
    tokenized_train, tokenized_valid = tokenize(train_data, valid_data, vocab, gene_ids, logger,
                                                config["preprocess"]["include_zero_gene"])
    prepare_data(tokenized_train,
                 tokenized_valid,
                 train_celltype_labels,
                 valid_celltype_labels,
                 train_batch_labels,
                 valid_batch_labels,
                 mask_value,
                 pad_value,
                 config["preprocess"]["mask_ratio"],
                 config["train"]["epochs"])
    prepare_dataloader()
