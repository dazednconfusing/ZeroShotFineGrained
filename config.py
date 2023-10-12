class Config:
    gpu = '0'
    split = 'easy'

    image_feat = 'vpde'  # ['vpde', 'resnet']

    # ['bert', 'sbert_mean', sbert_cls, 'xlnet','vanilla', 'similarity', 'similarity_VRS']
    text_feat = 'sbert_mean'
    max_sentence_length = 128  # Used for embedding documents at the sentence level

    pre_finetune = False

    # ['cls', 'mlm', 'both'] (classification or masked language model or both)
    pre_finetune_task = 'cls'
    pre_finetune_epochs = 20
    num_adaptable_embedder_layers = 0
    cls_weight = 1  # scale of cls loss term if both cls and mlm are used
    tokens2sentence = 'mean'  # ['mean', 'cls]

    # Filters out num_proto_vecs most relevant sentences using attention
    text_proto_filtering = True
    num_proto_vecs = 15
    num_heads = 1
    num_decoder_layers = 1
    proto_hidden_dim = 500

    # NOTE: if proto vectors are used, simple computes a sum, otherwise it computes a mean
    sentences2doc = 'simple'  # ['simple', 'lstm', 'pointwise_ffn']
    lstm_hidden_dim = 500
    text2final = 'ffn'  # ['ffn', None]
    num_text2final_layers = 1
    final_dim = 2000
    text2final_hidden_dim = 1000

    # image2final = 'proto'  # ['proto', 'ffn', None]
    # proto2image = 'ffn'  # ['ffn', None]
    # num_proto2image
    seed = 500  # Manual seed
    lr = 1e-3
    batch_size = 100
    epochs = 500

    dropout = 0.1
    batch_norm = True
    weight_decay = 0
