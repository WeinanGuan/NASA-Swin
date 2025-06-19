import torch

from .nasa_swin import NASA_Swin


def load_pretrained(pretrained, model):
    print(f"==============> Loading weight {pretrained} for fine-tuning......")
    checkpoint = torch.load(pretrained, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            print(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C2:
            print(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            print("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            print(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    print(f"=> loaded successfully '{pretrained}'")

    del checkpoint
    torch.cuda.empty_cache()


def build_model(config, pretrained=None, is_simmim=False):
    model_type=config['MODEL']['TYPE']

    # accelerate layernorm
    if config['FUSED_LAYERNORM']:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm
    
    if model_type == 'NASA-Swin':
        model = NASA_Swin(img_size=config['DATA']['IMG_SIZE'],
                            patch_size=config['MODEL']['SWIN']['PATCH_SIZE'],
                            in_chans=config['MODEL']['SWIN']['IN_CHANS'],
                            num_classes=config['MODEL']['NUM_CLASSES'],
                            embed_dim=config['MODEL']['SWIN']['EMBED_DIM'],
                            depths=config['MODEL']['SWIN']['DEPTHS'],
                            num_heads=config['MODEL']['SWIN']['NUM_HEADS'],
                            window_size=config['MODEL']['SWIN']['WINDOW_SIZE'],
                            mlp_ratio=config['MODEL']['SWIN']['MLP_RATIO'],
                            qkv_bias=config['MODEL']['SWIN']['QKV_BIAS'],
                            qk_scale=config['MODEL']['SWIN']['QK_SCALE'],
                            drop_rate=config['MODEL']['DROP_RATE'],
                            drop_path_rate=config['MODEL']['DROP_PATH_RATE'],
                            ape=config['MODEL']['SWIN']['APE'],
                            norm_layer=layernorm,
                            patch_norm=config['MODEL']['SWIN']['PATCH_NORM'],
                            start_nasa_stage=config['MODEL']['SWIN']['START_NASA_STAGE'],
                            in_nasa_stages=config['MODEL']['SWIN']['IN_NASA_STAGES'],
                            end_nasa_stage=config['MODEL']['SWIN']['END_NASA_STAGE'],
                            use_checkpoint=config['TRAIN']['USE_CHECKPOINT'],
                            fused_window_process=config['FUSED_WINDOW_PROCESS'])
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")
    
    if pretrained:
        load_pretrained(pretrained, model)
    
    return model
