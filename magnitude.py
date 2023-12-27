import torch
import pickle
import copy
import argparse

def get_module():

    res = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "decoder.output_projection.weight"]

    p1 = ["encoder.layers.", "decoder.layers."]
    p2 = ["0.", "1.", "2.", "3.", "4.", "5."]
    self_ = ["self_attn.k_proj.weight", "self_attn.k_proj.bias",
            "self_attn.v_proj.weight", "self_attn.v_proj.bias",
            "self_attn.q_proj.weight", "self_attn.q_proj.bias",
            "self_attn.out_proj.weight", "self_attn.out_proj.bias",
            "self_attn_layer_norm.weight", "self_attn_layer_norm.bias",]

    cross_ = ["encoder_attn.k_proj.weight", "encoder_attn.k_proj.bias",
            "encoder_attn.v_proj.weight", "encoder_attn.v_proj.bias",
            "encoder_attn.q_proj.weight", "encoder_attn.q_proj.bias",
            "encoder_attn.out_proj.weight", "encoder_attn.out_proj.bias",
            "encoder_attn_layer_norm.weight", "encoder_attn_layer_norm.bias",]
    
    fc_ = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", 
                "final_layer_norm.weight", "final_layer_norm.bias"]
    
    prefix = "module.module."
    for a in p1:
        if a == p1[0]:
            for b in p2[:]:
                for c in self_[:-2]:
                    res.append(a + b + c)
                for c in fc_[:-2]:
                    res.append(a + b + c)
        else:
            for b in p2[:]:
                for c in self_[:-2]:
                    res.append(a + b + c)
                for c in cross_[:-2]:
                    res.append(a + b + c)
                for c in fc_[:-2]:
                    res.append(a + b + c)
    return res


def main(args):
    f = open(args.pre_ckt_path, 'rb')
    a = torch.load(f)
    state_new = copy.deepcopy(a)

    mask_matrix = {}
    all_module = get_module()
    ratio = args.prune_ratio
    for key in all_module:
    	tmp = state_new['model'][key]
    	tmp = tmp.view(-1)
    	#important parameter -> 1, unimportant parameter -> 0.
    	tmp_mask = torch.ones_like(tmp)
    	topk = int(tmp.size(0) * ratio)
    	index = torch.topk(torch.abs(tmp), topk, dim=-1, largest=False)[1]
    	tmp.scatter_(-1, index, 0.)
    	tmp_mask.scatter_(-1, index, 0.)
    	tmp = tmp.view(state_new['model'][key].size())
    	tmp_mask = tmp_mask.view(state_new['model'][key].size())
    	state_new['model'][key] = tmp
    	mask_matrix[key] = tmp_mask

    if not args.only_mask:
        ckt = open(args.save_ckt_path, 'wb')
        torch.save(state_new, ckt)

    g = open(args.save_mask_path, 'wb')
    torch.save(mask_matrix, g)



if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument('--pre-ckt-path', type=str, help='The path to the pretraing checkpoint.')
    arg.add_argument('--save-ckt-path', type=str, help='The path to save the pruned checkpoint.')
    arg.add_argument('--save-mask-path', type=str, help='The path to save the parameter mask matrix.')
    arg.add_argument('--prune-ratio', type=float, help='The ratio of parameters to be pruned.')
    arg.add_argument('--only-mask', action='store_true', help='If only save the mask matrix.')
    args = arg.parse_args()
    main(args)




