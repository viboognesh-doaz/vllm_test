import argparse
import json
import torch
import transformers

def from_2_way_softmax(causal_lm, seq_cls_model, tokenizer, tokens, device):
    assert len(tokens) == 2
    lm_head_weights = causal_lm.lm_head.weight
    false_id = tokenizer.convert_tokens_to_ids(tokens[0])
    true_id = tokenizer.convert_tokens_to_ids(tokens[1])
    score_weight = lm_head_weights[true_id].to(device).to(torch.float32) - lm_head_weights[false_id].to(device).to(torch.float32)
    with torch.no_grad():
        seq_cls_model.score.weight.copy_(score_weight.unsqueeze(0))
        if seq_cls_model.score.bias is not None:
            seq_cls_model.score.bias.zero_()

def no_post_processing(causal_lm, seq_cls_model, tokenizer, tokens, device):
    lm_head_weights = causal_lm.lm_head.weight
    token_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
    score_weight = lm_head_weights[token_ids].to(device)
    with torch.no_grad():
        seq_cls_model.score.weight.copy_(score_weight)
        if seq_cls_model.score.bias is not None:
            seq_cls_model.score.bias.zero_()
method_map = {function.__name__: function for function in [from_2_way_softmax, no_post_processing]}

def converting(model_name, classifier_from_tokens, path, method, use_pad_token=False, device='cpu'):
    assert method in method_map
    if method == 'from_2_way_softmax':
        assert len(classifier_from_tokens) == 2
        num_labels = 1
    else:
        num_labels = len(classifier_from_tokens)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    causal_lm = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
    seq_cls_model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True, device_map=device)
    method_map[method](causal_lm, seq_cls_model, tokenizer, classifier_from_tokens, device)
    seq_cls_model.config.use_pad_token = use_pad_token
    seq_cls_model.config.pad_token_id = tokenizer.pad_token_id
    seq_cls_model.save_pretrained(path)
    tokenizer.save_pretrained(path)

def parse_args():
    parser = argparse.ArgumentParser(description='Converting *ForCausalLM models to *ForSequenceClassification models.')
    parser.add_argument('--model_name', type=str, default='BAAI/bge-reranker-v2-gemma', help='Model name')
    parser.add_argument('--classifier_from_tokens', type=str, default='["Yes"]', help='classifier from tokens')
    parser.add_argument('--method', type=str, default='no_post_processing', help='Converting converting')
    parser.add_argument('--use-pad-token', action='store_true', help='Whether to use pad_token')
    parser.add_argument('--path', type=str, default='./bge-reranker-v2-gemma-seq-cls', help='Path to save converted model')
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_args()
    converting(model_name=args.model_name, classifier_from_tokens=json.loads(args.classifier_from_tokens), method=args.method, use_pad_token=args.use_pad_token, path=args.path)