import lm_eval
import numpy as np
import yaml
'\nLM eval harness on model to compare vs HF baseline computed offline.\nConfigs are found in configs/$MODEL.yaml\n\npytest -s -v test_lm_eval_correctness.py     --config-list-file=configs/models-small.txt     --tp-size=1\n'
RTOL = 0.08

def launch_lm_eval(eval_config, tp_size):
    trust_remote_code = eval_config.get('trust_remote_code', False)
    max_model_len = eval_config.get('max_model_len', 4096)
    model_args = f"pretrained={eval_config['model_name']},tensor_parallel_size={tp_size},enforce_eager=true,add_bos_token=true,trust_remote_code={trust_remote_code},max_model_len={max_model_len}"
    results = lm_eval.simple_evaluate(model='vllm', model_args=model_args, tasks=[task['name'] for task in eval_config['tasks']], num_fewshot=eval_config['num_fewshot'], limit=eval_config['limit'], batch_size='auto')
    return results

def test_lm_eval_correctness_param(config_filename, tp_size):
    eval_config = yaml.safe_load(config_filename.read_text(encoding='utf-8'))
    results = launch_lm_eval(eval_config, tp_size)
    success = True
    for task in eval_config['tasks']:
        for metric in task['metrics']:
            ground_truth = metric['value']
            measured_value = results['results'][task['name']][metric['name']]
            print(f"{task['name']} | {metric['name']}: ground_truth={ground_truth} | measured={measured_value}")
            success = success and np.isclose(ground_truth, measured_value, rtol=RTOL)
    assert success