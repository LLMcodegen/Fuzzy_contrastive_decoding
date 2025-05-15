import torch
from transformers import LogitsProcessor
from typing import List, Union, Tuple, Set, Optional
import torch.nn.functional as F
import gzip
import json
import os
import jsonlines
from bigcode_eval.fuzzy_system.fuzzy_class import Fuzzyclass

class EnsembleLogitsProcessor(LogitsProcessor):

    def __init__(self, uncertain:str, filter_num:float, mean_num:float, num_beams: int, source_weights: List[float] = None, preserve_bos_token: bool = False):
        self.filter_num = filter_num
        self.mean_num = mean_num
        self.num_beams = num_beams
        self.source_weights = source_weights
        self.preserve_bos_token = preserve_bos_token
        self.fuzzy_prejudge = Fuzzyclass(epochs = 400, max_acc = 0.00, last_epoch = 406)
        # self.uncertain = "fuzzy"
        self.total_fcd = 0
        self.total = 0
        self.uncertain = uncertain
        self.sim_js = []
        self.token_id = 0
        # self.uncertain = "fuzzy11"
        # self.number = 1
        # self.number = number
    
    def _diff_maximun(self, tensor):
        max_value, max_index = torch.topk(tensor, 1)
        tensor_without_max = tensor[tensor != max_value]
        second_max_value, _ = torch.topk(tensor_without_max, 1)
        difference = max_value - second_max_value
        return difference
    
    def _kurtosis(self, tensor):
        mean = torch.mean(tensor)
        std = torch.std(tensor)
        fourth_moment = torch.mean((tensor - mean) ** 4)
        kurtosis = fourth_moment / (std ** 4) - 3
        return kurtosis
    
    def _contrast_decoding(self, input_ids, scores):
        
        std_number = torch.std(scores[0])
        mean_value = torch.mean(scores[0])
        filter_low = mean_value * torch.Tensor([self.filter_num]).to(scores.device)
        mask = scores[0] >= filter_low
        scores[0, ~mask] = torch.Tensor([0.00]).to(scores.device)
        scores[1, ~mask] = torch.Tensor([0.00]).to(scores.device)
        batch_size = int(input_ids.size(0) / self.num_beams)
        if self.source_weights is not None:
            assert len(self.source_weights) == batch_size
            source_weights = torch.Tensor(self.source_weights).to(scores.device)
        else:
            source_weights = 1/(batch_size-1) * torch.ones((batch_size,), device=scores.device)
        for i in range(self.num_beams):
            beam_indices = self.num_beams * torch.arange(batch_size, device=scores.device, dtype=torch.long) + i
            cands = scores[beam_indices]
            mean_scores = torch.log((source_weights.unsqueeze(-1).expand(-1, scores.size(-1)) * cands).sum(dim=0))
            for j in beam_indices:
                scores[j] = mean_scores
        return scores
    def result_total(self):
        print("fuzzy contrast decodings:",self.total_fcd)
        print("contrast decodings:",self.total)
    
    def js_divergence(self, P, Q):
        """
        计算两个概率分布的 Jensen-Shannon Divergence
        参数:
        P (tensor): 第一个概率分布 (1D Tensor)
        Q (tensor): 第二个概率分布 (1D Tensor)
        返回:
        float: 两个概率分布的 Jensen-Shannon Divergence
        """
        P = P / P.sum()
        Q = Q / Q.sum()
        M = 0.5 * (P + Q)
        def kl_divergence(P, Q):
            return (P * torch.log(P / Q)).sum()
        
        jsd = 0.5 * kl_divergence(P, M) + 0.5 * kl_divergence(Q, M)
        return jsd.item()
    
    def _divergence(self):
        write_jsonl(f"./result_simi/simi_re16.jsonl", self.sim_js)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if self.preserve_bos_token and cur_len <= 1:
            return scores
        scores = F.softmax(scores, dim=-1)
        self.token_id = self.token_id + 1
        
        if self.uncertain == "fuzzy":
            std_number = torch.std(scores[0])*10000
            Difference_number = self._diff_maximun(scores[0])[0] * 10000
            kurtosis = self._kurtosis(scores[0])
            std_number = std_number.cpu()
            Difference_number = Difference_number.cpu()
            kurtosis = kurtosis.cpu()
            nested_list = [[std_number, Difference_number, kurtosis]]
            result = self.fuzzy_prejudge.fuzzy_prejudge(nested_list)
            self.total= self.total + 1
            if result == 1:
                scores = torch.log(scores)
            else:
                self.total_fcd = self.total_fcd + 1
                scores = self._contrast_decoding(input_ids, scores)
        else:
            std_number = torch.std(scores[0])
            mean_value = torch.mean(scores[0])
            
            if std_number.item() > self.mean_num: # 0.0035>0.004
                scores = torch.log(scores)
            else:
                scores = self._contrast_decoding(input_ids, scores)
        if torch.isnan(scores).any():
            scores = torch.nan_to_num(scores, nan=float('-inf'))
        return scores

def stream_jsonl(filename):
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r", encoding='utf-8') as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)

def write_jsonl(filename, data, append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'a'
    else:
        mode = 'w'
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with jsonlines.open(filename, mode) as fp:
            for x in data:
                fp.write(x)

def data_handle(task_name):
    multi = ['humaneval-cpp', 'humaneval-cs', 'humaneval-d', 'humaneval-go', 'humaneval-java', 'humaneval-jl', 'humaneval-js', 'humaneval-lua', 'humaneval-php', 'humaneval-pl', 'humaneval-py', 'humaneval-r', 'humaneval-rb', 'humaneval-rkt', 'humaneval-rs', 'humaneval-scala', 'humaneval-sh', 'humaneval-swift', 'humaneval-ts']
    if task_name == "humaneval":
        input_file_neg = "./humaneval_data_contrast/human_eval_neg.jsonl"
        data_neg = list(sample_neg["prompt"].strip() for idx_neg, sample_neg in enumerate(stream_jsonl(input_file_neg)))
    elif task_name == "humaneval-unstripped":
        input_file_neg = "./humaneval_data_contrast/human_eval_neg.jsonl"
        data_neg = list(sample_neg["prompt"] for idx_neg, sample_neg in enumerate(stream_jsonl(input_file_neg)))
    elif task_name in multi:
        input_file_neg = os.path.join(f"./MultiPL-E/data_MultiPL-E_neg/{task_name}", f"{task_name}-neg.jsonl")
        data_neg = list(sample_neg["prompt"].strip() for idx_neg, sample_neg in enumerate(stream_jsonl(input_file_neg)))
    return data_neg

def data_example(task_neg):
    if task_neg == 1:
        input_file_neg = "./human_data/data_examples/humaneval_reduce1.jsonl"
        data_neg = list(sample_neg["prompt"].strip() for idx_neg, sample_neg in enumerate(stream_jsonl(input_file_neg)))
    elif task_neg == 2:
        input_file_neg = "./human_data/data_examples/humaneval_reduce2.jsonl"
        data_neg = list(sample_neg["prompt"].strip() for idx_neg, sample_neg in enumerate(stream_jsonl(input_file_neg)))
    elif task_neg == 3:
        input_file_neg = "./human_data/data_examples/humaneval_reduce3.jsonl"
        data_neg = list(sample_neg["prompt"].strip() for idx_neg, sample_neg in enumerate(stream_jsonl(input_file_neg)))
    elif task_neg == 4:
        input_file_neg = "./human_data/data_examples/humaneval_reduce4.jsonl"
        data_neg = list(sample_neg["prompt"].strip() for idx_neg, sample_neg in enumerate(stream_jsonl(input_file_neg)))
    elif task_neg > 5:
        data_neg = None
    return data_neg
          
    