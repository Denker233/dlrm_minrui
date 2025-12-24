# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Description: an implementation of a deep learning recommendation model (DLRM)
# The model input consists of dense and sparse features. The former is a vector
# of floating point values. The latter is a list of sparse indices into
# embedding tables, which consist of vectors of floating point values.
# The selected vectors are passed to mlp networks denoted by triangles,
# in some cases the vectors are interacted through operators (Ops).
#
# output:
#                         vector of values
# model:                        |
#                              /\
#                             /__\
#                               |
#       _____________________> Op  <___________________
#     /                         |                      \
#    /\                        /\                      /\
#   /__\                      /__\           ...      /__\
#    |                          |                       |
#    |                         Op                      Op
#    |                    ____/__\_____           ____/__\____
#    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
# input:
# [ dense features ]     [sparse indices] , ..., [sparse indices]
#
# More precise definition of model layers:
# 1) fully connected layers of an mlp
# z = f(y)
# y = Wx + b
#
# 2) embedding lookup (for a list of sparse indices p=[p1,...,pk])
# z = Op(e1,...,ek)
# obtain vectors e1=E[:,p1], ..., ek=E[:,pk]
#
# 3) Operator Op can be one of the following
# Sum(e1,...,ek) = e1 + ... + ek
# Dot(e1,...,ek) = [e1'e1, ..., e1'ek, ..., ek'e1, ..., ek'ek]
# Cat(e1,...,ek) = [e1', ..., ek']'
# where ' denotes transpose operation
#
# References:
# [1] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
# Narayanan Sundaram, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu,
# Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii,
# Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko,
# Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong,
# Misha Smelyanskiy, "Deep Learning Recommendation Model for Personalization and
# Recommendation Systems", CoRR, arXiv:1906.00091, 2019

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import sys

# miscellaneous
import builtins
import datetime
import json
import sys
import time
from fractions import Fraction
import math
# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings
import re
# data generation
import dlrm_data_pytorch as dp

# For distributed run
import extend_distributed as ext_dist
import mlperf_logger

# numpy
import numpy as np
import optim.rwsadagrad as RowWiseSparseAdagrad
import sklearn.metrics

# pytorch
import torch
import torch.nn as nn

# dataloader
try:
    from internals import fbDataLoader, fbInputBatchFormatter

    has_internal_libs = True
except ImportError:
    has_internal_libs = False

from torch._ops import ops
from torch.autograd.profiler import record_function
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

# mixed-dimension trick
from tricks.md_embedding_bag import md_solver, PrEmbeddingBag

# quotient-remainder trick
from tricks.qr_embedding_bag import QREmbeddingBag

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        import onnx
    except ImportError as error:
        print("Unable to import onnx. ", error)

# from torchviz import make_dot
# import torch.nn.functional as Functional
# from torch.nn.parameter import Parameter

import os 
import pickle  
from collections import Counter 
# Import hot/cold preprocessed data loader
from dlrm_data_pytorch import make_hotcold_data_and_loaders

class EmbeddingAccessProfiler:
    """Profile embedding access patterns to identify hot/cold indices"""
    def __init__(self, num_tables):
        self.access_counts = [Counter() for _ in range(num_tables)]
        self.total_accesses = [0] * num_tables
        self.batch_count = 0
        
    def record_batch(self, feature_idx, indices):
        """Record accesses from a batch for a specific feature"""
        indices_np = indices.cpu().numpy()
        self.access_counts[feature_idx].update(indices_np)
        self.total_accesses[feature_idx] += len(indices_np)
    
    def analyze_all(self, ln_emb, hot_percentile, min_table_size):
        hot_cold_profiles = {}
        
        print(f"\n[PROFILE ANALYSIS] Starting analysis...")
        print(f"[PROFILE] Hot percentile: {hot_percentile}%")
        print(f"[PROFILE] Min table size: {min_table_size}")
        
        for table_idx in range(len(ln_emb)):
            # FIXED: Check if counter has data (don't use 'not in' on a list!)
            if table_idx >= len(self.access_counts):
                continue
                
            access_counts = self.access_counts[table_idx]
            
            # Skip if no accesses or table too small
            if not access_counts or ln_emb[table_idx] < min_table_size:
                continue
            
            # Calculate total accesses
            total_accesses = sum(access_counts.values())
            if total_accesses == 0:
                continue
            
            # Sort by access count (descending)
            sorted_embeddings = sorted(access_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Find minimum set of embeddings that cover hot_percentile% of accesses
            target_accesses = total_accesses * (hot_percentile / 100.0)
            cumulative_accesses = 0
            hot_embeddings = []
            
            for emb_id, count in sorted_embeddings:
                hot_embeddings.append(emb_id)
                cumulative_accesses += count
                if cumulative_accesses >= target_accesses:
                    break  # Stop once we've covered target% of accesses
            
            # Create hot/cold indices
            hot_size = len(hot_embeddings)
            cold_size = ln_emb[table_idx] - hot_size
            
            # Create mapping tensors
            hot_map_tensor = torch.full((ln_emb[table_idx],), -1, dtype=torch.long)
            cold_map_tensor = torch.full((ln_emb[table_idx],), -1, dtype=torch.long)
            
            hot_set = set(hot_embeddings)
            hot_idx = 0
            cold_idx = 0
            
            for original_id in range(ln_emb[table_idx]):
                if original_id in hot_set:
                    hot_map_tensor[original_id] = hot_idx
                    hot_idx += 1
                else:
                    cold_map_tensor[original_id] = cold_idx
                    cold_idx += 1
            
            hot_cold_profiles[table_idx] = {
                'hot_size': hot_size,
                'cold_size': cold_size,
                'hot_map_tensor': hot_map_tensor,
                'cold_map_tensor': cold_map_tensor,
                'hot_coverage': (cumulative_accesses / total_accesses) * 100
            }
            
            print(f"[PROFILE] C{table_idx+1}: {hot_size} hot embeddings ({hot_size/ln_emb[table_idx]*100:.2f}% of table) cover {cumulative_accesses/total_accesses*100:.1f}% of accesses")
        
        print(f"[PROFILE] Analysis complete! Found {len(hot_cold_profiles)} tables to split\n")
        return hot_cold_profiles
    
    def save(self, filepath):
        """Save profiling data"""
        data = {
            'access_counts': self.access_counts,
            'total_accesses': self.total_accesses,
            'batch_count': self.batch_count
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"[PROFILER] Saved access profile to {filepath}")
    
    def load(self, filepath):
        """Load profiling data"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.access_counts = data['access_counts']
            self.total_accesses = data['total_accesses']
            self.batch_count = data.get('batch_count', 0)
        print(f"[PROFILER] Loaded access profile from {filepath}")

        
exc = getattr(builtins, "IOError", "FileNotFoundError")

def extract_cpu_percentage(profile_data, function_name):
    # Use re.findall to capture all matches of percentage values for the function
    pattern = re.compile(rf"{re.escape(function_name)}.*?([\d.]+)%.*?([\d.]+)%")
    matches = pattern.findall(profile_data)

    
    # Return the second match if it exists, otherwise return 0.0
    if matches:
        # matches will be a list of tuples, extract the first tuple
        first_percentage, second_percentage = matches[0]
        return float(second_percentage)
    else:
        print(f"No percentages found for {function_name}")
        return None, None
def calculate_ratios(*percentages):
    # Find the smallest percentage as the base for normalization
    min_percentage = min(percentages)
    
    # Normalize the percentages by dividing each by the smallest percentage
    normalized_ratios = [Fraction(p / min_percentage).limit_denominator() for p in percentages]
    
    # Convert normalized ratios to a human-readable format (like 1:2:3:5)
    ratio_str = ":".join(str(r) for r in normalized_ratios)
    
    return ratio_str

def calculate_embeding_percentage(profile_data):
    embedding_bag_cpu = extract_cpu_percentage(profile_data, "aten::embedding_bag")
    embedding_bag_backward_cpu = extract_cpu_percentage(profile_data, "autograd::engine::evaluate_function: EmbeddingBagBac...")
    dlrm_forward_cpu = extract_cpu_percentage(profile_data, "DLRM forward")
    dlrm_backward_cpu = extract_cpu_percentage(profile_data, "DLRM backward")
    numerator = embedding_bag_cpu + embedding_bag_backward_cpu
    denominator = dlrm_forward_cpu + dlrm_backward_cpu

    addmm_cpu = extract_cpu_percentage(profile_data, "aten::addmm")
    bmm_cpu = extract_cpu_percentage(profile_data, "aten::bmm")
    relu_cpu = extract_cpu_percentage(profile_data, "aten::relu")
    bmm_backward_cpu = extract_cpu_percentage(profile_data, "autograd::engine::evaluate_function: BmmBackward0")
    addmm_backward_cpu = extract_cpu_percentage(profile_data, "autograd::engine::evaluate_function: AddmmBackward0")
    relu_cpu_backward = extract_cpu_percentage(profile_data, "autograd::engine::evaluate_function: ReluBackward0")

    if denominator > 0:
        ratio = numerator / denominator
    else:
        ratio = 0.0
    embedding_bag_outof_forward = embedding_bag_cpu/dlrm_forward_cpu
    embedding_back_outof_backward = embedding_bag_backward_cpu/dlrm_backward_cpu
    ratios = calculate_ratios(addmm_cpu, bmm_cpu, relu_cpu,embedding_bag_cpu, bmm_backward_cpu, addmm_backward_cpu,relu_cpu_backward,embedding_bag_backward_cpu)

# Print the result
    print(f"Ratios of CPU percentages (addmm_cpu:bmm_cpu:relu_cpu:embedding_bag_cpu:bmm_backward_cpu:addmm_backward_cpu:relu_cpu_backward:embedding_bag_backward_cpu): {ratios}")
    print(f"Overall embedding ratio (embedding_bag + embedding_bag_backward) / (DLRM forward + DLRM backward): {100*ratio:.2f}%")
    print(f"Embedding bag CPU out of DLRM forward CPU: {100*embedding_bag_outof_forward:.2f}%")
    print(f"Embedding backward CPU out of DLRM backward CPU: {100*embedding_back_outof_backward:.2f}%")
    print(f"embedding_bag_cpu: {embedding_bag_cpu:.2f}%")
    print(f"embedding_bag_backward_cpu: {embedding_bag_backward_cpu:.2f}%")

    print(f"addmm_cpu: {addmm_cpu:.2f}%")
    print(f"bmm_cpu: {bmm_cpu:.2f}%")
    print(f"relu_cpu: {relu_cpu:.2f}%")
    print(f"bmm_backward_cpu: {bmm_backward_cpu:.2f}%")
    print(f"addmm_backward_cpu: {addmm_backward_cpu:.2f}%")
    print(f"relu_cpu_backward: {relu_cpu_backward:.2f}%")
    print(f"dlrm_forward_cpu: {dlrm_forward_cpu:.2f}%")
    print(f"dlrm_backward_cpu: {dlrm_backward_cpu:.2f}%")


def time_wrap(use_gpu):
    if use_gpu:
        torch.cuda.synchronize()
    return time.time()


def dlrm_wrap(X, lS_o, lS_i, use_gpu, device, ndevices=1):
    with record_function("DLRM forward"):
        if use_gpu:  # .cuda()
            # lS_i can be either a list of tensors or a stacked tensor.
            # Handle each case below:
            if ndevices == 1:
                lS_i = (
                    [S_i.to(device) for S_i in lS_i]
                    if isinstance(lS_i, list)
                    else lS_i.to(device)
                )
                lS_o = (
                    [S_o.to(device) for S_o in lS_o]
                    if isinstance(lS_o, list)
                    else lS_o.to(device)
                )
        return dlrm(X.to(device), lS_o, lS_i)


def loss_fn_wrap(Z, T, use_gpu, device):
    with record_function("DLRM loss compute"):
        if args.loss_function == "mse" or args.loss_function == "bce":
            return dlrm.loss_fn(Z, T.to(device))
        elif args.loss_function == "wbce":
            loss_ws_ = dlrm.loss_ws[T.data.view(-1).long()].view_as(T).to(device)
            loss_fn_ = dlrm.loss_fn(Z, T.to(device))
            loss_sc_ = loss_ws_ * loss_fn_
            return loss_sc_.mean()


# The following function is a wrapper to avoid checking this multiple times in th
# loop below.
def unpack_batch(b):
    if args.data_generation == "internal":
        return fbInputBatchFormatter(b, args.data_size)
    else:
        # Experiment with unweighted samples
        return b[0], b[1], b[2], b[3], torch.ones(b[3].size()), None


class LRPolicyScheduler(_LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, decay_start_step, num_decay_steps):
        self.num_warmup_steps = num_warmup_steps
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_start_step + num_decay_steps
        self.num_decay_steps = num_decay_steps

        if self.decay_start_step < self.num_warmup_steps:
            sys.exit("Learning rate warmup must finish before the decay starts")

        super(LRPolicyScheduler, self).__init__(optimizer)

    def get_lr(self):
        step_count = self._step_count
        if step_count < self.num_warmup_steps:
            # warmup
            scale = 1.0 - (self.num_warmup_steps - step_count) / self.num_warmup_steps
            lr = [base_lr * scale for base_lr in self.base_lrs]
            self.last_lr = lr
        elif self.decay_start_step <= step_count and step_count < self.decay_end_step:
            # decay
            decayed_steps = step_count - self.decay_start_step
            scale = ((self.num_decay_steps - decayed_steps) / self.num_decay_steps) ** 2
            min_lr = 0.0000001
            lr = [max(min_lr, base_lr * scale) for base_lr in self.base_lrs]
            self.last_lr = lr
        else:
            if self.num_decay_steps > 0:
                # freeze at last, either because we're after decay
                # or because we're between warmup and decay
                lr = self.last_lr
            else:
                # do not adjust
                lr = self.base_lrs
        return lr


### define dlrm in PyTorch ###
class DLRM_Net(nn.Module):
    def create_mlp(self, ln, sigmoid_layer):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers
        return torch.nn.Sequential(*layers)


    def _create_hotcold_tables(self, m, ln, weighted_pooling, hot_cold_profiles, 
                            tables_to_split, start_idx=0):
        """
        Create hot/cold split embedding tables using function-based approach.
        """
        emb_l = nn.ModuleList()
        v_W_l = []
        emb_map = {}
        
        print(f"\n[HOTCOLD] Creating hot/cold split for {len(tables_to_split)} tables")
        
        for table_idx in tables_to_split:
            n = ln[table_idx]
            feature_name = f"C{table_idx+1}"
            
            if table_idx not in hot_cold_profiles:
                print(f"[HOTCOLD] Warning: No profile for {feature_name}, skipping")
                continue
            
            profile = hot_cold_profiles[table_idx]
            hot_size = profile['hot_size']
            cold_size = profile['cold_size']
            
            # Create separate hot and cold EmbeddingBag tables
            hot_emb = nn.EmbeddingBag(hot_size, m, mode='sum', sparse=True)
            cold_emb = nn.EmbeddingBag(cold_size, m, mode='sum', sparse=True)
            
            # Initialize weights
            W_hot = np.random.uniform(
                low=-np.sqrt(1 / hot_size), 
                high=np.sqrt(1 / hot_size), 
                size=(hot_size, m)
            ).astype(np.float32)
            hot_emb.weight.data = torch.tensor(W_hot, requires_grad=True)
            
            W_cold = np.random.uniform(
                low=-np.sqrt(1 / cold_size), 
                high=np.sqrt(1 / cold_size), 
                size=(cold_size, m)
            ).astype(np.float32)
            cold_emb.weight.data = torch.tensor(W_cold, requires_grad=True)
            
            hot_size_mb = (hot_size * m * 4) / (1024 * 1024)
            cold_size_mb = (cold_size * m * 4) / (1024 * 1024)
            
            print(f"[HOTCOLD] {feature_name}: hot={hot_size} ({hot_size_mb:.1f}MB), "
                f"cold={cold_size} ({cold_size_mb:.1f}MB)")
            
            # Store both tables
            hot_table_idx = start_idx + len(emb_l)
            cold_table_idx = hot_table_idx + 1
            
            emb_map[table_idx] = {
                'type': 'hotcold',
                'hot_table_idx': hot_table_idx,
                'cold_table_idx': cold_table_idx,
                'hot_map_tensor': profile['hot_map_tensor'],
                'cold_map_tensor': profile['cold_map_tensor'],
            }
            
            emb_l.append(hot_emb)
            emb_l.append(cold_emb)
            
            v_W_l.append(None if weighted_pooling is None else torch.ones(hot_size, dtype=torch.float32))
            v_W_l.append(None if weighted_pooling is None else torch.ones(cold_size, dtype=torch.float32))
        
        return emb_l, v_W_l, emb_map

    def _create_merged_tables(self, m, ln, weighted_pooling, merge_threshold, tables_to_merge, start_idx=0):
        """
        Merge small tables together to reduce table count and improve cache locality.
        """
        emb_list = []
        v_W_list = []
        index_map = {}
        
        if not tables_to_merge:
            return emb_list, v_W_list, index_map
        
        print(f"\n[MERGE] Creating merged tables from {len(tables_to_merge)} small tables")
        
        # Group tables into merge groups
        merge_groups = []
        current_group = []
        current_group_size = 0
        
        for idx, size in tables_to_merge:
            current_group.append((idx, size))
            current_group_size += size
            
            # Finalize group if it gets too large
            if current_group_size > merge_threshold * 5:
                merge_groups.append(current_group)
                current_group = []
                current_group_size = 0
        
        # Add final group
        if current_group:
            merge_groups.append(current_group)
        
        print(f"[MERGE] Created {len(merge_groups)} merge groups")
        
        # Create merged embedding tables
        for group_num, group in enumerate(merge_groups):
            total_size = sum(n for _, n in group)
            table_sizes = [n for _, n in group]
            feature_indices = [idx for idx, _ in group]
            print(f"[MERGE] Group {group_num}: merging features {['C'+str(i+1) for i in feature_indices]}")
            print(f"[MERGE]   Sizes: {table_sizes}, total={total_size}")
            
            # Create merged table with appropriate optimization
            if self.qr_flag and total_size > self.qr_threshold:
                print(f"[MERGE]   -> Using QR embedding (size={total_size})")
                EE = QREmbeddingBag(
                    total_size, m, self.qr_collisions,
                    operation=self.qr_operation,
                    mode="sum", sparse=True,
                )
            elif self.md_flag and total_size > self.md_threshold:
                print(f"[MERGE]   -> Using MD embedding (size={total_size})")
                base = max(m) if isinstance(m, list) else m
                _m = base
                EE = PrEmbeddingBag(total_size, _m, base)
                W = np.random.uniform(
                    low=-np.sqrt(1 / total_size), 
                    high=np.sqrt(1 / total_size), 
                    size=(total_size, _m)
                ).astype(np.float32)
                EE.embs.weight.data = torch.tensor(W, requires_grad=True)
            else:
                print(f"[MERGE]   -> Using standard EmbeddingBag")
                EE = nn.EmbeddingBag(total_size, m, mode="sum", sparse=True)
                W = np.random.uniform(
                    low=-np.sqrt(1 / total_size), 
                    high=np.sqrt(1 / total_size), 
                    size=(total_size, m)
                ).astype(np.float32)
                EE.weight.data = torch.tensor(W, requires_grad=True)
            
            # Track offsets for each feature in the merged table
            offset = 0
            current_table_idx = start_idx + len(emb_list)
            for orig_idx, n in group:
                index_map[orig_idx] = (current_table_idx, offset)
                offset += n
            
            emb_list.append(EE)
            v_W_list.append(None if weighted_pooling is None else torch.ones(total_size, dtype=torch.float32))
        
        # CRITICAL: Return the results!
        return emb_list, v_W_list, index_map


    def _create_standalone_tables(self, m, ln, weighted_pooling, standalone_tables, start_idx=0):
        """
        Create standalone embedding tables (no merging or splitting).
        
        Args:
            start_idx: Starting index for embedding tables in the global list
        """
        emb_list = []
        v_W_list = []
        index_map = {}

        #     # ADD THIS DEBUG:
        # print(f"[DEBUG] standalone_tables type: {type(standalone_tables)}")
        # print(f"[DEBUG] standalone_tables: {standalone_tables}")
        # if standalone_tables:
        #     print(f"[DEBUG] first item: {standalone_tables[0]}, type: {type(standalone_tables[0])}")
        
        for orig_idx, n in standalone_tables:
            # Create embedding table
            if self.qr_flag and n > self.qr_threshold:
                EE = QREmbeddingBag(
                    n, m, self.qr_collisions,
                    operation=self.qr_operation,
                    mode="sum", sparse=True,
                )
            elif self.md_flag and n > self.md_threshold:
                base = max(m) if isinstance(m, list) else m
                _m = m[orig_idx] if isinstance(m, list) and n > self.md_threshold else base
                EE = PrEmbeddingBag(n, _m, base)
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, _m)
                ).astype(np.float32)
                EE.embs.weight.data = torch.tensor(W, requires_grad=True)
            else:
                EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
                EE.weight.data = torch.tensor(W, requires_grad=True)
            
            current_table_idx = start_idx + len(emb_list)  # FIX: Use global index
            index_map[orig_idx] = (current_table_idx, 0)
            emb_list.append(EE)
            v_W_list.append(None if weighted_pooling is None else torch.ones(n, dtype=torch.float32))
        
        return emb_list, v_W_list, index_map

    def _calculate_split_sizes(self, total_size, num_splits):
        """
        Calculate sizes for splitting a table into num_splits sub-tables.
        Distributes rows as evenly as possible.
        
        Args:
            total_size: Total number of rows in the table
            num_splits: Number of sub-tables to create
        
        Returns:
            list: List of sub-table sizes (integers)
        """
        base_size = total_size // num_splits
        remainder = total_size % num_splits
        
        # Distribute remainder across first 'remainder' sub-tables
        split_sizes = []
        for i in range(num_splits):
            if i < remainder:
                split_sizes.append(base_size + 1)
            else:
                split_sizes.append(base_size)
        
        return split_sizes



    def _group_tables_by_size(self, ln, merge_threshold, split_threshold, min_table_size, 
                           min_split_rows, max_split_rows, hotcold_threshold=0, hot_cold_profiles=None):
       """
       Group tables by size for reorganization strategy.
       Tables are only split if they fall within [min_split_rows, max_split_rows] range.
       Hot/cold splitting takes priority over regular splitting when enabled.
       """
       groups = {
           'to_hotcold': [],    # Tables to split into hot/cold: [idx, ...] (just indices)
           'to_merge': [],      # Tables to merge: [(idx, size), ...]
           'to_split': [],      # Tables to split: [idx, ...] (just indices)
           'standalone': [],    # Tables to keep as-is: [(idx, size), ...]
           'stats': {
               'total': ln.size,
               'tiny': 0,
               'small': 0,
               'medium': 0,
               'large': 0,
               'hotcold': 0  
           }
       }
       
       for i in range(ln.size):
           size = ln[i]
           feature_name = f"C{i+1}"
           
           # PRIORITY 1: Hot/cold splitting (if enabled and profile exists)
           if (hotcold_threshold > 0 and 
               hot_cold_profiles is not None and 
               i in hot_cold_profiles and 
               size >= hotcold_threshold):
               groups['stats']['large'] += 1
               groups['stats']['hotcold'] += 1
               groups['to_hotcold'].append(i)  # Just index (matches _create_hotcold_tables)
               size_mb = (size * 256) / (1024 * 1024)
               print(f"[GROUP] Feature {feature_name} (size={size}, {size_mb:.1f}MB): marked for HOT/COLD SPLIT")
               continue  # Skip other checks
           
           # Categorize by size (existing logic)
           if size < min_table_size:
               groups['stats']['tiny'] += 1
               groups['standalone'].append((i, size))
               print(f"[GROUP] Feature {feature_name} (size={size}): STANDALONE (too small)")
               
           elif merge_threshold > 0 and min_table_size <= size < merge_threshold:
               groups['stats']['small'] += 1
               groups['to_merge'].append((i, size))
               print(f"[GROUP] Feature {feature_name} (size={size}): marked for MERGE")
               
           elif split_threshold > 0 and size >= split_threshold:
               # Check if table is within splittable range
               if size < min_split_rows:
                   groups['stats']['medium'] += 1
                   groups['standalone'].append((i, size))
                   print(f"[GROUP] Feature {feature_name} (size={size}): STANDALONE (below min split rows)")
               elif size > max_split_rows:
                   groups['stats']['large'] += 1
                   groups['standalone'].append((i, size))
                   size_mb = (size * 256) / (1024 * 1024)
                   print(f"[GROUP] Feature {feature_name} (size={size}, {size_mb:.1f}MB): STANDALONE (above max split rows)")
               else:
                   # Table is within range, will be split - keep as index only
                   groups['stats']['large'] += 1
                   groups['to_split'].append(i)
                   size_mb = (size * 256) / (1024 * 1024)
                   print(f"[GROUP] Feature {feature_name} (size={size}, {size_mb:.1f}MB): marked for SPLIT")
               
           else:
               groups['stats']['medium'] += 1
               groups['standalone'].append((i, size))
               print(f"[GROUP] Feature {feature_name} (size={size}): STANDALONE (medium size)")
       
       return groups


    def _create_split_tables(self, m, ln, weighted_pooling, split_threshold, to_split, num_splits, start_idx=0):
        """
        Create split embedding tables using fixed num_splits for all tables.
        
        Args:
            to_split: List of table indices to split
            num_splits: Number of sub-tables to split each table into
        """
        emb_l = nn.ModuleList()
        v_W_l = []
        emb_map = {}
        
        print(f"\n[SPLIT] Splitting {len(to_split)} tables into {num_splits} sub-tables each")
        
        for table_idx in to_split:  # Just table index now, not tuple
            n = ln[table_idx]
            feature_name = f"C{table_idx+1}"
            
            # Split the table into sub-tables
            split_sizes = self._calculate_split_sizes(n, num_splits)
            
            sub_size = n // num_splits
            sub_size_mb = (sub_size * 256) / (1024 * 1024)
            print(f"[SPLIT] Table {feature_name} (size={n}): splitting into {num_splits} tables, ~{sub_size_mb:.1f}MB per sub-table, sizes={split_sizes}")
            
            # Create mapping for this split table: list of (table_idx, base_offset)
            mapping = []
            base_offset = 0
            
            for sub_idx, sub_size in enumerate(split_sizes):
                # Create sub-table
                if self.qr_flag and sub_size > self.qr_threshold:
                    EE = QREmbeddingBag(
                        sub_size, m, self.qr_collisions,
                        operation=self.qr_operation,
                        mode="sum", sparse=True,
                    )
                elif self.md_flag and sub_size > self.md_threshold:
                    base = max(m) if isinstance(m, list) else m
                    _m = m[table_idx] if isinstance(m, list) and sub_size > self.md_threshold else base
                    EE = PrEmbeddingBag(sub_size, _m, base)
                    W = np.random.uniform(
                        low=-np.sqrt(1 / sub_size), high=np.sqrt(1 / sub_size), size=(sub_size, _m)
                    ).astype(np.float32)
                    EE.embs.weight.data = torch.tensor(W, requires_grad=True)
                else:
                    EE = nn.EmbeddingBag(sub_size, m, mode="sum", sparse=True)
                    W = np.random.uniform(
                        low=-np.sqrt(1 / sub_size), high=np.sqrt(1 / sub_size), size=(sub_size, m)
                    ).astype(np.float32)
                    EE.weight.data = torch.tensor(W, requires_grad=True)
                
                print(f"[SPLIT]   -> Sub-table {sub_idx}: Using standard EmbeddingBag (size={sub_size})")
                
                # Track mapping: (global_table_idx, base_offset_in_original_table)
                current_table_idx = start_idx + len(emb_l)
                mapping.append((current_table_idx, base_offset))
                base_offset += sub_size
                
                emb_l.append(EE)
                
                if weighted_pooling is None:
                    v_W_l.append(None)
                else:
                    v_W_l.append(torch.ones(sub_size, dtype=torch.float32))
            
            # Store mapping for this feature: list of (table_idx, base_offset) tuples
            emb_map[table_idx] = mapping
        
        return emb_l, v_W_l, emb_map


    def create_emb(self, m, ln, weighted_pooling=None, merge_threshold=0, split_threshold=0, 
                    min_table_size=50, num_splits=4, min_split_rows=100000, max_split_rows=5000000,
                    hotcold_threshold=0, hot_cold_profiles=None):
        """
        Create embedding tables with optional merging, splitting, and hot/cold strategies.
        """
        # Initialize emb_index_map for all cases
        self.emb_index_map = {}
        
        # If no reorganization, use original logic
        if merge_threshold == 0 and split_threshold == 0 and hotcold_threshold == 0:
            print("[REORG] No table reorganization - using original embedding creation")
            emb_l = nn.ModuleList()
            v_W_l = []
            
            for i in range(0, ln.size):
                if ext_dist.my_size > 1:
                    if i not in self.local_emb_indices:
                        continue
                n = ln[i]

                # construct embedding operator (original logic)
                if self.qr_flag and n > self.qr_threshold:
                    EE = QREmbeddingBag(
                        n, m, self.qr_collisions,
                        operation=self.qr_operation,
                        mode="sum", sparse=True,
                    )
                elif self.md_flag and n > self.md_threshold:
                    base = max(m) if isinstance(m, list) else m
                    _m = m[i] if isinstance(m, list) and n > self.md_threshold else base
                    EE = PrEmbeddingBag(n, _m, base)
                    W = np.random.uniform(
                        low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, _m)
                    ).astype(np.float32)
                    EE.embs.weight.data = torch.tensor(W, requires_grad=True)
                else:
                    EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
                    W = np.random.uniform(
                        low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                    ).astype(np.float32)
                    EE.weight.data = torch.tensor(W, requires_grad=True)
                
                # Map each feature to its own table
                self.emb_index_map[i] = (len(emb_l), 0)
                
                if weighted_pooling is None:
                    v_W_l.append(None)
                else:
                    v_W_l.append(torch.ones(n, dtype=torch.float32))
                emb_l.append(EE)
            
            print(f"[REORG] Created {len(emb_l)} separate embedding tables")
            return emb_l, v_W_l
        
        # Table reorganization enabled
        print(f"\n{'='*60}")
        print(f"[REORG] Embedding Table Reorganization Strategy")
        print(f"[REORG] Merge threshold: {merge_threshold} (0 = disabled)")
        print(f"[REORG] Split threshold: {split_threshold} (0 = disabled)")
        print(f"[REORG] Hot/Cold threshold: {hotcold_threshold} (0 = disabled)")
        print(f"[REORG] Min table size for merging: {min_table_size}")
        print(f"[REORG] Split range: [{min_split_rows}, {max_split_rows}] rows")
        print(f"[REORG] Number of splits: {num_splits}")
        print(f"{'='*60}")
        
        # Group tables by size (FIXED: pass all required parameters)
        groups = self._group_tables_by_size(
            ln, merge_threshold, split_threshold, 
            min_table_size, min_split_rows, max_split_rows,
            hotcold_threshold, hot_cold_profiles
        )
        
        # Print statistics
        stats = groups['stats']
        print(f"\n[REORG] Table categorization:")
        print(f"[REORG]   Total tables: {stats['total']}")
        print(f"[REORG]   Tiny (< {min_table_size}): {stats['tiny']} tables")
        print(f"[REORG]   Small (merge candidates): {stats['small']} tables")
        print(f"[REORG]   Medium (keep as-is): {stats['medium']} tables")
        print(f"[REORG]   Large (split candidates): {stats['large']} tables")
        print(f"[REORG]   Hot/Cold candidates: {stats['hotcold']} tables")
        
        # Create all tables
        emb_l = nn.ModuleList()
        v_W_l = []
        current_idx = 0
        
        # 1. Create hot/cold tables (FIXED: this was missing!)
        if hotcold_threshold > 0 and groups['to_hotcold'] and hot_cold_profiles:
            hotcold_emb, hotcold_v_W, hotcold_map = self._create_hotcold_tables(
                m, ln, weighted_pooling, hot_cold_profiles, groups['to_hotcold'], start_idx=current_idx
            )
            emb_l.extend(hotcold_emb)
            v_W_l.extend(hotcold_v_W)
            self.emb_index_map.update(hotcold_map)
            current_idx += len(hotcold_emb)
        
        # 2. Create merged tables
        if merge_threshold > 0 and groups['to_merge']:
            merged_emb, merged_v_W, merged_map = self._create_merged_tables(
                m, ln, weighted_pooling, merge_threshold, groups['to_merge'], start_idx=current_idx
            )
            emb_l.extend(merged_emb)
            v_W_l.extend(merged_v_W)
            self.emb_index_map.update(merged_map)
            current_idx += len(merged_emb)

        # 3. Create split tables
        if split_threshold > 0 and groups['to_split']:
            split_emb, split_v_W, split_map = self._create_split_tables(
                m, ln, weighted_pooling, split_threshold, groups['to_split'], 
                num_splits, start_idx=current_idx
            )
            emb_l.extend(split_emb)
            v_W_l.extend(split_v_W)
            self.emb_index_map.update(split_map)
            current_idx += len(split_emb)

        # 4. Create standalone tables
        if groups['standalone']:
            standalone_emb, standalone_v_W, standalone_map = self._create_standalone_tables(
                m, ln, weighted_pooling, groups['standalone'], start_idx=current_idx
            )
            emb_l.extend(standalone_emb)
            v_W_l.extend(standalone_v_W)
            self.emb_index_map.update(standalone_map)
        
        print(f"\n[REORG] Final result:")
        print(f"[REORG]   Original tables: {ln.size}")
        print(f"[REORG]   Final tables: {len(emb_l)}")
        print(f"[REORG]   Change: {len(emb_l) - ln.size:+d} tables")
        print(f"{'='*60}\n")
        
        return emb_l, v_W_l

    def __init__(
        self,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        arch_interaction_op=None,
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
        qr_flag=False,
        qr_operation="mult",
        qr_collisions=0,
        qr_threshold=200,
        md_flag=False,
        md_threshold=200,
        weighted_pooling=None,
        loss_function="bce",
        thread_count = None,
        num_splits=2,
        merge_threshold=0,     
        split_threshold=0,     
        min_table_size=50,
        min_split_rows=100000,      
        max_split_rows=5000000,
        hotcold_threshold=0,
        hot_cold_profiles=None,
        access_profiler=None,
    ):
        super(DLRM_Net, self).__init__()

        if (
            (m_spa is not None)
            and (ln_emb is not None)
            and (ln_bot is not None)
            and (ln_top is not None)
            and (arch_interaction_op is not None)
        ):

            # save arguments
            self.time_interact =0
            self.time_look_up =0
            self.time_mlp=0
            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            self.loss_function = loss_function
            self.thread_count = thread_count
            self.merge_threshold = merge_threshold
            self.split_threshold = split_threshold
            self.min_table_size = min_table_size
            self.num_splits = num_splits
            self.min_split_rows = min_split_rows 
            self.max_split_rows = max_split_rows 
            self.hotcold_threshold = hotcold_threshold     
            self.hot_cold_profiles = hot_cold_profiles     
            self.access_profiler = access_profiler      
            self.profiling_enabled = (access_profiler is not None)    
            self.remap_time = 0.0
            self.lookup_time = 0.0
            self.remap_count = 0
            self.standard_lookup_time = 0.0
            self.standard_lookup_count = 0

            if self.profiling_enabled:
                print("[DLRM] Profiling ENABLED - will record access patterns")
            elif hot_cold_profiles is not None:
                print("[DLRM] Hot/cold split ENABLED - using existing profiles")
            else:
                print("[DLRM] Standard mode - no profiling, no hot/cold")

            if weighted_pooling is not None and weighted_pooling != "fixed":
                self.weighted_pooling = "learned"
            else:
                self.weighted_pooling = weighted_pooling
            # create variables for QR embedding if applicable
            self.qr_flag = qr_flag
            if self.qr_flag:
                self.qr_collisions = qr_collisions
                self.qr_operation = qr_operation
                self.qr_threshold = qr_threshold
            # create variables for MD embedding if applicable
            self.md_flag = md_flag
            if self.md_flag:
                self.md_threshold = md_threshold

            # If running distributed, get local slice of embedding tables
            if ext_dist.my_size > 1:
                n_emb = len(ln_emb)
                if n_emb < ext_dist.my_size:
                    sys.exit(
                        "only (%d) sparse features for (%d) devices, table partitions will fail"
                        % (n_emb, ext_dist.my_size)
                    )
                self.n_global_emb = n_emb
                self.n_local_emb, self.n_emb_per_rank = ext_dist.get_split_lengths(
                    n_emb
                )
                self.local_emb_slice = ext_dist.get_my_slice(n_emb)
                self.local_emb_indices = list(range(n_emb))[self.local_emb_slice]

            # create operators
            if ndevices <= 1:
                self.emb_l, w_list = self.create_emb(
                    m_spa,
                    ln_emb,
                    weighted_pooling,
                    self.merge_threshold,
                    self.split_threshold,
                    self.min_table_size,
                    self.num_splits,
                    self.min_split_rows,
                    self.max_split_rows,
                    self.hotcold_threshold,      
                    self.hot_cold_profiles,     
                    
                )
                if self.weighted_pooling == "learned":
                    self.v_W_l = nn.ParameterList()
                    for w in w_list:
                        self.v_W_l.append(Parameter(w))
                else:
                    self.v_W_l = w_list
            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(ln_top, sigmoid_top)

            # quantization
            self.quantize_emb = False
            self.emb_l_q = []
            self.quantize_bits = 32

            # specify the loss function
            if self.loss_function == "mse":
                self.loss_fn = torch.nn.MSELoss(reduction="mean")
            elif self.loss_function == "bce":
                self.loss_fn = torch.nn.BCELoss(reduction="mean")
            elif self.loss_function == "wbce":
                self.loss_ws = torch.tensor(
                    np.fromstring(args.loss_weights, dtype=float, sep="-")
                )
                self.loss_fn = torch.nn.BCELoss(reduction="none")
            else:
                sys.exit(
                    "ERROR: --loss-function=" + self.loss_function + " is not supported"
                )

    def apply_mlp(self, x, layers):
        # approach 1: use ModuleList
        # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers
        return layers(x)

    def _build_lookup_groups(self):
        """
        Build groups for efficient lookup based on emb_index_map.
        """
        lookup_groups = []
        
        # Build reverse mapping: which features map to each table?
        table_to_features = {}
        split_features = {}
        
        for feat_idx, mapping in self.emb_index_map.items():
            if isinstance(mapping, list):
                # Split feature
                split_features[feat_idx] = mapping
            else:
                # Merged or standalone feature
                table_idx, offset = mapping
                if table_idx not in table_to_features:
                    table_to_features[table_idx] = []
                table_to_features[table_idx].append((feat_idx, offset))
        
        # Process merged/standalone tables
        for table_idx, features in table_to_features.items():
            features.sort(key=lambda x: x[1])  # Sort by offset
            
            if len(features) > 1:
                # Multiple features in same table = merged table
                lookup_groups.append({
                    'type': 'merged',
                    'table_idx': table_idx,
                    'features': features
                })
            else:
                # Single feature = standalone table
                feat_idx, offset = features[0]
                lookup_groups.append({
                    'type': 'standalone',
                    'table_idx': table_idx,
                    'feature': feat_idx,
                    'offset': offset
                })
        
        # Process split features
        for feat_idx, sub_tables in split_features.items():
            for sub_idx, (table_idx, base_offset) in enumerate(sub_tables):
                lookup_groups.append({
                    'type': 'split',
                    'table_idx': table_idx,
                    'feature': feat_idx,
                    'base_offset': base_offset,
                    'sub_idx': sub_idx
                })
        
        return lookup_groups
    def apply_emb(self, lS_o, lS_i, emb_l, v_W_l, 
                  lS_o_cold=None, lS_i_cold=None, hotcold_mask=None, use_preprocessed=False):
        """
        Apply embeddings with optional preprocessed hot/cold support.
        
        When use_preprocessed=True:
            - lS_o, lS_i contain HOT indices/offsets (already remapped!)
            - lS_o_cold, lS_i_cold contain COLD indices/offsets (already remapped!)
            - hotcold_mask indicates which tables use hot/cold split
            - NO runtime remapping needed = huge speedup!
        
        When use_preprocessed=False:
            - Original behavior with runtime remapping for hot/cold tables
        """
        start_time = time.time()

        # Initialize timing counters (once)
        if not hasattr(self, 'remap_time'):
            self.remap_time = 0
            self.lookup_time = 0
            self.remap_count = 0
            self.standard_lookup_time = 0
            self.standard_lookup_count = 0

        ly = []
        if not hasattr(self, '_debug_batch_count'):
               self._debug_batch_count = 0
        
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]
            if k == 2 and self._debug_batch_count < 1:
                print(f"\n[DEBUG C3 ORIGINAL INDICES] Batch {self._debug_batch_count}:")
                orig_indices = sparse_index_group_batch.cpu().numpy() if hasattr(sparse_index_group_batch, 'cpu') else sparse_index_group_batch
                print(f"  Original indices[:30]: {orig_indices[:30].tolist()}")
                if len(orig_indices) > 1:
                    diffs = np.diff(orig_indices[:30])
                    print(f"  Differences between consecutive: {diffs.tolist()}")
                    print(f"  Are they sequential? Max diff: {diffs.max()}, Min diff: {diffs.min()}")
            # ===== PROFILING: Record accesses if profiler is active =====
            if hasattr(self, 'access_profiler') and self.access_profiler is not None:
                if self.profiling_enabled and self.access_profiler is not None:
                    self.access_profiler.record_batch(k, sparse_index_group_batch)

            # ===== PREPROCESSED HOT/COLD MODE =====
            if use_preprocessed and hotcold_mask is not None and hotcold_mask[k]:
                # Indices are ALREADY REMAPPED - no runtime remapping needed!
                lookup_start = time.time()
                
                hot_offset = lS_o[k]
                hot_index = lS_i[k]
                cold_offset = lS_o_cold[k]
                cold_index = lS_i_cold[k]
                
                batch_size = len(hot_offset) - 1
                device = hot_index.device if len(hot_index) > 0 else (
                    cold_index.device if len(cold_index) > 0 else 'cpu')
                
                # Get table mapping
                emb_mapping = self.emb_index_map.get(k)
                
                if emb_mapping is None or not isinstance(emb_mapping, dict) or emb_mapping.get('type') != 'hotcold':
                    # Fallback: not a hot/cold table, use standard lookup
                    table_idx, offset = self.emb_index_map.get(k, (k, 0)) if not isinstance(emb_mapping, dict) else (k, 0)
                    E = emb_l[table_idx] if table_idx < len(emb_l) else emb_l[k]
                    V = E(hot_index, hot_offset)
                    ly.append(V)
                    continue
                
                hot_table_idx = emb_mapping['hot_table_idx']
                cold_table_idx = emb_mapping['cold_table_idx']
                
                # Hot lookup (indices already remapped - NO overhead!)
                if len(hot_index) > 0:
                    V_hot = emb_l[hot_table_idx](hot_index, hot_offset[:-1])
                else:
                    emb_dim = emb_l[hot_table_idx].embedding_dim
                    dtype = emb_l[hot_table_idx].weight.dtype
                    V_hot = torch.zeros(batch_size, emb_dim, device=device, dtype=dtype)
                
                # Cold lookup
                if len(cold_index) > 0:
                    V_cold = emb_l[cold_table_idx](cold_index, cold_offset[:-1])
                else:
                    emb_dim = emb_l[cold_table_idx].embedding_dim
                    dtype = emb_l[cold_table_idx].weight.dtype
                    V_cold = torch.zeros(batch_size, emb_dim, device=device, dtype=dtype)
                
                # Combine hot + cold
                V = V_hot + V_cold
                # if not hasattr(self, '_debug_batch_count'):
                #     self._debug_batch_count = 0

                if k == 2 and self._debug_batch_count < 3:  # Debug table C3 (index 2)
                    print(f"\n[DEBUG C3] Table {k}:")
                    print(f"  batch_size: {batch_size}")
                    print(f"  Hot: {len(hot_index)} indices, offset shape: {hot_offset.shape}")
                    print(f"  Cold: {len(cold_index)} indices, offset shape: {cold_offset.shape}")
                    print(f"  V_hot shape: {V_hot.shape}, mean: {V_hot.mean():.6f}, std: {V_hot.std():.6f}")
                    print(f"  V_cold shape: {V_cold.shape}, mean: {V_cold.mean():.6f}, std: {V_cold.std():.6f}")
                    print(f"  V combined shape: {V.shape}, mean: {V.mean():.6f}, std: {V.std():.6f}")
                    print(f"  Hot indices[:10]: {hot_index[:10].tolist() if len(hot_index) > 0 else 'empty'}")
                    print(f"  Cold indices[:10]: {cold_index[:10].tolist() if len(cold_index) > 0 else 'empty'}")
                    print(f"  Hot table size: {emb_l[hot_table_idx].num_embeddings}")
                    print(f"  Cold table size: {emb_l[cold_table_idx].num_embeddings}")
                
                lookup_end = time.time()
                self.lookup_time += (lookup_end - lookup_start)
                self.remap_count += 1  # Count as processed (but no remap time!)
                
                ly.append(V)
                continue
            
            # ===== PREPROCESSED NON-HOTCOLD TABLE =====
            if use_preprocessed and hotcold_mask is not None and not hotcold_mask[k]:
                # Standard table in preprocessed mode - just do direct lookup
                standard_start = time.time()
                
                emb_mapping = self.emb_index_map.get(k, (k, 0))
                
                if isinstance(emb_mapping, tuple):
                    table_idx, offset = emb_mapping
                    adjusted_indices = sparse_index_group_batch + offset
                else:
                    table_idx = k
                    adjusted_indices = sparse_index_group_batch
                
                E = emb_l[table_idx] if table_idx < len(emb_l) else emb_l[0]
                V = E(adjusted_indices, sparse_offset_group_batch[:-1])
                
                standard_end = time.time()
                self.standard_lookup_time += (standard_end - standard_start)
                self.standard_lookup_count += 1
                
                ly.append(V)
                continue

            # ===== ORIGINAL RUNTIME REMAPPING MODE =====
            # Get the embedding table mapping for this feature
            emb_mapping = self.emb_index_map.get(k, (k, 0))
            
            if isinstance(emb_mapping, dict) and emb_mapping.get('type') == 'hotcold':
                # === HOT/COLD TABLE (runtime remapping) ===
                remap_start = time.time()
                
                hot_table_idx = emb_mapping['hot_table_idx']
                cold_table_idx = emb_mapping['cold_table_idx']
                hot_map = emb_mapping['hot_map_tensor'].to(sparse_index_group_batch.device)
                cold_map = emb_mapping['cold_map_tensor'].to(sparse_index_group_batch.device)
                
                batch_size = len(sparse_offset_group_batch) - 1
                device = sparse_index_group_batch.device
                
                # Vectorized mapping
                hot_indices_new = hot_map[sparse_index_group_batch]
                cold_indices_new = cold_map[sparse_index_group_batch]
                
                # Mask out invalid (-1) entries
                hot_mask = hot_indices_new >= 0
                cold_mask = cold_indices_new >= 0
                
                # Process hot embeddings
                if hot_mask.any():
                    valid_hot_indices = hot_indices_new[hot_mask]
                    hot_offsets = self._recompute_offsets(sparse_offset_group_batch, hot_mask)
                else:
                    valid_hot_indices = None
                    hot_offsets = None
                
                # Process cold embeddings
                if cold_mask.any():
                    valid_cold_indices = cold_indices_new[cold_mask]
                    cold_offsets = self._recompute_offsets(sparse_offset_group_batch, cold_mask)
                else:
                    valid_cold_indices = None
                    cold_offsets = None
                
                remap_end = time.time()
                self.remap_time += (remap_end - remap_start)
                self.remap_count += 1
                
                # TIME THE LOOKUPS
                lookup_start = time.time()
                
                if valid_hot_indices is not None:
                    if v_W_l[hot_table_idx] is not None:
                        hot_weights = v_W_l[hot_table_idx].to(device)
                        per_sample_weights = hot_weights[valid_hot_indices]
                    else:
                        per_sample_weights = None
                    
                    V_hot = emb_l[hot_table_idx](valid_hot_indices, hot_offsets, 
                                                per_sample_weights=per_sample_weights)
                else:
                    V_hot = torch.zeros(batch_size, emb_l[hot_table_idx].embedding_dim, 
                                    device=device, dtype=emb_l[hot_table_idx].weight.dtype)
                
                if valid_cold_indices is not None:
                    if v_W_l[cold_table_idx] is not None:
                        cold_weights = v_W_l[cold_table_idx].to(device)
                        per_sample_weights = cold_weights[valid_cold_indices]
                    else:
                        per_sample_weights = None
                    
                    V_cold = emb_l[cold_table_idx](valid_cold_indices, cold_offsets,
                                                per_sample_weights=per_sample_weights)
                else:
                    V_cold = torch.zeros(batch_size, emb_l[cold_table_idx].embedding_dim,
                                        device=device, dtype=emb_l[cold_table_idx].weight.dtype)
                
                # Combine
                V = V_hot + V_cold
                
                lookup_end = time.time()
                self.lookup_time += (lookup_end - lookup_start)
                
                ly.append(V)
                
            elif isinstance(emb_mapping, list):
                # === SPLIT TABLE === (existing code)
                V_accumulated = None
                actual_batch_size = None
                
                for sub_idx, (table_idx, base_offset) in enumerate(emb_mapping):
                    E = emb_l[table_idx]
                    sub_table_size = E.num_embeddings
                    
                    range_start = base_offset
                    range_end = base_offset + sub_table_size
                    mask = (sparse_index_group_batch >= range_start) & (sparse_index_group_batch < range_end)
                    
                    if mask.any():
                        adjusted_indices = torch.clamp(sparse_index_group_batch - base_offset, 0, sub_table_size - 1)
                        
                        if v_W_l[table_idx] is not None:
                            base_weights = v_W_l[table_idx].gather(0, adjusted_indices)
                            per_sample_weights = base_weights * mask.float()
                        else:
                            per_sample_weights = mask.float()
                        
                        V_part = E(adjusted_indices, sparse_offset_group_batch, per_sample_weights=per_sample_weights)
                        
                        if actual_batch_size is None:
                            actual_batch_size = V_part.shape[0]
                    else:
                        if actual_batch_size is None:
                            actual_batch_size = len(sparse_offset_group_batch) - 1
                        
                        V_part = torch.zeros(actual_batch_size, E.embedding_dim, 
                                            dtype=E.weight.dtype, device=E.weight.device)
                    
                    if V_accumulated is None:
                        V_accumulated = V_part
                    else:
                        V_accumulated = V_accumulated + V_part
                
                ly.append(V_accumulated)
                
            else:
                # === MERGED or STANDALONE TABLE === (existing code)
                standard_start = time.time()
                
                table_idx, offset = emb_mapping
                adjusted_indices = sparse_index_group_batch + offset
                
                if v_W_l[table_idx] is not None:
                    per_sample_weights = v_W_l[table_idx].gather(0, adjusted_indices)
                else:
                    per_sample_weights = None

                if self.quantize_emb:
                    if self.quantize_bits == 4:
                        QV = ops.quantized.embedding_bag_4bit_rowwise_offsets(
                            self.emb_l_q[table_idx],
                            adjusted_indices,
                            sparse_offset_group_batch,
                            per_sample_weights=per_sample_weights,
                        )
                    elif self.quantize_bits == 8:
                        QV = ops.quantized.embedding_bag_byte_rowwise_offsets(
                            self.emb_l_q[table_idx],
                            adjusted_indices,
                            sparse_offset_group_batch,
                            per_sample_weights=per_sample_weights,
                        )
                    ly.append(QV)
                else:
                    E = emb_l[table_idx]
                    V = E(
                        adjusted_indices,
                        sparse_offset_group_batch,
                        per_sample_weights=per_sample_weights,
                    )
                    ly.append(V)
                
                standard_end = time.time()
                self.standard_lookup_time += (standard_end - standard_start)
                self.standard_lookup_count += 1
        
        # print(f"\n[DEBUG apply_emb] Output sizes:")
        # for idx, v in enumerate(ly):
        #     print(f"  ly[{idx}]: {v.shape}")
        if hasattr(self, '_debug_batch_count'):
            self._debug_batch_count += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.time_look_up += elapsed_time
        return ly

    def _recompute_offsets(self, original_offsets, mask):
        """Helper to recompute offsets after filtering"""
        counts = []
        for i in range(len(original_offsets) - 1):
            start = original_offsets[i]
            end = original_offsets[i+1]
            count = mask[start:end].sum().item()
            counts.append(count)
        
        new_offsets = [0]
        for count in counts:
            new_offsets.append(new_offsets[-1] + count)
        
        return torch.tensor(new_offsets, dtype=torch.long, device=original_offsets.device)
    #  using quantizing functions from caffe2/aten/src/ATen/native/quantized/cpu
    def quantize_embedding(self, bits):

        n = len(self.emb_l)
        self.emb_l_q = [None] * n
        for k in range(n):
            if bits == 4:
                self.emb_l_q[k] = ops.quantized.embedding_bag_4bit_prepack(
                    self.emb_l[k].weight
                )
            elif bits == 8:
                self.emb_l_q[k] = ops.quantized.embedding_bag_byte_prepack(
                    self.emb_l[k].weight
                )
            else:
                return
        self.emb_l = None
        self.quantize_emb = True
        self.quantize_bits = bits

    def interact_features(self, x, ly):
        start_time = time.time()
        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            # append dense feature with the interactions (into a row vector)
            # approach 1: all
            # Zflat = Z.view((batch_size, -1))
            # approach 2: unique
            _, ni, nj = Z.shape
            # approach 1: tril_indices
            # offset = 0 if self.arch_interaction_itself else -1
            # li, lj = torch.tril_indices(ni, nj, offset=offset)
            # approach 2: custom
            offset = 1 if self.arch_interaction_itself else 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )
        
        end_time = time.time() 

        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        self.time_interact+=elapsed_time

        return R

    def forward(self, dense_x, lS_o, lS_i, lS_o_cold=None, lS_i_cold=None, hotcold_mask=None):
        # Store preprocessed cold data for use in apply_emb
        if lS_o_cold is not None:
            self._preprocessed_cold_data = (lS_o_cold, lS_i_cold, hotcold_mask)
        else:
            self._preprocessed_cold_data = None
        
        if ext_dist.my_size > 1:
            # multi-node multi-device run
            return self.distributed_forward(dense_x, lS_o, lS_i)
        elif self.ndevices <= 1:
            # single device run
            return self.sequential_forward(dense_x, lS_o, lS_i)
        else:
            # single-node multi-device run
            return self.parallel_forward(dense_x, lS_o, lS_i)

    def distributed_forward(self, dense_x, lS_o, lS_i):
        batch_size = dense_x.size()[0]
        # WARNING: # of ranks must be <= batch size in distributed_forward call
        if batch_size < ext_dist.my_size:
            sys.exit(
                "ERROR: batch_size (%d) must be larger than number of ranks (%d)"
                % (batch_size, ext_dist.my_size)
            )
        if batch_size % ext_dist.my_size != 0:
            sys.exit(
                "ERROR: batch_size %d can not split across %d ranks evenly"
                % (batch_size, ext_dist.my_size)
            )

        dense_x = dense_x[ext_dist.get_my_slice(batch_size)]
        lS_o = lS_o[self.local_emb_slice]
        lS_i = lS_i[self.local_emb_slice]

        if (len(self.emb_l) != len(lS_o)) or (len(self.emb_l) != len(lS_i)):
            sys.exit(
                "ERROR: corrupted model input detected in distributed_forward call"
            )
        # start_time = time.time()
        # embeddings
        with record_function("DLRM embedding forward"):
            ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l)
        # end_time = time.time()

        # # Calculate the elapsed time
        # elapsed_time = end_time - start_time
        # self.time_look_up += elapsed_time

        # WARNING: Note that at this point we have the result of the embedding lookup
        # for the entire batch on each rank. We would like to obtain partial results
        # corresponding to all embedding lookups, but part of the batch on each rank.
        # Therefore, matching the distribution of output of bottom mlp, so that both
        # could be used for subsequent interactions on each device.
        if len(self.emb_l) != len(ly):
            sys.exit("ERROR: corrupted intermediate result in distributed_forward call")

        a2a_req = ext_dist.alltoall(ly, self.n_emb_per_rank)

        # start_time = time.time()
        with record_function("DLRM bottom nlp forward"):
            x = self.apply_mlp(dense_x, self.bot_l)

        # Stop the timer after MLP construction
        # end_time = time.time()

        # # Calculate the elapsed time
        # elapsed_time = end_time - start_time
        # self.time_mlp+=elapsed_time
        ly = a2a_req.wait()
        ly = list(ly)


        # interactions
        
        with record_function("DLRM interaction forward"):
            z = self.interact_features(x, ly)
        

        # top mlp
        start_time = time.time()
        with record_function("DLRM top nlp forward"):
            p = self.apply_mlp(z, self.top_l)
        # Stop the timer after MLP construction
        end_time = time.time() 
        
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        self.time_mlp+=elapsed_time
        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p

        return z

    def sequential_forward(self, dense_x, lS_o, lS_i):
        # process dense features (using bottom mlp), resulting in a row vector
        start_time = time.time()
        x = self.apply_mlp(dense_x, self.bot_l)
        # Stop the timer after MLP construction
        end_time = time.time()
        

        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        self.time_mlp+=elapsed_time
        # debug prints
        # print("intermediate")
        # print(x.detach().cpu().numpy())

        # start_time = time.time()
        # process sparse features(using embeddings), resulting in a list of row vectors
        if hasattr(self, '_preprocessed_cold_data') and self._preprocessed_cold_data is not None:
            lS_o_cold, lS_i_cold, hotcold_mask = self._preprocessed_cold_data
            ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l,
                               lS_o_cold=lS_o_cold, lS_i_cold=lS_i_cold, 
                               hotcold_mask=hotcold_mask, use_preprocessed=True)
        else:
            ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l)
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # self.time_look_up += elapsed_time
        # for y in ly:
        #     print(y.detach().cpu().numpy())

        # interact features (dense and sparse)
        start_time = time.time()
        z = self.interact_features(x, ly)
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.time_interact+=elapsed_time
        # print(z.detach().cpu().numpy())

        start_time = time.time()
        # obtain probability of a click (using top mlp)
        p = self.apply_mlp(z, self.top_l)
        end_time = time.time()
        

        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        self.time_mlp+=elapsed_time

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p

        return z

    def parallel_forward(self, dense_x, lS_o, lS_i):
        ### prepare model (overwrite) ###
        # WARNING: # of devices must be >= batch size in parallel_forward call
        batch_size = dense_x.size()[0]
        ndevices = min(self.ndevices, batch_size, len(self.emb_l))
        device_ids = range(ndevices)
        # WARNING: must redistribute the model if mini-batch size changes(this is common
        # for last mini-batch, when # of elements in the dataset/batch size is not even
        if self.parallel_model_batch_size != batch_size:
            self.parallel_model_is_not_prepared = True

        if self.parallel_model_is_not_prepared or self.sync_dense_params:
            # replicate mlp (data parallelism)
            self.bot_l_replicas = replicate(self.bot_l, device_ids)
            self.top_l_replicas = replicate(self.top_l, device_ids)
            self.parallel_model_batch_size = batch_size

        if self.parallel_model_is_not_prepared:
            # distribute embeddings (model parallelism)
            t_list = []
            w_list = []
            for k, emb in enumerate(self.emb_l):
                d = torch.device("cuda:" + str(k % ndevices))
                t_list.append(emb.to(d))
                if self.weighted_pooling == "learned":
                    w_list.append(Parameter(self.v_W_l[k].to(d)))
                elif self.weighted_pooling == "fixed":
                    w_list.append(self.v_W_l[k].to(d))
                else:
                    w_list.append(None)
            self.emb_l = nn.ModuleList(t_list)
            if self.weighted_pooling == "learned":
                self.v_W_l = nn.ParameterList(w_list)
            else:
                self.v_W_l = w_list
            self.parallel_model_is_not_prepared = False

        ### prepare input (overwrite) ###
        # scatter dense features (data parallelism)
        # print(dense_x.device)
        dense_x = scatter(dense_x, device_ids, dim=0)
        # distribute sparse features (model parallelism)
        if (len(self.emb_l) != len(lS_o)) or (len(self.emb_l) != len(lS_i)):
            sys.exit("ERROR: corrupted model input detected in parallel_forward call")

        t_list = []
        i_list = []
        for k, _ in enumerate(self.emb_l):
            d = torch.device("cuda:" + str(k % ndevices))
            t_list.append(lS_o[k].to(d))
            i_list.append(lS_i[k].to(d))
        lS_o = t_list
        lS_i = i_list

        ### compute results in parallel ###
        # bottom mlp
        # WARNING: Note that the self.bot_l is a list of bottom mlp modules
        # that have been replicated across devices, while dense_x is a tuple of dense
        # inputs that has been scattered across devices on the first (batch) dimension.
        # The output is a list of tensors scattered across devices according to the
        # distribution of dense_x.
        start_time = time.time()
        x = parallel_apply(self.bot_l_replicas, dense_x, None, device_ids)
        # Stop the timer after MLP construction
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        self.time_mlp+=elapsed_time
        # debug prints
        # print(x)

        # embeddings
        # start_time = time.time()
        ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l)
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # self.time_look_up+=elapsed_time
        # debug prints
        # print(ly)

        # butterfly shuffle (implemented inefficiently for now)
        # WARNING: Note that at this point we have the result of the embedding lookup
        # for the entire batch on each device. We would like to obtain partial results
        # corresponding to all embedding lookups, but part of the batch on each device.
        # Therefore, matching the distribution of output of bottom mlp, so that both
        # could be used for subsequent interactions on each device.
        if len(self.emb_l) != len(ly):
            sys.exit("ERROR: corrupted intermediate result in parallel_forward call")

        t_list = []
        for k, _ in enumerate(self.emb_l):
            d = torch.device("cuda:" + str(k % ndevices))
            y = scatter(ly[k], device_ids, dim=0)
            t_list.append(y)
        # adjust the list to be ordered per device
        ly = list(map(lambda y: list(y), zip(*t_list)))
        # debug prints
        # print(ly)

        # interactions
        z = []
        for k in range(ndevices):
            zk = self.interact_features(x[k], ly[k])
            z.append(zk)
        # debug prints
        # print(z)

        # top mlp
        # WARNING: Note that the self.top_l is a list of top mlp modules that
        # have been replicated across devices, while z is a list of interaction results
        # that by construction are scattered across devices on the first (batch) dim.
        # The output is a list of tensors scattered across devices according to the
        # distribution of z.
        start_time = time.time()
        p = parallel_apply(self.top_l_replicas, z, None, device_ids)

        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        self.time_mlp+=elapsed_time

        ### gather the distributed results ###
        p0 = gather(p, self.output_d, dim=0)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z0 = torch.clamp(
                p0, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
            )
        else:
            z0 = p0

        return z0


def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return value


def dash_separated_floats(value):
    vals = value.split("-")
    for val in vals:
        try:
            float(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of floats" % value
            )

    return value


def inference(
    args,
    dlrm,
    best_acc_test,
    best_auc_test,
    test_ld,
    device,
    use_gpu,
    log_iter=-1,
):
    test_accu = 0
    test_samp = 0

    if args.mlperf_logging:
        scores = []
        targets = []

    # Around line 2802-2825, replace the inference batch unpacking:

    for i, testBatch in enumerate(test_ld):
        # early exit if nbatches was set by the user and was exceeded
        if nbatches > 0 and i >= nbatches:
            break

        if args.use_hotcold_preprocessed:
            # Hotcold preprocessed format: FIXED to include X and T!
            X_test, T_test, lS_o_hot, lS_i_hot, lS_o_cold, lS_i_cold, hotcold_mask = testBatch
            if i == 0:
                print("\n[DEBUG] Preprocessed batch format:")
                print(f"  X_test shape: {X_test.shape}")
                print(f"  T_test shape: {T_test.shape}")
                print(f"  lS_o_hot: {len(lS_o_hot)} tables")
                print(f"  lS_i_hot: {len(lS_i_hot)} tables")
                print(f"  lS_o_cold: {len(lS_o_cold)} tables")
                print(f"  lS_i_cold: {len(lS_i_cold)} tables")
                print(f"  hotcold_mask: {hotcold_mask}")
                
                # Check a hot/cold table (e.g., C3 = table_idx 2)
                table_idx = 2
                print(f"\n  Table C{table_idx+1} (hot/cold split):")
                print(f"    Hot indices count: {len(lS_i_hot[table_idx])}")
                print(f"    Cold indices count: {len(lS_i_cold[table_idx])}")
                print(f"    Hot offsets: {lS_o_hot[table_idx][:5]}... (first 5)")
                print(f"    Cold offsets: {lS_o_cold[table_idx][:5]}... (first 5)")
            
            if T_test.dim() == 1:
                T_test = T_test.view(-1, 1)
            # These are already included in preprocessed data
            W_test = torch.ones(T_test.size())
            CBPP_test = None
            
            # Store for preprocessed embedding lookup
            lS_o_test = lS_o_hot
            lS_i_test = lS_i_hot
        else:
            X_test, lS_o_test, lS_i_test, T_test, W_test, CBPP_test = unpack_batch(
                testBatch
            )

        # Skip the batch if batch size not multiple of total ranks
        if ext_dist.my_size > 1 and X_test.size(0) % ext_dist.my_size != 0:
            print("Warning: Skiping the batch %d with size %d" % (i, X_test.size(0)))
            continue

        # forward pass
        if args.use_hotcold_preprocessed:
            # Move to device
            if use_gpu:
                X_test = X_test.to(device)
                lS_o_hot = [S_o.to(device) for S_o in lS_o_hot]
                lS_i_hot = [S_i.to(device) for S_i in lS_i_hot]
                lS_o_cold = [S_o.to(device) for S_o in lS_o_cold]
                lS_i_cold = [S_i.to(device) for S_i in lS_i_cold]
            
            with record_function("DLRM forward"):
                Z_test = dlrm(X_test.to(device), lS_o_hot, lS_i_hot, 
                             lS_o_cold, lS_i_cold, hotcold_mask)
        else:
            Z_test = dlrm_wrap(
                X_test,
                lS_o_test,
                lS_i_test,
                use_gpu,
                device,
                ndevices=ndevices,
            )
        ### gather the distributed results on each rank ###
        # For some reason it requires explicit sync before all_gather call if
        # tensor is on GPU memory
        if Z_test.is_cuda:
            torch.cuda.synchronize()
        (_, batch_split_lengths) = ext_dist.get_split_lengths(X_test.size(0))
        if ext_dist.my_size > 1:
            Z_test = ext_dist.all_gather(Z_test, batch_split_lengths)

        if args.mlperf_logging:
            S_test = Z_test.detach().cpu().numpy()  # numpy array
            T_test = T_test.detach().cpu().numpy()  # numpy array
            scores.append(S_test)
            targets.append(T_test)
        else:
            with record_function("DLRM accuracy compute"):
                # compute loss and accuracy
                S_test = Z_test.detach().cpu().numpy()  # numpy array
                T_test = T_test.detach().cpu().numpy()  # numpy array

                mbs_test = T_test.shape[0]  # = mini_batch_size except last
                A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))

                test_accu += A_test
                test_samp += mbs_test

    if args.mlperf_logging:
        with record_function("DLRM mlperf sklearn metrics compute"):
            scores = np.concatenate(scores, axis=0)
            targets = np.concatenate(targets, axis=0)

            metrics = {
                "recall": lambda y_true, y_score: sklearn.metrics.recall_score(
                    y_true=y_true, y_pred=np.round(y_score)
                ),
                "precision": lambda y_true, y_score: sklearn.metrics.precision_score(
                    y_true=y_true, y_pred=np.round(y_score)
                ),
                "f1": lambda y_true, y_score: sklearn.metrics.f1_score(
                    y_true=y_true, y_pred=np.round(y_score)
                ),
                "ap": sklearn.metrics.average_precision_score,
                "roc_auc": sklearn.metrics.roc_auc_score,
                "accuracy": lambda y_true, y_score: sklearn.metrics.accuracy_score(
                    y_true=y_true, y_pred=np.round(y_score)
                ),
            }

        validation_results = {}
        for metric_name, metric_function in metrics.items():
            validation_results[metric_name] = metric_function(targets, scores)
            writer.add_scalar(
                "mlperf-metrics-test/" + metric_name,
                validation_results[metric_name],
                log_iter,
            )
        acc_test = validation_results["accuracy"]
    else:
        acc_test = test_accu / test_samp
        writer.add_scalar("Test/Acc", acc_test, log_iter)

    model_metrics_dict = {
        "nepochs": args.nepochs,
        "nbatches": nbatches,
        "nbatches_test": nbatches_test,
        "state_dict": dlrm.state_dict(),
        "test_acc": acc_test,
    }

    if args.mlperf_logging:
        is_best = validation_results["roc_auc"] > best_auc_test
        if is_best:
            best_auc_test = validation_results["roc_auc"]
            model_metrics_dict["test_auc"] = best_auc_test
        print(
            "recall {:.4f}, precision {:.4f},".format(
                validation_results["recall"],
                validation_results["precision"],
            )
            + " f1 {:.4f}, ap {:.4f},".format(
                validation_results["f1"], validation_results["ap"]
            )
            + " auc {:.4f}, best auc {:.4f},".format(
                validation_results["roc_auc"], best_auc_test
            )
            + " accuracy {:3.3f} %, best accuracy {:3.3f} %".format(
                validation_results["accuracy"] * 100, best_acc_test * 100
            ),
            flush=True,
        )
    else:
        is_best = acc_test > best_acc_test
        if is_best:
            best_acc_test = acc_test
        print(
            " accuracy {:3.3f} %, best {:3.3f} %".format(
                acc_test * 100, best_acc_test * 100
            ),
            flush=True,
        )
    return model_metrics_dict, is_best


def run():
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    # model related parameters
    parser.add_argument('--thread-count', type=int, default=None, help='Number of threads to use for intra-op')
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    parser.add_argument(
        "--arch-embedding-size", type=dash_separated_ints, default="4-3-2"
    )
    # j will be replaced with the table number
    parser.add_argument("--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=dash_separated_ints, default="4-2-1")
    parser.add_argument(
        "--arch-interaction-op", type=str, choices=["dot", "cat"], default="dot"
    )
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    parser.add_argument("--weighted-pooling", type=str, default=None)
    # embedding table options
    parser.add_argument("--md-flag", action="store_true", default=False)
    parser.add_argument("--md-threshold", type=int, default=200)
    parser.add_argument("--md-temperature", type=float, default=0.3)
    parser.add_argument("--md-round-dims", action="store_true", default=False)
    parser.add_argument("--qr-flag", action="store_true", default=False)
    parser.add_argument("--qr-threshold", type=int, default=200)
    parser.add_argument("--qr-operation", type=str, default="mult")
    parser.add_argument("--qr-collisions", type=int, default=4)
    # activations and loss
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--loss-function", type=str, default="mse")  # or bce or wbce
    parser.add_argument(
        "--loss-weights", type=dash_separated_floats, default="1.0-1.0"
    )  # for wbce
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
    parser.add_argument("--round-targets", type=bool, default=False)
    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument(
        "--data-generation",
        type=str,
        choices=["random", "dataset", "internal"],
        default="random",
    )  # synthetic, dataset or internal
    parser.add_argument(
        "--rand-data-dist", type=str, default="uniform"
    )  # uniform or gaussian
    parser.add_argument("--rand-data-min", type=float, default=0)
    parser.add_argument("--rand-data-max", type=float, default=1)
    parser.add_argument("--rand-data-mu", type=float, default=-1)
    parser.add_argument("--rand-data-sigma", type=float, default=1)
    parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--memory-map", action="store_true", default=False)
    # training
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=1)  #change it back to 1 when needed
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=1)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument(
        "--dataset-multiprocessing",
        action="store_true",
        default=False,
        help="The Kaggle dataset can be multiprocessed in an environment \
                        with more than 7 CPU cores and more than 20 GB of memory. \n \
                        The Terabyte dataset can be multiprocessed in an environment \
                        with more than 24 CPU cores and at least 1 TB of memory.",
    )
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # quantize
    parser.add_argument("--quantize-mlp-with-bit", type=int, default=32)
    parser.add_argument("--quantize-emb-with-bit", type=int, default=32)
    # onnx
    parser.add_argument("--save-onnx", action="store_true", default=False)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # distributed
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dist-backend", type=str, default="")
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument("--test-num-workers", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--print-wall-time", action="store_true", default=False)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    parser.add_argument("--plot-compute-graph", action="store_true", default=False)
    parser.add_argument("--tensor-board-filename", type=str, default="run_kaggle_pt")
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    # mlperf logging (disables other output and stops early)
    parser.add_argument("--mlperf-logging", action="store_true", default=False)
    # stop at target accuracy Kaggle 0.789, Terabyte (sub-sampled=0.875) 0.8107
    parser.add_argument("--mlperf-acc-threshold", type=float, default=0.0)
    # stop at target AUC Terabyte (no subsampling) 0.8025
    parser.add_argument("--mlperf-auc-threshold", type=float, default=0.0)
    parser.add_argument("--mlperf-bin-loader", action="store_true", default=False)
    parser.add_argument("--mlperf-bin-shuffle", action="store_true", default=False)
    # mlperf gradient accumulation iterations
    parser.add_argument("--mlperf-grad-accum-iter", type=int, default=1)
    # LR policy
    parser.add_argument("--lr-num-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-start-step", type=int, default=0)
    parser.add_argument("--lr-num-decay-steps", type=int, default=0)

    #merge threshold
# Add this to your argument parser
    parser.add_argument(
        "--split-emb-threshold",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--merge-emb-threshold",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num-splits",
        type=int,
        default=2,
        help="Number of sub-tables to split large tables into (default: 2)"
    )

    parser.add_argument(
        "--min-table-size",
        type=int,
        default=50,
        help="Minimum table size to be considered for merging (default: 50)"
    )
    parser.add_argument(
        "--min-split-rows",
        type=int,
        default=100000,
        help="Minimum number of rows for a table to be eligible for splitting (default: 100000)"
    )

    parser.add_argument(
        "--max-split-rows",
        type=int,
        default=5000000,
        help="Maximum number of rows for a table to be eligible for splitting (default: 5000000)"
    )

        # Add to argument parser:
    parser.add_argument(
        "--profile-embedding-access",
        action="store_true",
        default=False,
        help="Profile embedding access patterns (Phase 1)"
    )

    parser.add_argument(
        "--profile-batches",
        type=int,
        default=-1,
        help="Number of batches to profile for access patterns"
    )

    parser.add_argument(
        "--save-access-profile",
        type=str,
        default="",
        help="Save access profile to this file"
    )

    parser.add_argument(
        "--load-access-profile",
        type=str,
        default="",
        help="Load access profile from this file and create hot/cold splits"
    )

    parser.add_argument(
        "--hotcold-emb-threshold",
        type=int,
        default=1000000,
        help="Minimum table size for hot/cold splitting"
    )

    parser.add_argument(
        "--hotcold-percentile",
        type=int,
        default=80,
        help="Percentile for hot embeddings (e.g., 80 = embeddings covering 80%% of accesses)"
    )

    parser.add_argument(
        "--use-hotcold-preprocessed",
        action="store_true",
        default=False,
        help="Use hot/cold preprocessed data (works with any batch size)",
    )
    parser.add_argument(
        "--hotcold-preprocessed-dir",
        type=str,
        default="./preprocessed_hotcold_agnostic",
        help="Directory containing hot/cold preprocessed data",
    )


    global args
    global nbatches
    global nbatches_test
    global writer
    # After parsing args
    args = parser.parse_args()


        

    if args.dataset_multiprocessing:
        assert sys.version_info[0] >= 3 and sys.version_info[1] > 7, (
            "The dataset_multiprocessing "
            + "flag is susceptible to a bug in Python 3.7 and under. "
            + "https://github.com/facebookresearch/dlrm/issues/172"
        )

    if args.mlperf_logging:
        mlperf_logger.log_event(key=mlperf_logger.constants.CACHE_CLEAR, value=True)
        mlperf_logger.log_start(
            key=mlperf_logger.constants.INIT_START, log_all_ranks=True
        )

    if args.weighted_pooling is not None:
        if args.qr_flag:
            sys.exit("ERROR: quotient remainder with weighted pooling is not supported")
        if args.md_flag:
            sys.exit("ERROR: mixed dimensions with weighted pooling is not supported")
    if args.quantize_emb_with_bit in [4, 8]:
        if args.qr_flag:
            sys.exit(
                "ERROR: 4 and 8-bit quantization with quotient remainder is not supported"
            )
        if args.md_flag:
            sys.exit(
                "ERROR: 4 and 8-bit quantization with mixed dimensions is not supported"
            )
        if args.use_gpu:
            sys.exit("ERROR: 4 and 8-bit quantization on GPU is not supported")

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)

    if args.test_mini_batch_size < 0:
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size
    if args.test_num_workers < 0:
        # if the parameter is not set, use the same parameter for training
        args.test_num_workers = args.num_workers

    use_gpu = args.use_gpu and torch.cuda.is_available()

    if not args.debug_mode:
        ext_dist.init_distributed(
            local_rank=args.local_rank, use_gpu=use_gpu, backend=args.dist_backend
        )

    if use_gpu:
        torch.cuda.manual_seed_all(args.numpy_rand_seed)
        torch.backends.cudnn.deterministic = True
        if ext_dist.my_size > 1:
            ngpus = 1
            device = torch.device("cuda", ext_dist.my_local_rank)
        else:
            ngpus = torch.cuda.device_count()
            device = torch.device("cuda", 0)
        print("Using {} GPU(s)...".format(ngpus))
    else:
        device = torch.device("cpu")
        print("Using CPU...")

    ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    # input data

    if args.mlperf_logging:
        mlperf_logger.barrier()
        mlperf_logger.log_end(key=mlperf_logger.constants.INIT_STOP)
        mlperf_logger.barrier()
        mlperf_logger.log_start(key=mlperf_logger.constants.RUN_START)
        mlperf_logger.barrier()

    if args.use_hotcold_preprocessed:
        # Use hot/cold preprocessed data (works with any batch size!)
        print("[INFO] Using hot/cold preprocessed data")
        print(f"[INFO] Directory: {args.hotcold_preprocessed_dir}")
        print(f"[INFO] Batch size: {args.mini_batch_size} (can be changed without re-preprocessing)")
        
        train_data, train_ld, test_data, test_ld = make_hotcold_data_and_loaders(
            preprocessed_dir=args.hotcold_preprocessed_dir,
            batch_size=args.mini_batch_size,
            num_workers=args.num_workers
        )
        
        ln_emb = train_data.table_sizes
        m_den = 13  # Criteo has 13 dense features
        
        print(f"[INFO] Loaded {train_data.num_samples:,} training samples")
        print(f"[INFO] Generated {len(train_ld):,} batches")
        
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_test = len(test_ld)
        
        # Load metadata to get profile file path
        metadata_file = os.path.join(args.hotcold_preprocessed_dir, 'metadata.pkl')
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        # Load hot/cold profiles from the original analyzed profile
        profile_file = metadata['profile_file']
        if not os.path.isabs(profile_file):
            # Try relative to current directory first, then relative to preprocessed dir
            if not os.path.exists(profile_file):
                profile_file = os.path.join(args.hotcold_preprocessed_dir, os.path.basename(profile_file))
        
        print(f"[INFO] Loading hot/cold profiles from: {profile_file}")
        with open(profile_file, 'rb') as f:
            hot_cold_profiles = pickle.load(f)
        
        print(f"[INFO] Hot/cold split enabled for tables: {list(hot_cold_profiles.keys())}")
        for table_idx in sorted(hot_cold_profiles.keys()):
            profile = hot_cold_profiles[table_idx]
            print(f"[INFO]   C{table_idx+1}: hot={profile['hot_size']:,}, cold={profile['cold_size']:,}")
        
        # Set the threshold to enable hot/cold in DLRM_Net
        args.hotcold_emb_threshold = 1  # Enable for all tables in the profile
        
    elif args.data_generation == "dataset":
        train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
        table_feature_map = {idx: idx for idx in range(len(train_data.counts))}
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_test = len(test_ld)

        ln_emb = train_data.counts
        # enforce maximum limit on number of vectors per embedding
        if args.max_ind_range > 0:
            ln_emb = np.array(
                list(
                    map(
                        lambda x: x if x < args.max_ind_range else args.max_ind_range,
                        ln_emb,
                    )
                )
            )
        else:
            ln_emb = np.array(ln_emb)
        m_den = train_data.m_den
        ln_bot[0] = m_den
    elif args.data_generation == "internal":
        if not has_internal_libs:
            raise Exception("Internal libraries are not available.")
        NUM_BATCHES = 5000
        nbatches = args.num_batches if args.num_batches > 0 else NUM_BATCHES
        train_ld, feature_to_num_embeddings = fbDataLoader(args.data_size, nbatches)
        ln_emb = np.array(list(feature_to_num_embeddings.values()))
        m_den = ln_bot[0]
    else:
        # input and target at random
        ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
        m_den = ln_bot[0]
        train_data, train_ld, test_data, test_ld = dp.make_random_data_and_loader(
            args, ln_emb, m_den
        )
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_test = len(test_ld)

    args.ln_emb = ln_emb.tolist() if hasattr(ln_emb, "tolist") else list(ln_emb)
    if args.mlperf_logging:
        print("command line args: ", json.dumps(vars(args)))

    ### parse command line arguments ###
    m_spa = args.arch_sparse_feature_size
    ln_emb = np.asarray(ln_emb)
    num_fea = ln_emb.size + 1  # num sparse + num dense features

    m_den_out = ln_bot[ln_bot.size - 1]
    if args.arch_interaction_op == "dot":
        # approach 1: all
        # num_int = num_fea * num_fea + m_den_out
        # approach 2: unique
        if args.arch_interaction_itself:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    elif args.arch_interaction_op == "cat":
        num_int = num_fea * m_den_out
    else:
        sys.exit(
            "ERROR: --arch-interaction-op="
            + args.arch_interaction_op
            + " is not supported"
        )
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    # sanity check: feature sizes and mlp dimensions must match
    if m_den != ln_bot[0]:
        sys.exit(
            "ERROR: arch-dense-feature-size "
            + str(m_den)
            + " does not match first dim of bottom mlp "
            + str(ln_bot[0])
        )
    if args.qr_flag:
        if args.qr_operation == "concat" and 2 * m_spa != m_den_out:
            sys.exit(
                "ERROR: 2 arch-sparse-feature-size "
                + str(2 * m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
                + " (note that the last dim of bottom mlp must be 2x the embedding dim)"
            )
        if args.qr_operation != "concat" and m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    else:
        if m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    if num_int != ln_top[0]:
        sys.exit(
            "ERROR: # of feature interactions "
            + str(num_int)
            + " does not match first dimension of top mlp "
            + str(ln_top[0])
        )

    # assign mixed dimensions if applicable
    if args.md_flag:
        m_spa = md_solver(
            torch.tensor(ln_emb),
            args.md_temperature,  # alpha
            d0=m_spa,
            round_dim=args.md_round_dims,
        ).tolist()

    # test prints (model arch)
    if args.debug_mode:
        print("model arch:")
        print(
            "mlp top arch "
            + str(ln_top.size - 1)
            + " layers, with input to output dimensions:"
        )
        print(ln_top)
        print("# of interactions")
        print(num_int)
        print(
            "mlp bot arch "
            + str(ln_bot.size - 1)
            + " layers, with input to output dimensions:"
        )
        print(ln_bot)
        print("# of features (sparse and dense)")
        print(num_fea)
        print("dense feature size")
        print(m_den)
        print("sparse feature size")
        print(m_spa)
        print(
            "# of embeddings (= # of sparse features) "
            + str(ln_emb.size)
            + ", with dimensions "
            + str(m_spa)
            + "x:"
        )
        print(ln_emb)

        print("data (inputs and targets):")
        for j, inputBatch in enumerate(train_ld):
            if j == 0 and args.save_onnx and not args.use_hotcold_preprocessed:
                X_onnx, lS_o_onnx, lS_i_onnx, _, _, _ = unpack_batch(inputBatch)

            if j < skip_upto_batch:
                continue

            if args.use_hotcold_preprocessed:
                # Hotcold preprocessed format includes X and T
                X, T, lS_o_hot, lS_i_hot, lS_o_cold, lS_i_cold, hotcold_mask = inputBatch

                if T.dim() == 1:
                    T = T.view(-1, 1)  # [batch_size] → [batch_size, 1]
                
                W = torch.ones(T.size())
                CBPP = None
                
                lS_o = lS_o_hot
                lS_i = lS_i_hot
            else:
                X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)

            torch.set_printoptions(precision=4)
            # early exit if nbatches was set by the user and has been exceeded
            if nbatches > 0 and j >= nbatches:
                break
            print("mini-batch: %d" % j)
            print(X.detach().cpu())
            # transform offsets to lengths when printing
            print(
                torch.IntTensor(
                    [
                        np.diff(
                            S_o.detach().cpu().tolist() + list(lS_i[i].shape)
                        ).tolist()
                        for i, S_o in enumerate(lS_o)
                    ]
                )
            )
            print([S_i.detach().cpu() for S_i in lS_i])
            print(T.detach().cpu())

    global ndevices
    ndevices = min(ngpus, args.mini_batch_size, num_fea - 1) if use_gpu else -1

    ### construct the neural network specified above ###
    # WARNING: to obtain exactly the same initialization for
    # the weights we need to start from the same random seed.
    # np.random.seed(args.numpy_rand_seed)
    global last_time_look_up
    global last_time_mlp
    global last_time_interact
    global dlrm

    last_time_look_up = 0
    last_time_mlp = 0
    last_time_interact = 0
    # Determine if we should use hot/cold profiles
    if args.use_hotcold_preprocessed:
        hotcold_threshold_val = args.hotcold_emb_threshold
        hotcold_profiles_val = hot_cold_profiles
    else:
        hotcold_threshold_val = 0
        hotcold_profiles_val = None
    
    dlrm = DLRM_Net(
        m_spa,
        ln_emb,
        ln_bot,
        ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        sync_dense_params=args.sync_dense_params,
        loss_threshold=args.loss_threshold,
        ndevices=ndevices,
        qr_flag=args.qr_flag,
        qr_operation=args.qr_operation,
        qr_collisions=args.qr_collisions,
        qr_threshold=args.qr_threshold,
        md_flag=args.md_flag,
        md_threshold=args.md_threshold,
        weighted_pooling=args.weighted_pooling,
        loss_function=args.loss_function,
        thread_count=args.thread_count,
        num_splits=args.num_splits,
        merge_threshold=args.merge_emb_threshold,    
        split_threshold=args.split_emb_threshold,    
        min_table_size=args.min_table_size,
        min_split_rows=args.min_split_rows,
        max_split_rows=args.max_split_rows,
        hotcold_threshold=hotcold_threshold_val,
        hot_cold_profiles=hotcold_profiles_val,
    )

    # test prints
    if args.debug_mode:
        print("initial parameters (weights and bias):")
        for param in dlrm.parameters():
            print(param.detach().cpu().numpy())
        # print(dlrm)

    if use_gpu:
        # Custom Model-Data Parallel
        # the mlps are replicated and use data parallelism, while
        # the embeddings are distributed and use model parallelism
        dlrm = dlrm.to(device)  # .cuda()
        if dlrm.ndevices > 1:
            dlrm.emb_l, dlrm.v_W_l = dlrm.create_emb(
                m_spa, ln_emb, args.weighted_pooling
            )
        else:
            if dlrm.weighted_pooling == "fixed":
                for k, w in enumerate(dlrm.v_W_l):
                    dlrm.v_W_l[k] = w.cuda()

    # distribute data parallel mlps
    if ext_dist.my_size > 1:
        if use_gpu:
            device_ids = [ext_dist.my_local_rank]
            dlrm.bot_l = ext_dist.DDP(dlrm.bot_l, device_ids=device_ids)
            dlrm.top_l = ext_dist.DDP(dlrm.top_l, device_ids=device_ids)
        else:
            dlrm.bot_l = ext_dist.DDP(dlrm.bot_l)
            dlrm.top_l = ext_dist.DDP(dlrm.top_l)

    if not args.inference_only:
        if use_gpu and args.optimizer in ["rwsadagrad", "adagrad"]:
            sys.exit("GPU version of Adagrad is not supported by PyTorch.")
        # specify the optimizer algorithm
        opts = {
            "sgd": torch.optim.SGD,
            "rwsadagrad": RowWiseSparseAdagrad.RWSAdagrad,
            "adagrad": torch.optim.Adagrad,
        }

        parameters = (
            dlrm.parameters()
            if ext_dist.my_size == 1
            else [
                {
                    "params": [p for emb in dlrm.emb_l for p in emb.parameters()],
                    "lr": args.learning_rate,
                },
                # TODO check this lr setup
                # bottom mlp has no data parallelism
                # need to check how do we deal with top mlp
                {
                    "params": dlrm.bot_l.parameters(),
                    "lr": args.learning_rate,
                },
                {
                    "params": dlrm.top_l.parameters(),
                    "lr": args.learning_rate,
                },
            ]
        )
        optimizer = opts[args.optimizer](parameters, lr=args.learning_rate)
        lr_scheduler = LRPolicyScheduler(
            optimizer,
            args.lr_num_warmup_steps,
            args.lr_decay_start_step,
            args.lr_num_decay_steps,
        )

    ### main loop ###

    # training or inference
    best_acc_test = 0
    best_auc_test = 0
    skip_upto_epoch = 0
    skip_upto_batch = 0
    total_time = 0
    total_loss = 0
    total_iter = 0
    total_samp = 0

    if args.mlperf_logging:
        mlperf_logger.mlperf_submission_log("dlrm")
        mlperf_logger.log_event(
            key=mlperf_logger.constants.SEED, value=args.numpy_rand_seed
        )
        mlperf_logger.log_event(
            key=mlperf_logger.constants.GLOBAL_BATCH_SIZE, value=args.mini_batch_size
        )

    # Load model is specified
    if not (args.load_model == ""):
        print("Loading saved model {}".format(args.load_model))
        if use_gpu:
            if dlrm.ndevices > 1:
                # NOTE: when targeting inference on multiple GPUs,
                # load the model as is on CPU or GPU, with the move
                # to multiple GPUs to be done in parallel_forward
                ld_model = torch.load(args.load_model)
            else:
                # NOTE: when targeting inference on single GPU,
                # note that the call to .to(device) has already happened
                ld_model = torch.load(
                    args.load_model,
                    map_location=torch.device("cuda"),
                    # map_location=lambda storage, loc: storage.cuda(0)
                )
        else:
            # when targeting inference on CPU
            ld_model = torch.load(args.load_model, map_location=torch.device("cpu"))
        dlrm.load_state_dict(ld_model["state_dict"])
        ld_j = ld_model["iter"]
        ld_k = ld_model["epoch"]
        ld_nepochs = ld_model["nepochs"]
        ld_nbatches = ld_model["nbatches"]
        ld_nbatches_test = ld_model["nbatches_test"]
        ld_train_loss = ld_model["train_loss"]
        ld_total_loss = ld_model["total_loss"]
        if args.mlperf_logging:
            ld_gAUC_test = ld_model["test_auc"]
        ld_acc_test = ld_model["test_acc"]
        if not args.inference_only:
            optimizer.load_state_dict(ld_model["opt_state_dict"])
            best_acc_test = ld_acc_test
            total_loss = ld_total_loss
            skip_upto_epoch = ld_k  # epochs
            skip_upto_batch = ld_j  # batches
        else:
            args.print_freq = ld_nbatches
            args.test_freq = 0

        print(
            "Saved at: epoch = {:d}/{:d}, batch = {:d}/{:d}, ntbatch = {:d}".format(
                ld_k, ld_nepochs, ld_j, ld_nbatches, ld_nbatches_test
            )
        )
        print(
            "Training state: loss = {:.6f}".format(
                ld_train_loss,
            )
        )
        if args.mlperf_logging:
            print(
                "Testing state: accuracy = {:3.3f} %, auc = {:.3f}".format(
                    ld_acc_test * 100, ld_gAUC_test
                )
            )
        else:
            print("Testing state: accuracy = {:3.3f} %".format(ld_acc_test * 100))

    if args.inference_only:
        # Currently only dynamic quantization with INT8 and FP16 weights are
        # supported for MLPs and INT4 and INT8 weights for EmbeddingBag
        # post-training quantization during the inference.
        # By default we don't do the quantization: quantize_{mlp,emb}_with_bit == 32 (FP32)
        assert args.quantize_mlp_with_bit in [
            8,
            16,
            32,
        ], "only support 8/16/32-bit but got {}".format(args.quantize_mlp_with_bit)
        assert args.quantize_emb_with_bit in [
            4,
            8,
            32,
        ], "only support 4/8/32-bit but got {}".format(args.quantize_emb_with_bit)
        if args.quantize_mlp_with_bit != 32:
            if args.quantize_mlp_with_bit in [8]:
                quantize_dtype = torch.qint8
            else:
                quantize_dtype = torch.float16
            dlrm = torch.quantization.quantize_dynamic(
                dlrm, {torch.nn.Linear}, quantize_dtype
            )
        if args.quantize_emb_with_bit != 32:
            dlrm.quantize_embedding(args.quantize_emb_with_bit)
            # print(dlrm)

    print("time/loss/accuracy (if enabled):")

    hot_cold_profiles = None
    
    if args.use_hotcold_preprocessed:
        pass  # Model already created with hot/cold profiles above
    # PHASE 1: PROFILING MODE - Record access patterns and exit
    elif args.profile_embedding_access:
        print("\n" + "="*80)
        print("PHASE 1: PROFILING MODE - Recording Embedding Access Patterns")
        print("="*80)
        print(f"[PROFILE] Batch size: {args.mini_batch_size}")
        if args.profile_batches <= 0:
            total_train_batches = len(train_ld)
            print(f"[PROFILE] Auto-detected {total_train_batches} training batches (profiling ALL)")
            args.profile_batches = total_train_batches
        else:
            print(f"[PROFILE] Batches to profile: {args.profile_batches}")
        print(f"[PROFILE] Size threshold: {args.hotcold_emb_threshold}")
        print(f"[PROFILE] Output file: {args.save_access_profile}")
        print("="*80 + "\n")
        
        # Create profiler
        profiler = EmbeddingAccessProfiler(len(ln_emb))
        
        # Create DLRM with profiler (no hot/cold tables yet)
        dlrm = DLRM_Net(
            m_spa,
            ln_emb,
            ln_bot,
            ln_top,
            sigmoid_bot = args.sigmoid_bot if hasattr(args, 'sigmoid_bot') else -1,
            sigmoid_top = args.sigmoid_top if hasattr(args, 'sigmoid_top') else -1,
            arch_interaction_op=args.arch_interaction_op,
            arch_interaction_itself=args.arch_interaction_itself,
            qr_flag=args.qr_flag,
            qr_operation=args.qr_operation,
            qr_collisions=args.qr_collisions,
            qr_threshold=args.qr_threshold,
            md_flag=args.md_flag,
            md_threshold=args.md_threshold,
            weighted_pooling=args.weighted_pooling,
            loss_function=args.loss_function,
            loss_threshold=args.loss_threshold,
            merge_threshold=args.merge_emb_threshold,  
            split_threshold=args.split_emb_threshold,  
            min_table_size=args.min_table_size,
            min_split_rows=args.min_split_rows,
            max_split_rows=args.max_split_rows,
            num_splits=args.num_splits,
            hotcold_threshold=0,  # Disable hot/cold during profiling
            hot_cold_profiles=None,  # No profiles yet
            access_profiler=profiler  # Pass profiler for recording
        )
        
        # Move to device
        if use_gpu:
            dlrm = dlrm.to(device)
        
        print(f"\n[PROFILE] Starting to profile {args.profile_batches} batches...")
        print(f"[PROFILE] This will record which embeddings are accessed")
        print(f"[PROFILE] No training is performed, just inference\n")
        
        # Run inference to collect access patterns
        dlrm.eval()  # Set to eval mode
        batch_count = 0
        
        with torch.no_grad():
            for j, (X, lS_o, lS_i, T) in enumerate(train_ld):
                if args.profile_batches > 0 and batch_count >= args.profile_batches:
                    break
                
                # Move data to device
                if use_gpu:
                    X = X.to(device)
                    lS_o = [S_o.to(device) for S_o in lS_o]
                    lS_i = [S_i.to(device) for S_i in lS_i]
                
                # Forward pass (profiler records accesses inside apply_emb)
                Z = dlrm(X, lS_o, lS_i)
                
                batch_count += 1
                
                if batch_count % 500 == 0:
                    print(f"[PROFILE] Processed {batch_count}/{args.profile_batches} batches")
        
        print(f"\n[PROFILE] Profiling complete! Processed {batch_count} batches")
        
        # Analyze and save profiles
        print(f"\n[PROFILE] Analyzing access patterns...")
        print(f"[PROFILE]   Percentile: {args.hotcold_percentile}%")
        print(f"[PROFILE]   Min table size: {args.hotcold_emb_threshold}")
        print(f"\n[DEBUG] Before analyze_all:")
        print(f"  ln_emb type: {type(ln_emb)}, length: {len(ln_emb)}")
        print(f"  ln_emb[:5]: {ln_emb[:5]}")
        print(f"  hotcold_percentile: {args.hotcold_percentile}")
        print(f"  hotcold_emb_threshold: {args.hotcold_emb_threshold}")
        print(f"  profiler has access_counts: {hasattr(profiler, 'access_counts')}")
        if hasattr(profiler, 'access_counts'):
            print(f"  access_counts length: {len(profiler.access_counts)}")
            print(f"  Table 2 (C3) has data: {len(profiler.access_counts[2]) if profiler.access_counts[2] else 0} unique embeddings")
        
        hot_cold_profiles = profiler.analyze_all(
            ln_emb, 
            args.hotcold_percentile, 
            args.hotcold_emb_threshold
        )

                # ADD DEBUG AFTER
        print(f"\n[DEBUG] After analyze_all:")
        print(f"  Result type: {type(hot_cold_profiles)}")
        print(f"  Result length: {len(hot_cold_profiles)}")
        print(f"  Result keys: {list(hot_cold_profiles.keys())}")
        
        # Save raw profile
        print(f"\n[PROFILE] Saving profiles...")
        profiler.save(args.save_access_profile)
        print(f"[PROFILE]   Raw profile: {args.save_access_profile}")
        
        # Save analyzed profile
        analyzed_file = args.save_access_profile.replace('.pkl', '_analyzed.pkl')
        with open(analyzed_file, 'wb') as f:
            pickle.dump(hot_cold_profiles, f)
        print(f"[PROFILE]   Analyzed profile: {analyzed_file}")
        
        print("\n" + "="*80)
        print("PROFILING COMPLETE!")
        print("="*80)
        print(f"Profile files created:")
        print(f"  - {args.save_access_profile}")
        print(f"  - {analyzed_file}")
        print(f"\nUse these profiles with --load-access-profile for testing")
        print("="*80 + "\n")
        
        sys.exit(0)  # Exit after profiling
    
    # PHASE 2: TESTING MODE - Load existing profile and use hot/cold split
    elif args.load_access_profile:
        print("\n" + "="*80)
        print("PHASE 2: TESTING MODE - Using Hot/Cold Split")
        print("="*80)
        print(f"[HOTCOLD] Loading profile: {args.load_access_profile}")
        print(f"[HOTCOLD] Percentile: {args.hotcold_percentile}%")
        print(f"[HOTCOLD] Size threshold: {args.hotcold_emb_threshold}")
        print("="*80 + "\n")
        
        # Load analyzed profile
        analyzed_file = args.load_access_profile.replace('.pkl', '_analyzed.pkl')
        
        if not os.path.exists(analyzed_file):
            print(f"[ERROR] Analyzed profile not found: {analyzed_file}")
            print(f"[ERROR] Make sure profiling was completed successfully")
            sys.exit(1)
        
        with open(analyzed_file, 'rb') as f:
            hot_cold_profiles = pickle.load(f)
        
        print(f"[HOTCOLD] Successfully loaded profiles for {len(hot_cold_profiles)} tables:")
        for table_idx in sorted(hot_cold_profiles.keys()):
            profile = hot_cold_profiles[table_idx]
            hot_size = profile['hot_size']
            cold_size = profile['cold_size']
            total_size = hot_size + cold_size
            hot_pct = (hot_size / total_size) * 100 if total_size > 0 else 0
            hot_mb = (hot_size * m_spa * 4) / (1024 * 1024)
            print(f"[HOTCOLD]   C{table_idx+1}: {hot_size:,} hot ({hot_pct:.2f}%, {hot_mb:.1f}MB) + {cold_size:,} cold")
        print()
        
        # Create DLRM with hot/cold tables (NO profiler)
        dlrm = DLRM_Net(
            m_spa,
            ln_emb,
            ln_bot,
            ln_top,
            sigmoid_bot = args.sigmoid_bot if hasattr(args, 'sigmoid_bot') else -1,
            sigmoid_top = args.sigmoid_top if hasattr(args, 'sigmoid_top') else -1,
            arch_interaction_op=args.arch_interaction_op,
            arch_interaction_itself=args.arch_interaction_itself,
            qr_flag=args.qr_flag,
            qr_operation=args.qr_operation,
            qr_collisions=args.qr_collisions,
            qr_threshold=args.qr_threshold,
            md_flag=args.md_flag,
            md_threshold=args.md_threshold,
            weighted_pooling=args.weighted_pooling,
            loss_function=args.loss_function,
            loss_threshold=args.loss_threshold,
            merge_threshold=args.merge_emb_threshold, 
            split_threshold=args.split_emb_threshold,  
            min_table_size=args.min_table_size,
            min_split_rows=args.min_split_rows,
            max_split_rows=args.max_split_rows,
            num_splits=args.num_splits,
            hotcold_threshold=args.hotcold_emb_threshold,  # Enable hot/cold
            hot_cold_profiles=hot_cold_profiles,  # Use loaded profiles
            access_profiler=None  # NO PROFILER in test mode
        )
    
    # PHASE 3: BASELINE MODE - No profiling, no hot/cold split
    else:
        print("\n" + "="*80)
        print("PHASE 3: BASELINE MODE - Standard DLRM (No Hot/Cold Split)")
        print("="*80 + "\n")
        
        # Create standard DLRM (no profiler, no hot/cold)
        dlrm = DLRM_Net(
            m_spa,
            ln_emb,
            ln_bot,
            ln_top,
            sigmoid_bot = args.sigmoid_bot if hasattr(args, 'sigmoid_bot') else -1,
            sigmoid_top = args.sigmoid_top if hasattr(args, 'sigmoid_top') else -1,
            arch_interaction_op=args.arch_interaction_op,
            arch_interaction_itself=args.arch_interaction_itself,
            qr_flag=args.qr_flag,
            qr_operation=args.qr_operation,
            qr_collisions=args.qr_collisions,
            qr_threshold=args.qr_threshold,
            md_flag=args.md_flag,
            md_threshold=args.md_threshold,
            weighted_pooling=args.weighted_pooling,
            loss_function=args.loss_function,
            loss_threshold=args.loss_threshold,
            merge_threshold=args.merge_emb_threshold, 
            split_threshold=args.split_emb_threshold,  
            min_table_size=args.min_table_size,
            min_split_rows=args.min_split_rows,
            max_split_rows=args.max_split_rows,
            num_splits=args.num_splits,
            hotcold_threshold=0,  # Disable hot/cold
            hot_cold_profiles=None,  # No profiles
            access_profiler=None  # NO PROFILER
        )
    
    # ============================================================================
    # MOVE MODEL TO DEVICE
    # ============================================================================
    
    if use_gpu:
        if dlrm.ndevices > 1:
            dlrm = dlrm.to(device)  # .cuda() # moves to "cuda:0"
        else:
            dlrm = dlrm.to(device)
        
        # Distribute data parallel mlps (if distributed)
        if ext_dist.my_size > 1:
            if use_gpu:
                device_ids = [ext_dist.my_local_rank]
                dlrm.bot_l = ext_dist.DDP(dlrm.bot_l, device_ids=device_ids)
                dlrm.top_l = ext_dist.DDP(dlrm.top_l, device_ids=device_ids)
            else:
                dlrm.bot_l = ext_dist.DDP(dlrm.bot_l)
                dlrm.top_l = ext_dist.DDP(dlrm.top_l)
        
        # Reinitialize optimizer with new parameters
        if not args.inference_only:
            parameters = (
                dlrm.parameters()
                if ext_dist.my_size == 1
                else [
                    {
                        "params": [p for emb in dlrm.emb_l for p in emb.parameters()],
                        "lr": args.learning_rate,
                    },
                    {
                        "params": dlrm.bot_l.parameters(),
                        "lr": args.learning_rate,
                    },
                    {
                        "params": dlrm.top_l.parameters(),
                        "lr": args.learning_rate,
                    },
                ]
            )
            optimizer = opts[args.optimizer](parameters, lr=args.learning_rate)
            lr_scheduler = LRPolicyScheduler(
                optimizer,
                args.lr_num_warmup_steps,
                args.lr_decay_start_step,
                args.lr_num_decay_steps,
            )
        
        print("[HOTCOLD] Model recreated with hot/cold splits\n")
        print("="*80 + "\n")

    if args.mlperf_logging:
        # LR is logged twice for now because of a compliance checker bug
        mlperf_logger.log_event(
            key=mlperf_logger.constants.OPT_BASE_LR, value=args.learning_rate
        )
        mlperf_logger.log_event(
            key=mlperf_logger.constants.OPT_LR_WARMUP_STEPS,
            value=args.lr_num_warmup_steps,
        )

        # use logging keys from the official HP table and not from the logging library
        mlperf_logger.log_event(
            key="sgd_opt_base_learning_rate", value=args.learning_rate
        )
        mlperf_logger.log_event(
            key="lr_decay_start_steps", value=args.lr_decay_start_step
        )
        mlperf_logger.log_event(
            key="sgd_opt_learning_rate_decay_steps", value=args.lr_num_decay_steps
        )
        mlperf_logger.log_event(key="sgd_opt_learning_rate_decay_poly_power", value=2)

    tb_file = "./" + args.tensor_board_filename
    writer = SummaryWriter(tb_file)

    ext_dist.barrier()
    with torch.autograd.profiler.profile(
        args.enable_profiling, use_cuda=use_gpu, record_shapes=True
    ) as prof:
        if args.thread_count:
            torch.set_num_threads(args.thread_count)
        start_time = time_wrap(use_gpu)
        print("PyTorch Intra-op threads:", torch.get_num_threads())
        if not args.inference_only:
            k = 0
            total_time_begin = 0
            while k < args.nepochs:
                if args.mlperf_logging:
                    mlperf_logger.barrier()
                    mlperf_logger.log_start(
                        key=mlperf_logger.constants.BLOCK_START,
                        metadata={
                            mlperf_logger.constants.FIRST_EPOCH_NUM: (k + 1),
                            mlperf_logger.constants.EPOCH_COUNT: 1,
                        },
                    )
                    mlperf_logger.barrier()
                    mlperf_logger.log_start(
                        key=mlperf_logger.constants.EPOCH_START,
                        metadata={mlperf_logger.constants.EPOCH_NUM: (k + 1)},
                    )

                if k < skip_upto_epoch:
                    continue

                if args.mlperf_logging:
                    previous_iteration_time = None

                #where timer started   
                
                total_dlrm = 0
                total_dlrm_back = 0
                total_dlrm_backpass = 0

                for j, inputBatch in enumerate(train_ld):
                    if j == 0 and args.save_onnx:
                        if not args.use_hotcold_preprocessed:
                            X_onnx, lS_o_onnx, lS_i_onnx, _, _, _ = unpack_batch(inputBatch)

                    if j < skip_upto_batch:
                        continue

                    # Unpack batch based on data source
                    if args.use_hotcold_preprocessed:
                        # Preprocessed hot/cold format: 7-element tuple
                        X, T, lS_o_hot, lS_i_hot, lS_o_cold, lS_i_cold, hotcold_mask = inputBatch
                        
                        # Reshape T to match expected format [batch_size, 1]
                        if T.dim() == 1:
                            T = T.view(-1, 1)
                        
                        # Use hot lists for standard DLRM forward pass
                        lS_o = lS_o_hot
                        lS_i = lS_i_hot
                        
                        # Create dummy weight and CBPP
                        W = torch.ones(T.size())
                        CBPP = None
                    else:
                        # Standard unpacking
                        X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)

                    if args.mlperf_logging:
                        current_time = time_wrap(use_gpu)
                        if previous_iteration_time:
                            iteration_time = current_time - previous_iteration_time
                        else:
                            iteration_time = 0
                        previous_iteration_time = current_time
                    else:
                        t1 = time_wrap(use_gpu)

                    # early exit if nbatches was set by the user and has been exceeded
                    if nbatches > 0 and j >= nbatches:
                        break

                    # Skip the batch if batch size not multiple of total ranks
                    if ext_dist.my_size > 1 and X.size(0) % ext_dist.my_size != 0:
                        print(
                            "Warning: Skiping the batch %d with size %d"
                            % (j, X.size(0))
                        )
                        continue

                    mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
                    
                    #start the timer for the top mlp
                    # start_time = time.time()
                    # forward pass
                    # forward pass
                    dlrm_time_1 = time_wrap(use_gpu)
                    if args.use_hotcold_preprocessed:
                        # Move to device
                        if use_gpu:
                            X = X.to(device)
                            lS_o_hot = [S_o.to(device) for S_o in lS_o_hot]
                            lS_i_hot = [S_i.to(device) for S_i in lS_i_hot]
                            lS_o_cold = [S_o.to(device) for S_o in lS_o_cold]
                            lS_i_cold = [S_i.to(device) for S_i in lS_i_cold]
                        
                        with record_function("DLRM forward"):
                            Z = dlrm(X.to(device), lS_o_hot, lS_i_hot,
                                    lS_o_cold, lS_i_cold, hotcold_mask)
                    else:
                        Z = dlrm_wrap(
                            X,
                            lS_o,
                            lS_i,
                            use_gpu,
                            device,
                            ndevices=ndevices,
                        )
                    dlrm_time_2 = time_wrap(use_gpu)
                    total_dlrm += (dlrm_time_2 -dlrm_time_1)
                    

                    if ext_dist.my_size > 1:
                        T = T[ext_dist.get_my_slice(mbs)]
                        W = W[ext_dist.get_my_slice(mbs)]

                    # loss
                    E = loss_fn_wrap(Z, T, use_gpu, device)

                    # compute loss and accuracy
                    L = E.detach().cpu().numpy()  # numpy array
                    # training accuracy is not disabled
                    # S = Z.detach().cpu().numpy()  # numpy array
                    # T = T.detach().cpu().numpy()  # numpy array

                    # # print("res: ", S)

                    # # print("j, train: BCE ", j, L)

                    # mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
                    # A = np.sum((np.round(S, 0) == T).astype(np.uint8))
                    dlrm_back_1 = time_wrap(use_gpu)
                    with record_function("DLRM backward"):
                        # scaled error gradient propagation
                        # (where we do not accumulate gradients across mini-batches)
                        if (
                            args.mlperf_logging
                            and (j + 1) % args.mlperf_grad_accum_iter == 0
                        ) or not args.mlperf_logging:
                            optimizer.zero_grad()
                        # backward pass
                        dlrm_backpass_1 = time_wrap(use_gpu)
                        E.backward()
                        dlrm_backpass_2 = time_wrap(use_gpu)
                        total_dlrm_backpass += (dlrm_backpass_2 - dlrm_backpass_1)

                        # optimizer
                        if (
                            args.mlperf_logging
                            and (j + 1) % args.mlperf_grad_accum_iter == 0
                        ) or not args.mlperf_logging:
                            optimizer.step()
                            lr_scheduler.step()
                    dlrm_back_2 = time_wrap(use_gpu)
                    total_dlrm_back += (dlrm_back_2 - dlrm_back_1)
                    # # Stop the timer after MLP construction
                    # end_time = time.time()

                    # # Calculate the elapsed time
                    # elapsed_time = end_time - start_time
                    # self.time_mlp+=elapsed_time

                    if args.mlperf_logging:
                        total_time += iteration_time
                    else:
                        t2 = time_wrap(use_gpu)
                        total_time += t2 - t1

                    total_loss += L * mbs
                    total_iter += 1
                    total_samp += mbs

                    should_print = ((j + 1) % args.print_freq == 0) or (
                        j + 1 == nbatches
                    )
                    should_test = (
                        (args.test_freq > 0)
                        and (args.data_generation in ["dataset", "random"])
                        and (((j + 1) % args.test_freq == 0) or (j + 1 == nbatches))
                    )

                    # print time, loss and accuracy
                    if should_print or should_test:
                        gT = 1000.0 * total_time / total_iter if args.print_time else -1
                        total_time = 0

                        train_loss = total_loss / total_samp
                        total_loss = 0

                        str_run_type = (
                            "inference" if args.inference_only else "training"
                        )

                        wall_time = ""
                        if args.print_wall_time:
                            wall_time = " ({})".format(time.strftime("%H:%M"))
                        time_dlrm = total_dlrm* 1000/total_iter
                        time_dlrm_back_iteration = total_dlrm_back *1000 /total_iter
                        time_dlrm_backpass_iteration = total_dlrm_backpass *1000 /total_iter
                        time_diff_look_up = (dlrm.time_look_up - last_time_look_up) *1000 /total_iter
                        time_diff_mlp = (dlrm.time_mlp - last_time_mlp) * 1000 / total_iter
                        time_diff_interact = (dlrm.time_interact - last_time_interact) * 1000 / total_iter
                        print(
                            "Finished {} it {}/{} of epoch {}, {:.2f} ms/it,".format(
                                str_run_type, j + 1, nbatches, k, gT
                            )
                            + " loss {:.6f}".format(train_loss)
                            + wall_time,
                            flush=True,
                        )
                        print(f"Embedding lookup time: {time_diff_look_up:.2f} ms, MLP time: {time_diff_mlp:.2f} ms, interaction time: {time_diff_interact:.2f} ms \n"
                        f"Running dlrm: {time_dlrm:.2f} ms, backpropagation after dlrm: {time_dlrm_back_iteration:.2f} ms, backpass: {time_dlrm_backpass_iteration:.2f} ms")

                        last_time_look_up = dlrm.time_look_up
                        last_time_mlp = dlrm.time_mlp
                        last_time_interact = dlrm.time_interact
                        total_dlrm = 0
                        total_dlrm_back=0
                        total_dlrm_backpass=0

                        log_iter = nbatches * k + j + 1
                        writer.add_scalar("Train/Loss", train_loss, log_iter)

                        total_iter = 0
                        total_samp = 0

                    # testing
                    if should_test:
                        epoch_num_float = (j + 1) / len(train_ld) + k + 1
                        if args.mlperf_logging:
                            mlperf_logger.barrier()
                            mlperf_logger.log_start(
                                key=mlperf_logger.constants.EVAL_START,
                                metadata={
                                    mlperf_logger.constants.EPOCH_NUM: epoch_num_float
                                },
                            )

                        # don't measure training iter time in a test iteration
                        if args.mlperf_logging:
                            previous_iteration_time = None
                        print(
                            "Testing at - {}/{} of epoch {},".format(j + 1, nbatches, k)
                        )
                        model_metrics_dict, is_best = inference(
                            args,
                            dlrm,
                            best_acc_test,
                            best_auc_test,
                            test_ld,
                            device,
                            use_gpu,
                            log_iter,
                        )

                        if (
                            is_best
                            and not (args.save_model == "")
                            and not args.inference_only
                        ):
                            model_metrics_dict["epoch"] = k
                            model_metrics_dict["iter"] = j + 1
                            model_metrics_dict["train_loss"] = train_loss
                            model_metrics_dict["total_loss"] = total_loss
                            model_metrics_dict["opt_state_dict"] = (
                                optimizer.state_dict()
                            )
                            print("Saving model to {}".format(args.save_model))
                            torch.save(model_metrics_dict, args.save_model)

                        if args.mlperf_logging:
                            mlperf_logger.barrier()
                            mlperf_logger.log_end(
                                key=mlperf_logger.constants.EVAL_STOP,
                                metadata={
                                    mlperf_logger.constants.EPOCH_NUM: epoch_num_float
                                },
                            )

                        # Uncomment the line below to print out the total time with overhead
                        # print("Total test time for this group: {}" \
                        # .format(time_wrap(use_gpu) - accum_test_time_begin))

                        if (
                            args.mlperf_logging
                            and (args.mlperf_acc_threshold > 0)
                            and (best_acc_test > args.mlperf_acc_threshold)
                        ):
                            print(
                                "MLPerf testing accuracy threshold "
                                + str(args.mlperf_acc_threshold)
                                + " reached, stop training"
                            )
                            break

                        if (
                            args.mlperf_logging
                            and (args.mlperf_auc_threshold > 0)
                            and (best_auc_test > args.mlperf_auc_threshold)
                        ):
                            print(
                                "MLPerf testing auc threshold "
                                + str(args.mlperf_auc_threshold)
                                + " reached, stop training"
                            )
                            if args.mlperf_logging:
                                mlperf_logger.barrier()
                                mlperf_logger.log_end(
                                    key=mlperf_logger.constants.RUN_STOP,
                                    metadata={
                                        mlperf_logger.constants.STATUS: mlperf_logger.constants.SUCCESS
                                    },
                                )
                            break

                if args.mlperf_logging:
                    mlperf_logger.barrier()
                    mlperf_logger.log_end(
                        key=mlperf_logger.constants.EPOCH_STOP,
                        metadata={mlperf_logger.constants.EPOCH_NUM: (k + 1)},
                    )
                    mlperf_logger.barrier()
                    mlperf_logger.log_end(
                        key=mlperf_logger.constants.BLOCK_STOP,
                        metadata={mlperf_logger.constants.FIRST_EPOCH_NUM: (k + 1)},
                    )
                k += 1  # nepochs
            if args.mlperf_logging and best_auc_test <= args.mlperf_auc_threshold:
                mlperf_logger.barrier()
                mlperf_logger.log_end(
                    key=mlperf_logger.constants.RUN_STOP,
                    metadata={
                        mlperf_logger.constants.STATUS: mlperf_logger.constants.ABORTED
                    },
                )
        else:
            print("Testing for inference only")
            inference(
                args,
                dlrm,
                best_acc_test,
                best_auc_test,
                test_ld,
                device,
                use_gpu,
            )

    # profiling
    if args.enable_profiling:
        time_stamp = str(datetime.datetime.now()).replace(" ", "_")
        with open("dlrm_s_pytorch" + time_stamp + "_shape.prof", "w") as prof_f:
            prof_f.write(
                prof.key_averages(group_by_input_shape=True).table(
                    sort_by="self_cpu_time_total"
                )
            )
        with open("dlrm_s_pytorch" + time_stamp + "_total.prof", "w") as prof_f:
            profile_data = prof.key_averages().table(sort_by="self_cpu_time_total")
            prof_f.write(profile_data)
            calculate_embeding_percentage(profile_data)
        prof.export_chrome_trace("dlrm_s_pytorch" + time_stamp + ".json")
        # print(prof.key_averages().table(sort_by="cpu_time_total"))

    # plot compute graph
    if args.plot_compute_graph:
        sys.exit(
            "ERROR: Please install pytorchviz package in order to use the"
            + " visualization. Then, uncomment its import above as well as"
            + " three lines below and run the code again."
        )
        # V = Z.mean() if args.inference_only else E
        # dot = make_dot(V, params=dict(dlrm.named_parameters()))
        # dot.render('dlrm_s_pytorch_graph') # write .pdf file

    # test prints
    if not args.inference_only and args.debug_mode:
        print("updated parameters (weights and bias):")
        for param in dlrm.parameters():
            print(param.detach().cpu().numpy())

    # export the model in onnx
    if args.save_onnx:
        """
        # workaround 1: tensor -> list
        if torch.is_tensor(lS_i_onnx):
            lS_i_onnx = [lS_i_onnx[j] for j in range(len(lS_i_onnx))]
        # workaound 2: list -> tensor
        lS_i_onnx = torch.stack(lS_i_onnx)
        """
        # debug prints
        # print("inputs", X_onnx, lS_o_onnx, lS_i_onnx)
        # print("output", dlrm_wrap(X_onnx, lS_o_onnx, lS_i_onnx, use_gpu, device))
        dlrm_pytorch_onnx_file = "dlrm_s_pytorch.onnx"
        batch_size = X_onnx.shape[0]
        print("X_onnx.shape", X_onnx.shape)
        if torch.is_tensor(lS_o_onnx):
            print("lS_o_onnx.shape", lS_o_onnx.shape)
        else:
            for oo in lS_o_onnx:
                print("oo.shape", oo.shape)
        if torch.is_tensor(lS_i_onnx):
            print("lS_i_onnx.shape", lS_i_onnx.shape)
        else:
            for ii in lS_i_onnx:
                print("ii.shape", ii.shape)

        # name inputs and outputs
        o_inputs = (
            ["offsets"]
            if torch.is_tensor(lS_o_onnx)
            else ["offsets_" + str(i) for i in range(len(lS_o_onnx))]
        )
        i_inputs = (
            ["indices"]
            if torch.is_tensor(lS_i_onnx)
            else ["indices_" + str(i) for i in range(len(lS_i_onnx))]
        )
        all_inputs = ["dense_x"] + o_inputs + i_inputs
        # debug prints
        print("inputs", all_inputs)

        # create dynamic_axis dictionaries
        do_inputs = (
            [{"offsets": {1: "batch_size"}}]
            if torch.is_tensor(lS_o_onnx)
            else [
                {"offsets_" + str(i): {0: "batch_size"}} for i in range(len(lS_o_onnx))
            ]
        )
        di_inputs = (
            [{"indices": {1: "batch_size"}}]
            if torch.is_tensor(lS_i_onnx)
            else [
                {"indices_" + str(i): {0: "batch_size"}} for i in range(len(lS_i_onnx))
            ]
        )
        dynamic_axes = {"dense_x": {0: "batch_size"}, "pred": {0: "batch_size"}}
        for do in do_inputs:
            dynamic_axes.update(do)
        for di in di_inputs:
            dynamic_axes.update(di)
        # debug prints
        print(dynamic_axes)
        # export model
        torch.onnx.export(
            dlrm,
            (X_onnx, lS_o_onnx, lS_i_onnx),
            dlrm_pytorch_onnx_file,
            verbose=True,
            opset_version=11,
            input_names=all_inputs,
            output_names=["pred"],
            dynamic_axes=dynamic_axes,
        )
        # recover the model back
        dlrm_pytorch_onnx = onnx.load("dlrm_s_pytorch.onnx")
        # check the onnx model
        onnx.checker.check_model(dlrm_pytorch_onnx)
    total_time_end = time_wrap(use_gpu)
    # After inference, print hot/cold timing breakdown
    if hasattr(dlrm, 'remap_count') and dlrm.remap_count > 0:
        print(f"\n{'='*80}")
        print(f"[HOT/COLD TIMING BREAKDOWN]")
        print(f"{'='*80}")
        print(f"  Hot/cold remap operations:     {dlrm.remap_count:>12,}")
        print(f"  Standard lookup operations:    {dlrm.standard_lookup_count:>12,}")
        print(f"  Total remap time:              {dlrm.remap_time:>12.2f}s")
        print(f"  Total lookup time (hot+cold):  {dlrm.lookup_time:>12.2f}s")
        print(f"  Total standard lookup time:    {dlrm.standard_lookup_time:>12.2f}s")
        print(f"  Avg remap per operation:       {dlrm.remap_time/dlrm.remap_count*1000:>12.3f}ms")
        print(f"  Avg lookup per operation:      {dlrm.lookup_time/dlrm.remap_count*1000:>12.3f}ms")
        print(f"  Remap overhead %:              {dlrm.remap_time/(dlrm.remap_time+dlrm.lookup_time)*100:>12.1f}%")
        print(f"{'='*80}\n")
    elif hasattr(dlrm, 'standard_lookup_count') and dlrm.standard_lookup_count > 0:
        print(f"\n{'='*80}")
        print(f"[BASELINE TIMING]")
        print(f"{'='*80}")
        print(f"  Standard lookup operations:    {dlrm.standard_lookup_count:>12,}")
        print(f"  Total standard lookup time:    {dlrm.standard_lookup_time:>12.2f}s")
        print(f"  Avg lookup per operation:      {dlrm.standard_lookup_time/dlrm.standard_lookup_count*1000:>12.3f}ms")
        print(f"{'='*80}\n")
    print(f"The embedding time is {dlrm.time_look_up}")
    print(f"The interaction time is {dlrm.time_interact}")
    print(f"The total time is {total_time_end-start_time}")
    print("Command used to run the program: " + " ".join(sys.argv))



if __name__ == "__main__":
    run()
