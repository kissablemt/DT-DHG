import torch


def batched_dot_mul_sum(a, b):
    '''Computes batched dot by multiplying and summing'''
    return a.mul(b).sum(-1)


def batched_dot_bmm(a, b):
    '''Computes batched dot by reducing to bmm'''
    a = a.reshape(-1, 1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1], 1)
    return torch.bmm(a, b).flatten(-3)


import torch.utils.benchmark as benchmark

x = torch.randn(10000, 1024, device='cuda')

import pickle

ab_test_results = []
for env in ('environment A: mul/sum', 'environment B: bmm'):
    for b, n in ((1, 1), (1024, 10000), (10000, 1)):
        x = torch.ones((b, n))
        dot_fn = (batched_dot_mul_sum if env == 'environment A: mul/sum' else batched_dot_bmm)
        m = benchmark.Timer(
            stmt='batched_dot(x, x)',
            globals={'x': x, 'batched_dot': dot_fn},
            num_threads=1,
            label='Batched dot',
            description=f'[{b}, {n}]',
            env=env,
        ).blocked_autorange(min_run_time=1)
        ab_test_results.append(pickle.dumps(m))

ab_results = [pickle.loads(i) for i in ab_test_results]
compare = benchmark.Compare(ab_results)
compare.trim_significant_figures()
compare.colorize()
compare.print()

from torch.utils.benchmark.op_fuzzers import binary

results = []
for tensors, tensor_params, params in binary.BinaryOpFuzzer(seed=0).take(10):
    sub_label=f"{params['k0']:<6} x {params['k1']:<4} {'' if tensor_params['x']['is_contiguous'] else '(discontiguous)'}"
    results.append(benchmark.Timer(
        stmt='batched_dot_mul_sum(x, x)',
        setup='from __main__ import batched_dot_mul_sum',
        globals=tensors,
        label='Batched dot',
        sub_label=sub_label,
        description='mul/sum',
    ).blocked_autorange(min_run_time=1))
    results.append(benchmark.Timer(
        stmt='batched_dot_bmm(x, x)',
        setup='from __main__ import batched_dot_bmm',
        globals=tensors,
        label='Batched dot',
        sub_label=sub_label,
        description='bmm',
    ).blocked_autorange(min_run_time=1))

compare = benchmark.Compare(results)
compare.trim_significant_figures()
compare.colorize(rowwise=True)
compare.print()