import argparse
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
import torch
def get_args():
    parser = argparse.ArgumentParser(description="Model configuration arguments")

    parser.add_argument("--vocab_size", type=int, required=False, default=10000, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, required=True, help="Context window length")
    parser.add_argument("--d_model", type=int, required=True, help="Model hidden dimension size")
    parser.add_argument("--num_layers", type=int, required=True, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, required=True, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, required=True, help="Feedforward hidden dimension size")
    parser.add_argument("--rope_theta", type=float, required=False, default=10000, help="Rotary positional embedding theta")
    parser.add_argument("--warmup_steps", type=int, required=False, default=5)
    parser.add_argument("--benchmark_steps", type=int, required=False, default=10)
    parser.add_argument("--benchmark_options", type=str, required=False, default='forward', help='forward, backward, both')

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = get_args()
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).cuda()
    optim = AdamW(model.parameters())
    batch_size = 4
    x = torch.randint(low=0, high=args.vocab_size-1, size=(batch_size, args.context_length)).cuda()
    y =  torch.randint(low=0, high=args.vocab_size-1, size=(batch_size, args.context_length)).cuda()
    for _ in range(args.warmup_steps):
        optim.zero_grad()
        outputs = model(x)
        loss = cross_entropy(outputs, y)
        loss.backward()
        optim.step()
    from timeit import default_timer as timer
    durations = []
    
            
    for _ in range(args.benchmark_steps):
        if args.benchmark_options == 'both':
            start = timer()
            optim.zero_grad()
            outputs = model(x)
            loss = cross_entropy(outputs, y)
    
            loss.backward()
            torch.cuda.synchronize() 
            end = timer()
            durations.append(end-start)
        elif args.benchmark_options == 'forward':
            start = timer()
            optim.zero_grad()
            outputs = model(x)
            loss = cross_entropy(outputs, y)
            torch.cuda.synchronize() 
            end = timer()
            durations.append(end-start)
        elif args.benchmark_options == 'backward':
            optim.zero_grad()
            outputs = model(x)
            loss = cross_entropy(outputs, y)
            start = timer()
            torch.cuda.synchronize() 
            loss.backward()
            torch.cuda.synchronize() 
            end = timer()
            durations.append(end-start)

    import numpy as np
    arr = np.array(durations)
    
    mean = np.mean(arr)
    std = np.std(arr)
    
    print("Mean:", mean)
    print("Standard Deviation:", std)
            #optim.step()
    
    