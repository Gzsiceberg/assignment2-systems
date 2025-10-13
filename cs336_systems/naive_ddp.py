from dataclasses import dataclass
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rich import print
from timeit import default_timer as tx

@dataclass
class DistributedConfig:
    world_size: int = 4
    backend: str = "gloo"

def setup(rank, config: DistributedConfig):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(config.backend, rank=rank, world_size=config.world_size)

def dist_main(rank, config: DistributedConfig):
    setup(rank, config)
    torch.manual_seed(0 + rank)
    device = torch.device(f'cuda:{rank}' if config.backend == 'nccl' else 'cpu')
    input_dim = 128
    output_dim = 2
    batch_size = 128
    model = torch.nn.Linear(input_dim, output_dim, bias=False).to(device)
    # weight = model.weight
    # mean_weight = weight.mean()
    # std_weight = weight.std()
    # print(f"Rank {rank}: before weight mean={mean_weight}, std={std_weight}")
    with torch.no_grad():
        for param in model.parameters():
            dist.broadcast(param, src=0)
    # mean_weight = model.weight.mean()
    # std_weight = model.weight.std()
    # print(f"Rank {rank}: after weight mean={mean_weight}, std={std_weight}")
    dist.barrier()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    base_model = torch.nn.Linear(input_dim, output_dim, bias=False).to(device)
    base_model.load_state_dict(model.state_dict())
    base_opt = torch.optim.AdamW(base_model.parameters(), lr=1e-2)
    assert torch.allclose(base_model.weight, model.weight)

    scatter_data = None
    scatter_labels = None
    X_batch = None
    Y_batch = None
    X = torch.zeros((batch_size // config.world_size, input_dim), device=device)
    Y = torch.zeros((batch_size // config.world_size, output_dim), device=device)
    if rank == 0:
        X_batch = torch.randn((batch_size, input_dim), device=device)
        Y_batch = torch.randn((batch_size, output_dim), device=device)
        scatter_data = [X_batch[i::config.world_size].contiguous() for i in range(config.world_size)]
        scatter_labels = [Y_batch[i::config.world_size].contiguous() for i in range(config.world_size)]
        assert all([x.shape == (batch_size // config.world_size, input_dim) for x in scatter_data])
        assert all([y.shape == (batch_size // config.world_size, output_dim) for y in scatter_labels])
        assert len(scatter_data) == config.world_size
        assert len(scatter_labels) == config.world_size
        # print(f"ScatterData: {scatter_data[0]}") --- IGNORE ---
    dist.scatter(X, scatter_data, src=0)
    dist.scatter(Y, scatter_labels, src=0)
    print(f"Rank {rank}: X.mean={X.mean():.2f}, X.std={X.std():.2f} shape={X.shape}")
    print(f"Rank {rank}: y.mean={Y.mean():.2f}, y.std={Y.std():.2f} shape={Y.shape}")
    if rank == 0:
        assert torch.allclose(scatter_labels[0], Y), "Scatter failed"

    base_loss = torch.tensor(0.0)
    for epoch in range(200):

        output = model(X)
        loss = torch.nn.functional.mse_loss(output, Y)
        loss.backward()
        if model.weight.grad is None:
            raise ValueError("Gradient is None")
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad.mul_(1.0 / config.world_size)
        if rank == 0:
            base_output = base_model(X_batch)
            base_loss = torch.nn.functional.mse_loss(base_output, Y_batch)
            base_loss.backward()
            for p0, p1 in zip(base_model.parameters(), model.parameters()):
                if p0.grad is None or p1.grad is None:
                    raise ValueError("Gradient is None")
                if not torch.allclose(p0.grad, p1.grad, atol=1e-5):
                    raise ValueError("Gradients are not close!")
            base_opt.step()
            base_opt.zero_grad()
        opt.step()
        opt.zero_grad()

        if rank == 0 and epoch % 10 == 0:
            print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item():.5f} base Loss: {base_loss.item():.5f}")
    
    dist.barrier()
    if rank == 0:
        print("Final model stats:")
    weight = model.weight
    mean_weight = weight.mean()
    std_weight = weight.std()
    print(f"Rank {rank}: weight mean={mean_weight}, std={std_weight}")
    if rank == 0:
        base_weight = base_model.weight
        mean_weight = base_weight.mean()
        std_weight = base_weight.std()
        print(f"Base model: weight mean={mean_weight}, std={std_weight}")
        assert torch.allclose(base_model.weight, model.weight, atol=1e-2), "Weights are not close!"
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    config = DistributedConfig()
    config.world_size = 2
    mp.spawn(dist_main, args=(config,), nprocs=config.world_size)


