
# Multi-Agent Transformer (MAT)

Installation:
```bash
mamba create -yn mat python="3.12"
mamba activate mat
pip install -r requirements.txt
```

### TODO

- [ ] Benchmark
- [ ] Kolmogorov complexity comparator badge to compare with OG impl
- [ ] seqlen > nagents 
  - [ ] short action history useful?
  - [ ] hidden agents (STM context / registers)
- [ ] GATO (multi-modal, multi-task)
- [ ] Streaming Deep RL

### Appreciation

- [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/); a great resource for many big and small details in the implementation of PPO.
