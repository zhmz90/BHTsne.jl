using BHTsne
using Base.Test

samples = [1.0 0.0; 0.0 1.0]
result = bh_tsne(samples, no_dims=2, perplexity=0.1, randseed=2016)
@test_approx_eq result [-2070.3067044 6659.34094976; 2070.3067044 -6659.34094976]
