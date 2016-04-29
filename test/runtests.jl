src = "../src"
if !in(src,LOAD_PATH)
    push!(LOAD_PATH, src)
end
using BHtsne
using Base.Test

samples = [1.0 0.0; 0.0 1.0]
result = bh_tsne(samples, no_dims=2, perplexity=0.1)
@test result == [-2458.83181442 -6525.87718385; 2458.83181442 6525.87718385]
