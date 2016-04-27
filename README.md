# BHTsne

This package is a Julia interface to [Van der Maaten's Barnes-Hut implementation of t-Distributed Stochastic Neighbor Embedding](https://github.com/lvdmaaten/bhtsne).
There are also other wrappers available in [Python](https://github.com/lvdmaaten/bhtsne), [Matlab](https://github.com/lvdmaaten/bhtsne), [R](https://github.com/jkrijthe/Rtsne) and [Torch](https://github.com/clementfarabet/manifold).

### Installation
	Pkg.clone(https://github.com/zhmz90/BHTsne.jl.git)
	
### Usage
```Julia
	using BHTsne
	using RDatasets	
	using Gadfly
	
	iris = dataset("datasets", "iris") 	
	samples = convert(Array, iris[:,1:4])
	labels  = convert(Array, iris[:,5])
	
	results = bh_tsne(samples,perplexity=30,verbose=true)
	
	p = plot(x=results[:,1],y=results[:,2],color=l)
	draw(PDF("tsne_of_iris.pdf",4inch,3inch), p)
```

### APIs
```Julia
	function bh_tsne(samples;no_dims=2, initial_dims=50, perplexity=50,theta=0.5, randseed=-1, verbose=false)
```
