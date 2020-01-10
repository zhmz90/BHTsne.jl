module BHTsne

using Statistics
using LinearAlgebra
using PyCall
pystruct = pyimport("struct")

const BH_TSNE_path = joinpath(dirname(@__FILE__),"cpp","bh_tsne")

export bh_tsne


function __init__()
    if Sys.iswindows()
        error("Sorry, BHTsne.jl doesn't support Windows now")
    end
    if !isfile(BH_TSNE_path) 
        cd(dirname(BH_TSNE_path)) do
            run(`g++ sptree.cpp tsne.cpp -o bh_tsne -O2`)
        end
    end
end


function read_unpack(fmt,f)
    pystruct.unpack(fmt,read(f, pystruct.calcsize(fmt)))
end


function bh_tsne(samples;no_dims=2, initial_dims=50, perplexity=50,
                 theta=0.5, randseed=-1, verbose=true)
   
    samples = broadcast(-, samples, mean(samples,dims=1))
    cov_x = samples' * samples
    eig_val,eig_vec = eigen(cov_x)
   
    eig_vec = eig_vec[:,sortperm(eig_val,rev=true)]

    num_eigvec = size(eig_vec, 2)
    if initial_dims > num_eigvec
        initial_dims = num_eigvec
    end

    eig_vec = eig_vec[:,1:initial_dims]
    samples = samples * eig_vec
   
    sample_count, sample_dim = size(samples)

    mktempdir() do temp_dir
   open(joinpath(temp_dir, "data.dat"),"w") do data_file
       write(data_file,pystruct.pack("iiddi", sample_count,sample_dim, theta, perplexity,no_dims))
       nrow,ncol = size(samples)
       fmt = repeat("d",ncol)
       for i = 1:nrow
           data_packed = mapreduce(x->pystruct.pack("d",x), *, samples[i,:])
           write(data_file, data_packed)
       end
       if randseed != -1
           write(data_file, pystruct.pack("i",randseed))
       end
   end

           
        cd(temp_dir) do
            if verbose
                try
                    run(pipeline(`$BH_TSNE_path`, stdout=stderr))
                catch excp
                    warn(excp)
                    error("ERROR: Call to bh_tsne exited with a non-zero return code exit status,
                          please refer to the bh_tsne output for further details")
                end
            else
                try
                    run(pipeline(`$BH_TSNE_path`, stdout=devnull))
                catch excp
                    warn(excp)
                    error("ERROR: Call to bh_tsne exited with a non-zero return code exit status,
                          please enable verbose mode and refer to the bh_tsne output for further details")
                end
            end
        end    
        open(joinpath(temp_dir, "result.dat"),"r") do output_file
            result_samples, result_dims = read_unpack("ii", output_file)
            results = [read_unpack(repeat("d", result_dims), output_file) for _ in 1:result_samples]
            results = [(read_unpack("i", output_file),e) for e in results]
            sort!(results)
            ret = Array{Float64,2}(undef,result_samples,result_dims)
            for i in 1:result_samples
                ret[i,:] = collect(results[i][2])
            end
            return ret
        end
    end
       
end
#=
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "-d","--no_dims"
            arg_type = Int64
        "-p","--perplexity"
            arg_type = Float64
            default = 50
        "-t","--theta"
            arg_type = Float64
            default = 0.5
        "-r","--randseed"
            arg_type = Int64
            default = -1 
        "-n","--initial_dims"
            arg_type = Int64
            default = 50
        "-v","--verbose"
            action = :store_true
        "-i","--input"
            arg_type = ASCIIString
            default = STDIN
        "-o","--output"
            arg_type = ASCIIString
            default = STDOUT
    end
    
    parse_args(s)
end
=#

end
