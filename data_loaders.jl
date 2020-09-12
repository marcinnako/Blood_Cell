# This file have all functions that load and batch data
using Images, Random
using Flux: onehotbatch

function load_data2(path::String,perc::Float64,batch::Int64,type::String)
    lebs = ["EOSINOPHIL","LYMPHOCYTE","MONOCYTE","NEUTROPHIL"]
    im_size = (240,320) #OG
    h::Int64 = 0
    for leb in lebs
        n = Int.(floor(size(readdir(path*"\\"*type*"\\"*leb))[1]*perc))
        h += n
    end
    n_batch::Int64 = fld(h,batch) # Number of batches of size var(batch)
    n_add::Int64 = h - n_batch*batch  # Size of last batch
    X = [(Array{Float32}(undef, 120, 160,3,batch)) for i=1:n_batch]
    Y = [(Array{String}(undef,batch)) for i=1:n_batch]
    if(n_add!=0) # Add last batch of smaller size e.g.: [5,5,5,3]
        push!(X,(Array{Float32}(undef, 120, 160,3,n_add)))
        push!(Y,(Array{String}(undef,n_add)))
    end
    rng = Random.shuffle(collect(1:h)) # series of random numbers [1:#OfPic]
    i_rng = 1 # var that is used to assign random places in data
    for leb in lebs
        # Number of images of given class
        n = Int.(floor(size(readdir(path*"\\"*type*"\\"*leb))[1]*perc))
        names = readdir(path*"\\"*type*"\\"*leb) # Names of all pic of one class
        @simd for filename in names[1:n]
            idx = rng[i_rng] # Random index
            i_rng +=1 # Get next random index
            img = Images.imresize(Images.load(path*"\\"*type*"\\"*leb*"\\"*filename),120,160)
            nu_batch = cld(idx,batch) # Id of batch the pic will be in
            id_batch = mod(idx-1,batch) + 1 # Position in batch
            X[nu_batch][:,:,1,id_batch] = Images.red.(img)
            X[nu_batch][:,:,2,id_batch] = Images.green.(img)
            X[nu_batch][:,:,3,id_batch] = Images.blue.(img)
            Y[nu_batch][id_batch] = leb
        end
    end
    # OnehotEncoding
    if(n_add!=0) # Check if there is a last batch with smaller size
        Y = [onehotbatch(Y[i],lebs) for i=1:n_batch+1]
    else
        Y = [onehotbatch(Y[i],lebs) for i=1:n_batch]
    end
    X,Y
end
