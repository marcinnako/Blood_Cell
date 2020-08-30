# This file have all functions
# that load and batch data
using Images, Random


function load_data(path::String,perc::Float64,batch::Int64,type::String)
    lebs = ["EOSINOPHIL","LYMPHOCYTE","MONOCYTE","NEUTROPHIL"]
    im_size = (240,320)
    h::Int64 = 0
    for leb in lebs
        n = Int.(floor(size(readdir(path*"\\"*type*"\\"*leb))[1]*perc))
        h += n
    end
    n_batch::Int64 = Int.(floor(h/batch)) #Number of batches of size var(batch)
    n_add::Int64 = h - n_batch*batch  # Size of last batch
    X = [(Array{Float32}(undef, 240, 320,3,batch)) for i=1:n_batch]
    Y = [(Array{String}(undef,batch)) for i=1:n_batch]
    if(n_add!=0)
        push!(X,(Array{Float32}(undef, 240, 320,3,n_add)))
        push!(Y,(Array{String}(undef,n_add)))
    end
    rng = Random.shuffle(collect(1:h))
    i_rng = 1 # var that is used to assign random places in data
    @time for leb in lebs
        n = Int.(floor(size(readdir(path*"\\"*type*"\\"*leb))[1]*perc))
        names = readdir(path*"\\"*type*"\\"*leb)
        for filename in names[1:n]
            idx = rng[i_rng] - 1
            i_rng +=1
            img = Images.load(path*"\\"*type*"\\"*leb*"\\"*filename)
            n_batch = Int.(floor(idx/batch) + 1)
            id_batch = idx%batch + 1
            X[n_batch][:,:,1,id_batch] = Images.red.(img)
            X[n_batch][:,:,2,id_batch] = Images.green.(img)
            X[n_batch][:,:,3,id_batch] = Images.blue.(img)
            Y[n_batch][id_batch] = leb
        end
    end
    X,Y
end
