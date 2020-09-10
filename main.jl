include("data_loaders.jl")
using Flux: onecold
using CuArrays, Flux
using Statistics
using Base.Iterators: repeated

wd = "C:\\Users\\lukas\\Downloads\\9232_29380_bundle_archive\\dataset2-master\\dataset2-master\\images"
n = 1.0
b_size = 140

X,Y= load_data2(wd,n,b_size,"TRAIN")
X_t,Y_t= load_data2(wd,n,b_size,"TEST")
n_batch = size(X)[1]
m = Chain(
    Conv((2,2), 3=>16 ,pad = (1,1),relu),
    MaxPool((2,2)),
    Conv((2,2), 16=>8 ,pad = (1,1),relu),
    MaxPool((2,2)),
    Conv((2,2), 8=>4 ,pad = (1,1),relu),
    MaxPool((2,2)),
    Conv((2,2), 4=>4 ,pad = (1,1),relu),
    MaxPool((2,2)),
    x -> flatten(x),
    Dense(320,32),
    x -> relu.(x),
    Dense(32,16),
    x -> relu.(x),
    Dense(16,8),
    x -> relu.(x),
    Dense(8,4),
    softmax)

loss(x,y) = Flux.mse(m(x),y)
opt = ADAM()
accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))
function test(X,Y)
    n,out = size(X)[1], 0.0
    for i=1:n out += accuracy(X[i],Y[i]) end
    floor((out/n)*1e3)/1e3
end
function ltest(X,Y)
    n,out = size(X)[1], 0.0
    for i=1:n out += loss(X[i],Y[i]) end
    floor((out/n)*1e3)/1e3
end

@time println("Testing accuracy: ",test(X_t,Y_t))
m = m |> gpu
n_epochs = 30
println("Training Started")
@time for i=1:n_epochs
    println("Epoch nr: $i")
    @time for j=1:n_batch
        a, b = CuArray(X[j]), CuArray(Y[j])
        Flux.train!(loss, params(m), repeated((a,b),1), opt)
        α, β, df = nothing, nothing, nothing
        a, b = nothing, nothing
        GC.gc()
    end
end
println("Training ended")
m = m |> cpu
@time println("Testing accuracy: ",test(X_t,Y_t))
@time println("Trening accuracy: ",test(X,Y))
