@time begin
    using Flux: onecold
    using CuArrays, Flux
    using Statistics
    using Base.Iterators: repeated
    using Plots
    include("data_loaders.jl"),include("summary_function.jl")
end

wd = "C:\\Users\\lukas\\Downloads\\9232_29380_bundle_archive\\dataset2-master\\dataset2-master\\images"
n = 1.0
b_size = 160 #160 Max size

@time X,Y= load_data(wd,n,b_size,"TRAIN")
@time X_t,Y_t= load_data(wd,n,b_size,"TEST")
n_batch = size(X)[1]
n_testbatch = size(X_t)[1]

m = Chain( #Neural Network
    Conv((2,2), 3=>16 ,pad = (1,1),relu),
    MaxPool((2,2)),
    BatchNorm(16,relu),
    Dropout(0.2),
    Conv((2,2), 16=>8 ,pad = (1,1),relu),
    MaxPool((2,2)),
    BatchNorm(8,relu),
    Dropout(0.2),
    Conv((2,2), 8=>4 ,pad = (1,1),relu),
    MaxPool((2,2)),
    BatchNorm(4,relu),
    Dropout(0.2),
    Conv((2,2), 4=>4 ,pad = (1,1),relu),
    MaxPool((2,2)),
    BatchNorm(4,relu),
    Dropout(0.2),
    x -> flatten(x),
    Dense(320,32),
    x -> relu.(x),
    Dense(32,16),
    x -> relu.(x),
    Dense(16,8),
    x -> relu.(x),
    Dense(8,4),
    softmax)
model = m
loss(x,y) = Flux.mse(m(x),y)
opt = ADAM()
accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))

function test(X,Y)# Test accuracy over batched set
    n,out = size(X)[1], 0.0
    for i=1:n out += accuracy(X[i],Y[i]) end
    floor((out/n)*1e3)/1e3
end
function ltest(X,Y) # Test loss val over batched set
    n,out = size(X)[1], 0.0
    for i=1:n out += loss(X[i],Y[i]) end
    floor((out/n)*1e3)/1e3
end

testmode!(m)
@time acc = [test(X_t,Y_t)]
println("Testing accuracy: ",acc[1])
testmode!(m,false)

m = m |> gpu
n_epochs = 100
max = 0.0
@info "Training Started for $n_epochs epochs"
@time for ep=1:n_epochs
    global m
    @info "Epoch nr: $ep"
    @time @simd for j=1:n_batch #@simd speed up loop swpaping elements
        α, β = CuArray(X[j]), CuArray(Y[j])
        Flux.train!(loss, params(m), repeated((α,β),1), opt)
    end
    if(ep%4==0)
        global max
        global model
        m = m |> cpu
        testmode!(m)
        δ = test(X_t,Y_t)
        append!(acc,δ)
        @info "Testing accuracy: " δ
        if(max<δ)
            max = δ
            model = m
        end
        testmode!(m,false)
        m = m |> gpu
    end
end
@info "Training ended"
m = m |> cpu
testmode!(m)

@time println("Testing accuracy: ",test(X_t,Y_t))
@time println("Trening accuracy: ",test(X,Y))
