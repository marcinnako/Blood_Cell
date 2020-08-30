include("data_loaders.jl")

wd = "C:\\Users\\lukas\\Downloads\\9232_29380_bundle_archive\\dataset2-master\\dataset2-master\\images"
n = 1.0
b_size = 50

X,Y= load_data(wd,n,b_size,"TRAIN")
X_t,Y_t= load_data(wd,n,b_size,"TEST")
