using CSV, StatsBase, Statistics, DataFrames, Images, Colors, Plots

X_train = CSV.read("C:\\Users\\zoli\\ML\\hw1\\data\\Xtrain.csv", datarow=1)
X_test = CSV.read("C:\\Users\\zoli\\ML\\hw1\\data\\Xtest.csv", datarow=1)
y_train = CSV.read("C:\\Users\\zoli\\ML\\hw1\\data\\ytrain.csv", datarow=1, header=[:label])
y_test =CSV.read("C:\\Users\\zoli\\ML\\hw1\\data\\ytest.csv", datarow=1, header=[:label])
Q = convert(Array{Float64}, CSV.read("C:\\Users\\zoli\\ML\\hw1\\data\\Q.csv", datarow=1))
# prior parameters
k = 2
dim = 15
a = ones(k)
b = ones(k)
c = ones(k)
e = 1
f = 1

train = hcat(X_train, y_train)
N = by(train, :label, df -> DataFrame(N = size(df, 1)))
means = by(train, :label, df -> mean(convert(Array{Float64},df[1:end-1]), dims=1))

mu_0 = Array{Float64}(undef,k,dim)
lambda_0 = Array{Float64}(undef,k,1)
alpha_0 = Array{Float64}(undef,k,1)
beta_0 = Array{Float64}(undef,k,dim)
pi_0 =  pi_0 = [(f+N[:N][1])/(sum(N[:N])+e+f), (e+N[:N][2])/(sum(N[:N])+e+f)]
for i in 1:k
    data = convert(Array{Float64}, train[train[:label] .==i-1, 1:end-1])
    n_i =  N[N[:label] .==i-1, :N][1]
    mean_i = convert(Array{Float64}, means[means[:label] .==i-1, 2:end])
    lambda_0[i] = n_i+1/a[i]
    alpha_0[i] = b[i]+n_i/2
    mu_0[i, :] = n_i*mean_i/(lambda_0[i])
    beta_0[i, :] = (sum((data.-mean_i).^2, dims=1) + n_i/a[i]*(mean_i.^2)/lambda_0[i])/2 .+ c[i]
end

#predict
likelihood = Array{Float64}(undef, size(X_test, 1), k)
test = convert(Array{Float64}, X_test)
for i in 1:k
    likelihood[:,i] = (prod(((test.-mu_0[i, :]').^2)./(beta_0[i, :]*(1+lambda_0[i])./(lambda_0[i]))'.*(1/2) .+1, dims=2).^(-alpha_0[i]-0.5)) .*pi_0[i]
end
label = argmax(likelihood, dims=2)

# confusion matrix
confusion = zeros(2,2)
for i in 1:length(label)
    confusion[y_test[i,1]+1, label[i][2]] += 1
end

pic = reshape(Q*convert(Array{Float64}, X_test[450,:])', 28, 28)
picc = Gray.(pic')
