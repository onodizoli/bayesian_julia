using DelimitedFiles, LinearAlgebra, StatsBase, Statistics, Images, Colors, Plots, Random, Distributions

X_train = readdlm("C:\\Users\\zoli\\Desktop\\school\\Machine Learning\\Homeworks\\4\\data\\ratings.txt", ',', Int64)
X_test = readdlm("C:\\Users\\zoli\\Desktop\\school\\Machine Learning\\Homeworks\\4\\data\\ratings_test.txt", ',', Int64)
label = readdlm("C:\\Users\\zoli\\Desktop\\school\\Machine Learning\\Homeworks\\4\\data\\movies.txt", '\n')

missing_movies = [1325, 1414, 1577, 1604, 1637, 1681]


# Parameters
d = 10 # dimension of low rank matrix
sigma = 0.25
lamb = 10 # lambda
Sig = Diagonal(ones(d))/lamb # Covariance of V and U
Random.seed!(1234)

N1 = maximum(X_train[:,1])
N2 = size(label)[1]

T0 = 10 # burn in period
Tn = 20 # sampling period
n = 20 # sample size

U = rand(Float64, N1, d)
V = rand(Float64, d, N2)

sigma_U = [Array{Float64}(undef, d,d) for i in 1:N1]
mu_U =  [Array{Float64}(undef, d) for i in 1:N1]

score_matrix = Array{Int64}(undef, N1, N2)

usr_to_mves = [Array{Int64}([]) for i in 1:N1]
mve_to_usrs = [Array{Int64}([]) for i in 1:N2]



for i in 1:size(X_train)[1]
    score_matrix[X_train[i,1], X_train[i,2]] = X_train[i,3]
    push!(usr_to_mves[X_train[i,1]], X_train[i,2])
    push!(mve_to_usrs[X_train[i,2]], X_train[i,1])
end

for i in missing_movies
    score_matrix[N1, i] = 1
    push!(usr_to_mves[N1], i)
    push!(mve_to_usrs[i], N1)
end

trained_matricies = Array{Float64}(undef, 20, N1, N2)

for t in 1:(T0+Tn*(n-1))
    vv_t = [V[:,j]*V[:,j]' for j in 1:N2]

    for i in 1:N1
        sigma_Ui = inv(Sig + sum(vv_t[usr_to_mves[i]][:,:]).*sigma)
        mu_Ui = sigma_Ui*((V[:,usr_to_mves[i]] * score_matrix[i, usr_to_mves[i]]).*sigma)
        #println(typeof(sigma_Ui))
        sigma_Ui = convert(Array{Float64,2}, Hermitian(sigma_Ui))
        d = MvNormal(mu_Ui, sigma_Ui)
        U[i,:] = rand(d, 1)
    end

    uu_t = [U[i,:]*U[i,:]' for i in 1:N1]

    for j in 1:N2
        sigma_Vj = inv(Sig + sum(uu_t[mve_to_usrs[j]][:,:]).*sigma)
        mu_Vj = sigma_Vj*((U[mve_to_usrs[j],:]' * score_matrix[mve_to_usrs[j], j]).*sigma)
        sigma_Vj = convert(Array{Float64,2}, Hermitian(sigma_Vj))
        d = MvNormal(mu_Vj, sigma_Vj)
        V[:,j] = rand(d, 1)
    end

    if (t>=T0) & ((t-T0)%Tn==0)
        println(t)
        trained_matricies[convert(Int, (t-T0)/Tn+1), :,:] = U*V
    end
    #sigma_U = [inv(Sig + sum(vv_t[usr_to_mve[i]][:,:]).*sigma) for i in 1:N1]
    #mu_U = [sigma_U[i]*(V[:,usr_to_mve[i]] * score_matrix[i, usr_to_mve[i]]).*sigma for i in 1:N1]

    #d = [MvNormal(mu_U[i][:], sigma_U[i][:,:]) for i in 1:N1]
    #println(vv_t[1][:,:])

    #println(mu_U[1][:])
end

final_matrix = reshape(mean(trained_matricies, dims=1), N1, N2)

final_matrix = map(x ->max(x,1), final_matrix)
final_matrix = map(x ->min(x,5), final_matrix)

predictions = [final_matrix[X_test[i,1], X_test[i,2]] for i in 1:size(X_test)[1]]

rmse = sqrt(sum((predictions-X_test[:,3]).^2)/size(X_test)[1])
