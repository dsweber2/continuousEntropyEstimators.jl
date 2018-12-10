module continuousEntropyEstimators

using Distances, NearestNeighbors, SpecialFunctions
# using Statistics
# TODO: implement version in 1D, possibly 2D that doesn't rely on trees
# TODO: implement version that runs in parallel
export entropyEstimator, kraskovEstimator, NNestimator, entropy, mutualInformation, conditionalMutualInformation

# because I like math symbols
Γ(x) = gamma(x)
abstract type entropyEstimator end
mutable struct kraskovEstimator<: entropyEstimator end
mutable struct NNestimator<: entropyEstimator end

"""
    entropy(data::Array{T,2},  estimator::kraskovEstimator=kraskovEstimator(); k::Int64=3, logbase::S=2, treeType=Chebyshev()) where {T<:AbstractFloat, S<:Real}
    entropy(data::Array{Int64}, logbase::Float64=2.0)

computes the entropy of the data, which should be a d×N array.

If data is an array of floats, it is treated as a differential entropy calculation, by default done using knn's with k=3. treeType should be a distance from Distances. Some options include

| Distance     | As a p norm |
|--------------|:-----------:|
| Chebyshev()  | ``∞``       |
| Euclidean()  | ``2``       |
| Cityblock()  | ``1``       |
| Minkowski(p) | ``p``       |

If data is an array of ints, it uses the plug-in estimator.

TODO: Calculate the entropy in the method of Gao et al for mixed distributions
"""
function entropy(data::Array{T,2},  estimator::kraskovEstimator=kraskovEstimator(); k::Int64=3, logbase::S=2, treeType=Chebyshev()) where {T<:AbstractFloat, S<:Real}
    d, N = size(data)
    # perturb all the points to avoid problems with deterministic functions
    X = copy(data) + 1e-10*randn(size(data))
    tree = KDTree(X, treeType)
    if treeType==Chebyshev()
        cb = d*log(logbase, 2)
    elseif treeType==Euclidean()
        p=2
    elseif treeType==Cityblock()
        p=1
    elseif typeof(treeType)==Minkowski{Int64} || typeof(treeType)==Minkowski{Float64}
        p = treeType.p
    end
    if treeType!=Chebyshev()
        print(treeType)
        cb = d*log(logbase,2) + d*log(logbase, Γ(1+1/p))- log(logbase, Γ(1+d/p))
    end
    entropyEstimate = digamma(N)-digamma(k) + cb + d*log(logbase, 2) # this last factor is left implicit in eq 20
    for i=1:N
        _, dist = knn(tree, X[:,i], k+1, true)
        # check to see if this point is discrete, in which case Gao et al suggest using a plug-in estimator
        # if dist[end]<1e-10
        entropyEstimate += d/N*log(logbase, dist[end])
    end
    return entropyEstimate
end
continuousEntropyEstimators.entropy
# function entropy(data::Array{Float64,2}, estimator::NNestimator=NNestimator(); logbase=2, treeType=Chebyshev())
#     d, N = size(data)
#     # perturb all the points to avoid problems with deterministic functions
#     X = copy(data) + 1e-10*randn(size(data))
#     tree = KDTree(X, treeType)
#     entropyEstimate = γ/log(logbase) + log(logbase, 2)
#     for i=1:N
#         _, dist = knn(tree, X[:,i], 2, true)
#         entropyEstimate += 1/N*log(logbase, N*dist[end])
#     end
#     return entropyEstimate
# end
# unique([1.5, 1.5, 3, 201])
# data = rand([1,3,2], 10)
# uniqueVal = unique(data)
# data[data.==uniqueVal[1]]
# maximum(data.==uniqueVal[1])
function entropy(data::Array{Int64}, logbase::Float64=2.0)
    pEstimate, _ = pluginEstimator(data)
    pEstimate[pEstimate.==0] = 1
    return -pEstimate'*log.(logbase, pEstimate)
end

"""
    pluginEstimator(data::Array{Int64,(1,2)})
estimate the probability distribution of a discrete variable by using the counts
"""
function pluginEstimator(data::Array{Int64,2})
    Y = [data[:,i] for i=1:size(data, 2)]
    uniqueVal = unique(Y)
    counts = zeros(size(uniqueVal))
    for i = 1:size(counts,1)
        for j = 1:length(Y)
            counts[i] += Y[j]==uniqueVal[i]
        end
    end
    counts = counts/length(data)
    return (counts, uniqueVal)
end

function pluginEstimator(data::Array{Int64,1})
    uniqueVal = unique(data)
    counts = zeros(size(uniqueVal))
    for i = 1:size(counts,1)
        counts[i] = sum(data.==uniqueVal[i])
    end
    counts = counts/length(data)
    return (counts, uniqueVal)
end
# abs(entropy(rand([1,3], 1000000))-1)<1e-5 # uniform variable on two values has unit entropy base 2
"""
    mutualInformation(Xc::Array{Float64,2}, Y::Array{Float64,2}, k=3; logbase=2, treeType=Chebyshev(), version::Int64=2)
    mutualInformation(Xc::Array{Float64,2}, Yc::Array{Int64}; k=3, logbase=2, treeType=Chebyshev())
    mutualInformation(Xc::Array{Int64,2}, Yc::Array{Float64}; k=3, logbase=2, treeType=Chebyshev())

Calculate the mutual information between Xc and Yc using knn nearest neighbor-type methods.

make sure that the data is `d`×`N`, where `d` is the dimension and `N` is the number of examples. `Xc` and `Yc` should match in the second dimension, of course. `version` corresponds to the versions in the Kraskov et. al paper, and is either 1 or 2. Default is chosen for high dimension. It uses neighborhoods restricted to `Xc` (resp `Yc`) to calculate the contribution from that direction. In the case that one of the variables is discrete and want it treated as such, make sure it is integer. It is calculated using ``I(X;Y) = H(X)-H(X|Y)`` If both are discrete, it just uses ``H(X) + H(Y) - H(X,Y)``
NOTE: only Chebyshev is implemented at this time
"""
function mutualInformation(Xc::Array{Float64,2}, Yc::Array{Float64,2}; k=3, logbase=2, treeType=Chebyshev(), version::Int64=2)
    @assert size(Xc,2)==size(Yc,2)
    @assert treeType==Chebyshev()
    dx, N = size(Xc)
    dy, N = size(Yc)
    X = copy(Xc) + 1e-10*randn(dx, N)
    Y = copy(Yc) + 1e-10*randn(dy, N)
    XY = [X; Y]
    tree = KDTree(XY, treeType)
    entropyEstimate = digamma(k) + digamma(N)
    if version==2
        entropyEstimate-=1/k
    end
    xTree = KDTree(X, treeType)
    yTree = KDTree(Y, treeType)
    for i=1:N
        neighborhood, dist = knn(tree, XY[:,i], k+1, true)
        neighbor = neighborhood[end]
        dist = dist[end]
        if version==1
            # count the points within x(y) distance of ε/2
            nx = length(inrange(xTree, X[:,i], dist, false)) # note that we both need to add one for the algorithm and we "accidentally" overcount by one, so there's no change
            ny = length(inrange(yTree, Y[:,i], dist, false)) # same w/y
        elseif version==2
            εx = chebyshev(X[:, neighbor],X[:,i]) # \varepsilon
            nx = length(inrange(xTree, X[:,i], εx, false)) # note that we both need to add one for the algorithm and we "accidentally" overcount by one, so there's no change
            εy = chebyshev(Y[:, neighbor],Y[:,i])
            ny = length(inrange(yTree, Y[:,i], dist, false)) # note that we both need to add one for the algorithm and we "accidentally" overcount by one, so there's no change
        else
            error("``(version)'s not an allowed value of version. Try 1 or 2")
        end
        entropyEstimate -= 1/N*digamma(nx)
    end
    return entropyEstimate
end


function mutualInformation(Xc::Array{Float64,2}, Yc::Array{Int64}; k=3, logbase=2, treeType=Chebyshev())
    @assert size(Xc,2)==size(Yc,2)
    @assert treeType==Chebyshev()
    dx, N = size(Xc)
    dy, N = size(Yc)
    X = copy(Xc) + 1e-10*randn(dx, N)
    Y = copy(Yc)
    # if Xis pure continuous, Y discrete, it's best to use I(X,Y) = H(Y)-H(Y|X)
    # First order of business is getting a plug-in estimate for Y
    pEstimate, yVals = pluginEstimator(Y)
    Nxgy =pEstimate*N # the number of Xs given the value of y in index
    eeXgY = digamma(k) + digamma.(Nxgy) # a vector of entropy estimates given the value of y
    entropyEstimate = entropy(X,k=k, logbase=logbase, treeType=treeType)
    #
    if dy>1
        Yp = [data[:,i] for i=1:size(data, 2)]
        for i = 1:length(pEstimate)
            for j = 1:length(Yp)
                tmp = pEstimate[i]*entropy(X[:, Yp[j]==yVals[i]])
                entropyEstimate-=pEstimate[i]*entropy(X[:, Yp[j]==yVals[i]])
            end
        end
    else
        # Y is one dimensional
        for i = 1:length(pEstimate)
            entropyEstimate-=pEstimate[i]*entropy(X[:, reshape(Y.==yVals[i],(:))], k=k, logbase=logbase, treeType=treeType)
        end
    end
    return abs(entropyEstimate)
end

function mutualInformation(Xc::Array{Int64}, Yc::Array{Int64}, logbase::Float64=2.0)
    entropy(Xc,logbase=logbase) + entropy(Yc,logbase=logbase) - entropy([Xc;Yc],logbase=logbase)
end








# Various kinds of conditional mutual information
"""
    conditionalMutualInformation(Xc::Array{X},Yc::Array{Y}, Zc::Array{Z}, k::Int64=3, logbase::Float64=2.0, treeType=Chebyshev(), version::Int64=2)) where {X<:Real, Y<:Real, Z<:Real}

Calculate the conditional mutual information between `Xc`, `Yc` given `Zc`, ``I(X;Y|Z) = I(X,Z; Y) - I(Z;Y)``.
"""
function conditionalMutualInformation(Xc::Array{Float64},Yc::Array{Float64}, Zc::Array{Float64}, k::Int64=3, logbase::Float64=2.0, treeType=Chebyshev(), version::Int64=2)
    mutualInformation([Xc; Zc], Yc,k=k,logbase=logbase, treeType=treeType, version=version) - mutualInformation(Zc, Yc, k=k, logbase=logbase, treeType=treeType, version=version)
end
function conditionalMutualInformation(Xc::Array{Float64},Yc::Array{Int64}, Zc::Array{Float64}, k::Int64=3, logbase::Float64=2.0, treeType=Chebyshev(), version::Int64=2)
    mutualInformation([Xc; Zc], Yc,k=k,logbase=logbase, treeType=treeType) - mutualInformation(Zc, Yc, k=k, logbase=logbase, treeType=treeType)
end
end # module



# entropy([rand([1,3,2,4,2],5000)'; rand([4,3,2,1,4],5000)'])
# data = [rand([1,3,2],500)'; rand([4,3,2],500)']
# X = [data[:,i] for i=1:size(data, 2)]
# uniqueVal = unique(X)
# uniqueVal[1]
# entropy([2.5*rand([1,3,2],500)'; 2.5*rand([4,3,2],500)'])
# d = 20
# X = randn(d,  1000)
# Y = X+.01*randn(d, 1000)
# @time mutualInformation(X, Y, version=2)
# @time entropy(X)
# @time entropy(Y)
# get_entropy(rand(1000))
# d = 3
# X = randn(d, 100000)
# @time entropy(X, kraskovEstimator(), k=3, treeType = Chebyshev())- 1/2*log2((e*2π)^d)
# @time entropy(X, NNestimator(), treeType = Chebyshev())
# 1/2*log2((e*2π)^d)
# using InformationMeasures
# 1.3.^range(-.001, -15, 5)
# 1.3.^range(3, 42, 5)
# 2.^range(4,14,5)
# 1/2*log(2,(2π)^2)
# Ns = [10, 50, 100, 500, 1000] #, 3000, 5000, 8000, 10000]
# esttimatesk8 = [zeros(x) for x in Ns]
# for (i,estimates) in enumerate(esttimatesk8)
#     print(i)
#     @time for j=1:length(estimates)
#         data = randn(10,Ns[i])
#         estimates[j] = entropy(data,8)
#     end
# end
# # simple binning
#
# using Cumulants
# c = cumulants(data')
# using InformationMeasures
# using Plots
# plot(Ns, [mean.(esttimates), mean.(esttimatesk8)])
# std.(esttimates)
# mean(estimates1)
# std(estimates1)
# entropyEstimate
# @assert N>=d
# data+=1e-10*randn(size(data))
#
#
# test = randn(10,10000)
# get_entropy(test)
# 1/2*log2((2π)^10)
# function sReLU(x)
#     if x>=0
#         return x*exp(-1/(1-x)^2)
#     else
#         return 0
#     end
# end
#
# using Plots
# t=-.5:.00001:10
# plot(t,[sReLU.(t), t])
