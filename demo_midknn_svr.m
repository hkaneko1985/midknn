clear; close all;
% demonstration of SVR hyperparameter optimization using the midpoints between
% k-nearest-neighbor data points of a training dataset (midknn)
% as a validation dataset in regression

% settings
k = 10;
svrcs = 2.^( -5:10 );
svrepsilons = 2.^( -15:0 );
svrgammas = 2.^( -20:10 );

% generate a sample dataset
samplenum = 300;
originalX = rand( samplenum, 3 );
originaly = originalX(:,1) + originalX(:,2).^3 + 0.05*randn(samplenum, 1);

% standarize X and y
[X, Xaverage, Xstd] = zscore( originalX );
[y, yaverage, ystd] = zscore( originaly );

% optimize gamma by maximizing variance in Gram matrix
varianceofgrammatrix = zeros( 1, length(svrgammas) );
distancematrix = sum((X.^2), 2) * ones(1, size(X,1)) + ones(size(X,1), 1) * sum(X.^2,2)' - 2.*(X*(X')); 
for svrgammanumber = 1:length(svrgammas)
    grammatrix = exp( -svrgammas(svrgammanumber) .* distancematrix );
    varianceofgrammatrix(svrgammanumber) = var( grammatrix(:) );
end
optimalsvrgamma = svrgammas( varianceofgrammatrix == max(varianceofgrammatrix)); optimalsvrgamma = optimalsvrgamma(1);

% optimize C and epsilon with midknn
midknn_index = midknn( X , k );
Xmidknn = ( X( midknn_index(:,1) , : ) + X( midknn_index(:,2) , : ) ) / 2;
originalymidknn = ( originaly( midknn_index(:,1) , : ) + originaly( midknn_index(:,2) , : ) ) / 2;
r2midknns = zeros( length(svrcs), length(svrepsilons) );
rmsemidknns = zeros( length(svrcs), length(svrepsilons) );
for svrcnumber = 1:length(svrcs)
    for svrepsilonnumber = 1:length(svrepsilons)
        svrmodel = fitrsvm( X, y, 'BoxConstraint', svrcs(svrcnumber), 'Epsilon', svrepsilons(svrepsilonnumber), 'KernelFunction', 'rbf', 'KernelScale', 1/sqrt(optimalsvrgamma));
        estimatedymidknn = predict( svrmodel, Xmidknn )*ystd + yaverage;
        r2midknns(svrcnumber,svrepsilonnumber) = 1 - sum( (originalymidknn-estimatedymidknn).^2 ) / sum( (originalymidknn-mean(originalymidknn)).^2 );
        rmsemidknns(svrcnumber,svrepsilonnumber) = sqrt( 2*(length(y)+1) * sum( (originalymidknn-estimatedymidknn).^2 ) / length(y) / (length(originalymidknn)-1) );
    end
end
[ optimalsvrcnumber, optimalsvrepsilonnumber] = find( r2midknns == max(max(r2midknns)) ); optimalsvrcnumber=optimalsvrcnumber(1); optimalsvrepsilonnumber=optimalsvrepsilonnumber(1);
optimalsvrc = svrcs(optimalsvrcnumber);
optimalsvrepsilon = svrepsilons(optimalsvrepsilonnumber);

disp( ['Optimal C: 2^' num2str(log2(optimalsvrc))] );
disp( ['Optimal epsilon: 2^' num2str(log2(optimalsvrepsilon))] );
disp( ['Optimal gamma: 2^' num2str(log2(optimalsvrgamma))] );
