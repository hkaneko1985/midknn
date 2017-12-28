function midknn_index = midknn( X , k )
% calculate indexes of the midpoints between k-nearest-neighbor data points
% of a training dataset (midknn) as a validation dataset in regression
%
% --- input ---
% X : dataset of X-variables
% k : number of nearest neighbors (k in k-nearest-neighbor)
%
% --- output ---
% midknn_index : indexes of midpoints between k-nearest-neighbor data points
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

allsamplenumber = size(X,1);
samplepairnumbers = zeros(allsamplenumber,k);

for samplenumber = 1 : allsamplenumber
    distance = sqrt( sum( (X - repmat( X(samplenumber,:), allsamplenumber , 1 )) .^ 2 , 2 ) );
    [ ~, distanceorder ] = sort( distance );
    samplepairnumbers( samplenumber, : ) = distanceorder(2:k+1)';
end

midknn_index = zeros( allsamplenumber*k, 2 );
for nearestsamplenumber = 1 : k
    midknn_index( (nearestsamplenumber-1)*allsamplenumber+1:nearestsamplenumber*allsamplenumber, 1 ) = (1:allsamplenumber)';
    midknn_index( (nearestsamplenumber-1)*allsamplenumber+1:nearestsamplenumber*allsamplenumber, 2 ) = samplepairnumbers( :, nearestsamplenumber );
end
