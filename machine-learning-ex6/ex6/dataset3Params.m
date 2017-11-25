function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% test = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
% m = length(test);
% results = zeros(m*m, 3);

% for i = 1:m
% 	for j = 1:m
% 		C_tmp = test(i);
% 		sigma_tmp = test(j);
% 		model= svmTrain(X, y, C_tmp, @(x1, x2) gaussianKernel(x1, x2, sigma_tmp)); 
% 		predictions = svmPredict(model, Xval);
% 		error = mean(double(predictions ~= yval));
% 		row = (i-1)*m + j;
% 		results(row,2) = C_tmp;
% 		results(row,3) = sigma_tmp;
% 		results(row,1) = error;
% 	endfor
% endfor

% [w,iw]= min(results);
% row_index = iw(1);

% C = results(row_index,2)
% sigma = results(row_index,3)

C =  1;
sigma =  0.10000;



% =========================================================================

end
