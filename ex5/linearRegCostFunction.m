function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% sizeX = size(X)
% sizeY = size(y)
% sizeTheta = size(theta)
% sizeLambda = size(lambda)

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% another way to vertorization
% J = sum((X*theta - y).^2) / (2*m);
J = (X*theta - y)' * (X*theta - y) / (2*m);

% add regularation
thetaR = theta([2:end],:);
J += sum(thetaR.^2) * lambda / (2*m);

% gradient
grad = (X' * (X*theta - y)) / m + [0; lambda * thetaR / m];













% =========================================================================

grad = grad(:);

end
