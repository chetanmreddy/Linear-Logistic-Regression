function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

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

h=X*theta;
error=h-y;
err_sqr=error.^2;
s=sum(err_sqr);
i=(1/(2*m))*s;

theta(1)=0;
r=sum(theta.^2);
rr=(lambda/(2*m))*r;
J=i+rr;

p=(X'*error);
g=p./m;
w=(lambda/m)*theta;
grad=g+w;

% =========================================================================

grad = grad(:);

end
