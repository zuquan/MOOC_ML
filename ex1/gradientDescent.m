function [theta, J_history, theta_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_history = zeros(20, 2);
iter_t = 1;
% alpha = 0.003;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % temp1 = X*theta - y;
    % fprintf('size of temp1 %f\n', size(temp1));
    % size(temp1)

    % temp2 = (X*theta - y)';
    % fprintf('size of temp2 %f\n', size(temp2));
    % size(temp2)
    % fprintf('size of X %f\n', size(X));
    % size(X)

    % sizeofT = size(theta)

    theta = theta - ((X*theta - y)' * X)' * alpha / m; 

    % sizeofT = size(theta)

    % if iter<10
    %     fprintf('theta %f   and  cost %f\n', theta, computeCost(X,y,theta));
    % end
    


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);


    if iter_t<=20 && mod(iter, 5) == 1
        theta_history(iter_t,:) = theta;
        % fprintf('\niter %f theta is %f, %f\n', iter, theta);
        iter_t++;
    end


end

end
