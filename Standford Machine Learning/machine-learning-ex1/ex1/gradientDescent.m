function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


    % ALTERNATIVE SOLUTION
    % cost = (X * theta) - y;
    % theta_col_1 = theta(1) - ( (aplha / m) * sum(cost .* X[:,1]) );
    % theta_col_2 = theta(2) - ( (aplha / m) * sum(cost .* X[:,2]) );
    % theta = [theta_col_1, theta_col_2];


    % size cost: 97 1 --> X: 97 2 * theta: 2 1
    cost = (X * theta) - y;
    delta = (X' * cost);
    theta = theta - ( (alpha / m) * delta );
    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

end

end
