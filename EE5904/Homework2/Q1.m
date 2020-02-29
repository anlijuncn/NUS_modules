% Script for Q1 Homework 2
% wirtten by Lijun AN for EE5904

%% Parameter setting
eta = 0.001;
seed = 0;
MAX_iter = 100000;
epsilon = 1e-5;
xy = zeros(MAX_iter,2);
f = zeros(MAX_iter, 1);
save_dir = '/Users/ljan/Documents/MATLAB/EE5904/Part1/Homework2/Q1_outputs';
outstem = 'Newton_method_';
% use gradient descent or Hessian matrix
useHessian = true;
%% initialization
% initialize x and y 
x_init = 0.5434;
y_init = 0.2784;
f_init = rosenbrocks_valley(x_init, y_init);
% add x_init, x_init;
xy(1, 1) = x_init;
xy(1, 2) = y_init;
f(1) = f_init;
% train
[xy, f, iter] = train_rosen(xy, f, useHessian, eta, MAX_iter, epsilon);
% plot 
% xy_save_path = fullfile(save_dir, [outstem num2str(eta) '_xy.png']);
% f_save_path = fullfile(save_dir, [outstem num2str(eta) '_f.png']);
xy_save_path = fullfile(save_dir, [outstem 'xy.png']);
f_save_path = fullfile(save_dir, [outstem 'f.png']);

xy_xlabel = 'Input X';
xy_ylable = 'Input T';
% xy_title = ['(X, Y) Trajectory with Learning Rate ' num2str(eta)];
xy_title = ['(X, Y) Trajectory'];
f_xlabel = 'Iteration';
f_ylable = 'Function Value';
% f_title = ['Function Value Trajectory with Learning Rate ' num2str(eta)];
f_title = ['Function Value Trajectory'];
%% xy 
plot(xy(1:iter, 1), xy(1:iter, 2));
xlabel(xy_xlabel)
ylabel(xy_ylable)
% add figure title
title(xy_title)
% save figure
saveas(gcf, xy_save_path);
clf;
%% f
plot(f(1:iter));
xlabel(f_xlabel)
ylabel(f_ylable)
% add figure title
title(f_title)
% save figure
saveas(gcf, f_save_path);
clf;
%% training function
function [xy, f, iter] = train_rosen(xy, f, useHessian, eta, MAX_iter, epsilon)
    % train using gradient descent or Hessian matrix
    iter = 1;
    x = xy(iter, 1);
    y = xy(iter, 2);
    diff = 10000;
    while epsilon < diff
        % calculate Hessian and gradient matrix
        [g, H] = gradient_hessian(x, y);
        % update [x, y]
        if useHessian
            [new_x, new_y] = newton_method(x, y, g, H);
        else
            [new_x, new_y] = gradient_decesent(x, y, eta, g);
        end
        % replace x with new_x, y with new_y
        x = new_x;
        y = new_y;
        new_f = rosenbrocks_valley(new_x, new_y);
        diff = new_f;
        % record result
        xy(iter+1, 1) = new_x;
        xy(iter+1, 2) = new_y;
        f(iter) = new_f;
        % add iter
        iter = iter + 1;
    end
    iter = iter - 1;

end
%% Rosenbrock's Valley function
function f = rosenbrocks_valley(x, y)
    % Rosenbrock's Valley function implementation
    f = (1- x)*(1 - x) + 100 * (y - x*x)*(y - x*x);
end
%% Core code, calculate gradient and Hessian matrix
function [g, H] = gradient_hessian(x, y)
    % gradient and Hessian matrix calculation formula
    %  g = [400x^3-400xy+2x-2, 200y-200x^2]
    %  H = [[1200x^2-400y+2, -400x][-400x, 200]]
    % initialization
    g = zeros(2,1);
    H = zeros(2,2);
    % gradient vector
    g(1,1) = 400*power(x,3) - 400*x*y+ 2*x - 2;
    g(2,1) = 200*y - 200*power(x, 2);
    % Hessian matrix 
    H(1,1) = 1200*power(x,2) - 400*y + 2;
    H(1,2) = -400*x;
    H(2,1) = -400*x;
    H(2,2) = 200;
end
%% Part 1 core code, gradient decesent
function [new_x, new_y] = gradient_decesent(x, y, eta, g)
     % Gradinet decesent, w(k+1) = w(k) - eta + g(k)
     new_x = x - eta * g(1,1);
     new_y = y - eta * g(2,1);
end
%% Part 2 core code, newton method
function [new_x, new_y] = newton_method(x, y, g, H)
    % Newton's method, w(k+1) = w(k) - inv(H)*g
    delta = H\g;
    new_x = x - delta(1,1);
    new_y = y - delta(2,1);
end