%% EE5904 Part 1 Homemwork3 Q1
% Writen by An Lijun for EE5904
%% Dataset preparation and parameters
train_x = -1:0.05:1;
train_y = gen_y(train_x, true);
test_x = -1:0.01:1;
test_y = gen_y(test_x, false);
outdir = 'C:\Users\e0383065\Documents\MATLAB\EE5904\Part1\Homework3\Q1_outputs';
%% Run experiment Q1_1
[pred_y, tr_mse, te_mse] = RBFN(train_x, train_y, test_x, test_y);
q1_1_title = 'The approixmation performance of RBFN';
q1_1_save_path = fullfile(outdir, 'q1_1.png');
plot_test_performance(test_x, test_y, pred_y, q1_1_title, q1_1_save_path, 'RBFB Output');
%% Run experiment Q1_2
[pred_y, tr_mse, te_mse] = RBFN_FCSB(train_x, train_y, test_x, test_y, 15);
q1_1_title = 'The approixmation performance of RBFN-FCSB';
q1_1_save_path = fullfile(outdir, 'q1_2.png');
plot_test_performance(test_x, test_y, pred_y, q1_1_title, q1_1_save_path, 'RBFB-FCSB Output');
%% Run experiment Q1_3
lambda_set = [0, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 20, 50, 100];
for i = 1:length(lambda_set)
    lambda = lambda_set(i);
    [pred_y, tr_mse, te_mse] = RBFN_reg(train_x, train_y, test_x, test_y, lambda);
    disp(['Lambda: ' num2str(lambda) ' |Train MSE: ' num2str(tr_mse) ' |Test MSE: ' num2str(te_mse)]);
    title = ['The approixmation performance of RBFN-Reg (Lambda: ' num2str(lambda) ')'];
    save_path = fullfile(outdir, ['q1_3_lambda_' num2str(lambda) '.png']);
    plot_test_performance(test_x, test_y, pred_y, title, save_path, 'RBFB-Reg Output');
end
%% functions used in Q1
function [y] = gen_y(x, isTrain)
     % Generate y using x,
     % If train, adding Gaussian noise
     % If test, do not add Gaussion
     % create a zero vector to store y
     y = f(x);
     if isTrain 
         % Gaussian noise
         noise = 0.3*randn(size(y));
         y = y + noise;
     end
end

function [y] = f(x)
    % y_i = 1.2*sin(pi*x) - cos(2.4*pi*x_i)
    y = 1.2*sin(pi*x) - cos(2.4*pi*x);
end

function [pred_y, tr_mse, te_mse] = RBFN(train_x, train_y, test_x, test_y)
    % Fit a basic RBFN model:
    %    1.Using training data points as center
    %    2.Without regularization 
    % calculate distance and interpolation matrix
    tr_dist = distance_matrix(train_x', train_x);
    te_dist = distance_matrix(test_x', train_x);
    [tr_phi] = RBF_Gaussian(tr_dist, 0.01);
    [te_phi] = RBF_Gaussian(te_dist, 0.01);
    % get unique weight matrix w
    w = pinv(tr_phi)*train_y';
    % making predicton using te_phi
    pred_y = (te_phi * w)';
    % calculate train MSE and test MSE
    tr_mse = sum((train_y - (tr_phi*w)').^2)/size(train_x, 2);
    te_mse = sum((test_y - pred_y).^2)/size(test_x, 2);
end

function [pred_y, tr_mse, te_mse] = RBFN_FCSB(train_x, train_y, test_x, test_y, M)
    % Fit a RBFN model with Fixed Centers Selected at Random (RBFN_FCSB):
    %    1.Randomly pick M points from training set as centers
    %    2.Without regularization 
    % Radomly select M points as centers
    rand_index = randperm(size(train_x, 2), M);
    centers = train_x(rand_index);
    % calculate coefficients
    coef = (-(max(centers) - min(centers))^2)/ (2*length(centers));
    % calculate distance and interpolation matrix
    tr_dist = distance_matrix(train_x', centers);
    te_dist = distance_matrix(test_x', centers);
    [tr_phi] = RBF_Gaussian(tr_dist, coef);
    [te_phi] = RBF_Gaussian(te_dist, coef);
    % add bias to phi
    tr_phi = [ones(size(train_x, 2), 1), tr_phi];
    te_phi = [ones(size(test_x, 2), 1), te_phi];
    % get unique weight matrix w
    w = pinv(tr_phi)*train_y';
    % making predicton using te_phi
    pred_y = (te_phi * w)';
    % calculate train MSE and test MSE
    tr_mse = sum((train_y - (tr_phi*w)').^2)/size(train_x, 2);
    te_mse = sum((test_y - pred_y).^2)/size(test_x, 2);
end

function [pred_y, tr_mse, te_mse] = RBFN_reg(train_x, train_y, test_x, test_y, lambda)
    % Fit a RBFN model with Fixed Centers Selected at Random (RBFN_FCSB):
    %    1.Using training data points as center
    %    2.Adding regularization term lambda 
    % calculate distance and interpolation matrix
    tr_dist = distance_matrix(train_x', train_x);
    te_dist = distance_matrix(test_x', train_x);
    [tr_phi] = RBF_Gaussian(tr_dist, 0.01);
    [te_phi] = RBF_Gaussian(te_dist, 0.01);
    % add bias to phi
    tr_phi = [ones(size(train_x, 2), 1), tr_phi];
    te_phi = [ones(size(test_x, 2), 1), te_phi];
    % get unique weight matrix w
    w = pinv(tr_phi'*tr_phi + lambda*eye(size(train_x,2) + 1))*tr_phi'*train_y';
    % making predicton using te_phi
    pred_y = (te_phi * w)';
    % calculate train MSE and test MSE
    tr_mse = sum((train_y - (tr_phi*w)').^2)/size(train_x, 2);
    te_mse = sum((test_y - pred_y).^2)/size(test_x, 2);
end

function [dist] = distance_matrix(col_vector, row_vector)
    % Calculate distance matrix using col_vector and row_vector 
    % Here we are using Eulcidean distance
    % for example:
    % col_vector = [x_1, x_2, x_3]
    % row_vector = [x_1, x_2, x_3]
    % dist = [0, ||x_1 - x_2||, ||x_1 - x_3||;
    %         ||x_2 - x_1||, 0, ||x_2 - x_3||;
    %         ||x_3 - x_1||, ||x_3 - x_2||, 0]
    col_matrix = repmat(col_vector, 1, size(row_vector, 2));
    row_matrix = repmat(row_vector, size(col_vector, 1), 1);
    % element wise Eulclidean disrtance, since only 1 dim, same as L1
    dist = abs(col_matrix - row_matrix);

end

function [interpolation] = RBF_Gaussian(dist, std)
    % Gaussian RBF function
    interpolation = exp(power(dist, 2)*-1/(2*std));
end
%% Plot function to visualize model performance
function plot_test_performance(test_x, test_y, pred_y, fig_title, fig_save_path, model)
    % Visualizing model performance using test set by plotting test and
    % pred
    plot(test_x, test_y, 'b--', 'LineWidth', 1); hold on;
    plot(test_x, pred_y, 'r', 'LineWidth', 1);hold off;
    title(fig_title);
    xlabel('Test X');
    ylabel('Y');
    legend('Ideal Test Y', model, 'Location','northwest');
    set(gca,'fontsize',12);
    saveas(gcf, fig_save_path);
    clf
end
