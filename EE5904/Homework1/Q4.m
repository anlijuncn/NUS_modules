% For ANN, updates weithts and bias by following prcrodures
% w(n+1) = w(n) + \eta * e(n) * x(n)
% e(n) = d(n) - y(n) is error signal
% \eta > 0 
% Inputs to ANN
% Training sample, For AND
X = [0 0.8 1.6 3 4 5];
% true labels
D = [0.5 1 4 5 6 9];
% X = [0 1];
% % true labels
% D = [1 0];
% Function for ANN is going to converge 
NAME = 'LMS';
% hyperparameter, change here to adjust
eta = 0.028;
EPOCH = 100; 

% Train
ANN_Train(X, D, eta, 0, EPOCH, NAME)

function ANN_Train(X, D, eta, seed, EPOCH, NAME)
    % function ANN_Train(X, D, eta, EPOCH, NAME)
    % Train a ANN iteratively u%sing given data and hyperparameters
    % Inputs: 
    %     - X: Training sample
    %     - D: True label of training sample
    %     - eta: learning rate
    %     - seed: Random seed to make sure replicable
    %     - EPOCH: The number of passes through the entire training dataset
    %     - NAME: Name of function ANN is learning to converge
    % Outputs:
    %     - Final Weights: Weights learned from training samples
    %     - Weight tractjactory figure:
    %          A figure indicate how weights changes
    % Written by AN Lijun for EE5904
    
    % firstly get the shape of training samples
    [dim_x, nb_x] = size(X);
    % add bias to X 
    X_new = [ones(1, nb_x); X];
    % Randomly Initialize weights with given seed
    rng(seed, 'twister');
    W = rand(1, dim_x + 1); 
    disp(W);

    % Arrays to store weights for each epoch
    weights = zeros(1, dim_x + 1, EPOCH);
    % start to learn
    iter = 0;
    disp('Start Training');
    while iter < EPOCH
        % display iter, weight
        fprintf('Iter: %d \n', iter);
        disp('Weight vector');
        disp(W);
        weights(:, :, iter+1) = W;
        iter = iter + 1; 
        % make prediction using W and X
        Y = W*X_new; % regression
        E = D - Y; % error signal
        W = W + eta * E * X_new'; % update weights
    end
    % Draw weights and bias trajectory 
    % we only need to draw for epochs from 0 to iter
    % call weight_trajectory() function to draw weight tractjectory
    weight_trajectory(weights, iter, eta, NAME)
    % call Plot_dots_line() function to visualize classification
    % get dot1, dot2, dot3, dot4, dot5, dot6, line, vis_outpath and vis_title
    line = [W(2) W(1)];
    vis_title = ['Visualization of Classification for ' NAME];
    vis_outpath = [vis_title num2str(eta)];
    vis_outpath = [vis_outpath '.png'];
    
    Plot_dots_line([0 0.5], [0.8 1], [1.6 4], [3 5], [4 6], [5 9], line, vis_outpath, vis_title)
    
end

function weight_trajectory(weights, iter, eta, NAME)
    % function weight_trajectory(weights, iter, title, outpath)
    % Draw weights and bias trajectory for ANN training
    % Inputs:
    %      - weights: A 3-D array storing all weights during training
    %      - iter: Actual iteration in training process
    %      - eta: Learning rate
    %      - NAME: Name of function ANN is learning to converge
    % Outputs:
    %     - Weight tractjactory figure:
    %          A figure indicate how weights changes
    % Written by AN Lijun for EE5904
    
    fig_title = ['Weights Trajectory for ' NAME];
    outpath = [fig_title num2str(eta)];
    outpath = [outpath '.png'];
    % get number of weights need to plot
    [~, nb_weights, ~] = size(weights);
    % plot weight tractjectory
    x = 1:1:iter;
    % set dot shape and color for each weight
    A = {};
    for k = 1:nb_weights
        % get kth weight
        weight_k = weights(:,k,1:iter);
        weight_k = reshape(weight_k, [1 iter]);
        plot(x, weight_k);
        hold on
        % add legend
        if k == 1
            A = [A 'Bias weight'];
        else
            leg = sprintf('Weight: %d', k-1);
            A = [A cellstr(leg)];
        end
    end
    legend(A);
    % set xlabel and ylabel, title
    xlabel('Epoch')
    ylabel('Weight value')
    % add figure title
    title(fig_title)
    % save figure
    saveas(gcf, outpath);
    clf;
end

function Plot_dots_line(dot1, dot2, dot3, dot4, dot5, dot6, line, outpath, fig_title)
    % Plot_dots_line(dot1, dot2, dot3, dot4, line, outpath)

    % The function is to plot dots with different classes 
    % Inputs:
    %     - dot 1: [x y]
    %     - dot 2: [x y]
    %     - dot 3: [x y]
    %     - dot 4: [x y]
    %     - dot 5: [x y]
    %     - dot 6: [x y]
    %     for dots, (x, y) are coordinates 
    %     - line: 
    %            (k,b) y=kx+b
    %     - outpath: Outpath for saving figure
    %     - fig_title: Title of figure
    % Output:
    %     A figure saved in outpath
    % Written by AN Lijun for EE5904
    
    % we firstly define arrays saving 2 classes dots
    x = [];
    y = [];
    % dot1
    x = [x dot1(1)];
    y = [y dot1(2)];
    % dot2
    x = [x dot2(1)];
    y = [y dot2(2)];
    % dot3
    x = [x dot3(1)];
    y = [y dot3(2)];
    % dot4
    x = [x dot4(1)];
    y = [y dot4(2)];
    % dot5
    x = [x dot5(1)];
    y = [y dot5(2)];
    % dot6
    x = [x dot6(1)];
    y = [y dot6(2)];
    % draw scatter and line in one figure
    sz= 250;
    scatter(x, y, sz, 'o');
    hold on 
    hline = refline(line(1), line(2));
    hline.Color = 'k';
    % add legend
    legend('Training Samples', 'Separator')
    % add axis label 
    xlabel('x')
    ylabel('y')
    % add figure title
    title(fig_title)
    % save figure
    saveas(gcf, outpath);
end

