% For ANN, updates weithts and bias by following prcrodures
% w(n+1) = w(n) + \eta * e(n) * x(n)
% e(n) = d(n) - y(n) is error signal
% \eta > 0 
% Inputs to ANN
% Training sample, For AND
X = [0 0 1 1; 0 1 0 1];
% true labels
D = [0 0 0 1];
% X = [0 1];
% % true labels
% D = [1 0];
% Function for ANN is going to converge 
NAME = 'AND';
% hyperparameter, change here to adjust
eta = 0.05;
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
        Y = hardlim(W*X_new); % activation function hardlim
        E = D - Y; % error signal
        W = W + eta * E * X_new'; % update weights
    end
    % Draw weights and bias trajectory 
    % we only need to draw for epochs from 0 to iter
    % call weight_trajectory() function to draw weight tractjectory
    weight_trajectory(weights, iter, eta, NAME)
    % call Plot_dots_line() function to visualize classification
    % get dot1, dot2, dot3, dot4, line, vis_outpath and vis_title
    train = cat(1, D, X);
    line1 = [-W(1)/W(2) 0];
    if W(dim_x + 1) ~= 0
        line2 = [0 -W(1)/W(dim_x + 1)];
    else
        line2 = [-W(1)/W(2) 1];
    end
    vis_title = ['Visualization of Classification for ' NAME];
    vis_outpath = [vis_title num2str(eta)];
    vis_outpath = [vis_outpath '.png'];
    
    if dim_x == 2
        Plot_dots_line(train(:,1)', train(:,2)', train(:,3)', train(:,4)', line1, line2, vis_outpath, vis_title)
    else
        Plot_dots_line_onedim([1 0 0], [0 1 0], line1, line2, vis_outpath, vis_title)
    end
    
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

function Plot_dots_line(dot1, dot2, dot3, dot4, line1, line2, outpath, fig_title)
    % Plot_dots_line(dot1, dot2, dot3, dot4, line, outpath)

    % The function is to plot dots with different classes 
    % Inputs:
    %     - dot 1: [y x_1 x_2]
    %     - dot 2: [y x_1 x_2]
    %     - dot 3: [y x_1 x_2]
    %     - dot 4: [y x_1 x_2]
    %     for dots, y is class label, x_1 and x_2 are coordinates 
    %     - line1: 
    %            first spot separator cross
    %     - line2:
    %            second spot separator cross
    %     - outpath: Outpath for saving figure
    %     - fig_title: Title of figure
    % Output:
    %     A figure saved in outpath
    % Written by AN Lijun for EE5904
    
    % we firstly define arrays saving 2 classes dots
    class_0_x1 = [];
    class_0_x2 = [];
    class_1_x1 = [];
    class_1_x2 = [];
    % check which dot is beloging to which class
    % dot1
    if dot1(1) == 0
        class_0_x1 = [class_0_x1, dot1(2)];
        class_0_x2 = [class_0_x2, dot1(3)];
    else
        class_1_x1 = [class_1_x1, dot1(2)];
        class_1_x2 = [class_1_x2, dot1(3)];
    end
    % dot2
    if dot2(1) == 0
        class_0_x1 = [class_0_x1, dot2(2)];
        class_0_x2 = [class_0_x2, dot2(3)];
    else
        class_1_x1 = [class_1_x1, dot2(2)];
        class_1_x2 = [class_1_x2, dot2(3)];
    end
    % dot3
    if dot3(1) == 0
        class_0_x1 = [class_0_x1, dot3(2)];
        class_0_x2 = [class_0_x2, dot3(3)];
    else
        class_1_x1 = [class_1_x1, dot3(2)];
        class_1_x2 = [class_1_x2, dot3(3)];
    end
    % dot 4
    if dot4(1) == 0
        class_0_x1 = [class_0_x1, dot4(2)];
        class_0_x2 = [class_0_x2, dot4(3)];
    else
        class_1_x1 = [class_1_x1, dot4(2)];
        class_1_x2 = [class_1_x2, dot4(3)];
    end
    % draw scatter and line in one figure
    sz= 250;
    scatter(class_0_x1, class_0_x2, sz, 'o');
    hold on 
    scatter(class_1_x1, class_1_x2, sz, 'd');
    hold on 
    % Plot separator
    plot(line1, line2, 'k');
    % add legend
    legend('Class 0', 'Class 1', 'Separator')
    % add axis label 
    xlabel('x_1')
    ylabel('x_2')
    % add figure title
    title(fig_title)
    % save figure
    saveas(gcf, outpath);
end

function Plot_dots_line_onedim(dot1, dot2, line1, line2, outpath, fig_title)
    % Plot_dots_line(dot1, dot2, dot3, dot4, line, outpath)

    % The function is to plot dots with different classes 
    % Inputs:
    %     - dot 1: [y x_1 x_2]
    %     - dot 2: [y x_1 x_2]
    %     for dots, y is class label, x_1 and x_2 are coordinates 
    %     - line1: 
    %            first spot separator cross
    %     - line2:
    %            second spot separator cross
    %     - outpath: Outpath for saving figure
    %     - fig_title: Title of figure
    % Output:
    %     A figure saved in outpath
    % Written by AN Lijun for EE5904
    
    % we firstly define arrays saving 2 classes dots
    class_0_x1 = [];
    class_0_x2 = [];
    class_1_x1 = [];
    class_1_x2 = [];
    % check which dot is beloging to which class
    % dot1
    if dot1(1) == 0
        class_0_x1 = [class_0_x1, dot1(2)];
        class_0_x2 = [class_0_x2, dot1(3)];
    else
        class_1_x1 = [class_1_x1, dot1(2)];
        class_1_x2 = [class_1_x2, dot1(3)];
    end
    % dot2
    if dot2(1) == 0
        class_0_x1 = [class_0_x1, dot2(2)];
        class_0_x2 = [class_0_x2, dot2(3)];
    else
        class_1_x1 = [class_1_x1, dot2(2)];
        class_1_x2 = [class_1_x2, dot2(3)];
    end
    % draw scatter and line in one figure
    sz= 250;
    scatter(class_0_x1, class_0_x2, sz, 'o');
    hold on 
    scatter(class_1_x1, class_1_x2, sz, 'd');
    hold on 
    % Plot separator
    plot(line1, line2, 'k');
    % add legend
    legend('Class 0', 'Class 1', 'Separator')
    % add axis label 
    xlabel('x')
    ylabel('y')
    % add figure title
    title(fig_title)
    % save figure
    saveas(gcf, outpath);
end
