%% Script for Q2 Homework 2
% wirtten by Lijun AN for EE5904
%% data prepareation
train_x = -1:0.05:1;
train_y = func_generator(train_x);
test_x = -1:0.1:1;
test_y = func_generator(test_x);
%% parameters setting
seq_mode = false; % whether using sequential mode or batch mode
% number of nodes for 1-n-1 MLP network
nb_nodes = [1 2 3 4 5 6 7 8 9 10 20 50];
learning_rate = 0.001;
epochs = 100;
trainFcn = 'trainbr';
out_dir = '/Users/ljan/Documents/MATLAB/EE5904/Part1/Homework2/Q2_outputs';
sub_questions = 'Q2a';
out_stem = [sub_questions '_' num2str(learning_rate) '_' num2str(epochs) '_' trainFcn '_' num2str(seq_mode)];
%% train and test MLP
for i = 1:length(nb_nodes)
    fprintf('Train and test on MLP: 1-%d-1\n', nb_nodes(i));
    n = nb_nodes(i);
    [pred_y, net] = MLP_model(train_x, train_y, test_x, n, seq_mode, trainFcn, learning_rate, epochs);
    % draw the output
    subplot(3, 4, i);
    plot(test_x, test_y, 'b--', 'LineWidth', 1); hold on;
    plot(test_x, pred_y, 'r', 'LineWidth', 1);hold off;
    title(['n = ' num2str(n)]);
    set(gca,'fontsize',12);
    % make prediction out of train domain
    y1_pred = net(-3);
    y2_pred = net(3);
    disp(y1_pred)
    disp(y2_pred)

end

% save fig
fig_save_path = fullfile(out_dir, [out_stem '.png']);
saveas(gcf, fig_save_path);
clf

%% functions used in this script
function [y] = func_generator(x)
    % function to generate y from x given a specific function
    y = 1.2*sin(pi*x) - cos(2.4*pi*x);
end
function [pred_y, net] = MLP_model(train_x, train_y, test_x, n, seq_mode, trainFcn, lr, epochs)
    % function to train MLP and make prediction on test set
    % initialize a network, since we are doing regression, we need to call
    % Matlab toolbox FITNET
    net = patternnet(n);
    % set learning rate
    net.trainParam.lr = lr;
    net.trainParam.epochs = epochs;
    net.trainParam.showWindow=false;
    net.trainParam.showCommandLine=true;
    if seq_mode
        % train using adapt function
        for epoch = 1:epochs
            net = adapt(net, train_x, train_y);
        end
    else
        % train using train function
        net = train(net, train_x, train_y);
    end
    % making prediction
    pred_y = net(test_x);
    
end