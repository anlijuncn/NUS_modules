%% Script for Q3 Homework 2
% wirtten by Lijun AN for EE5904
%% data prepareation 
% we need to read and process images for Network's input
[train_images, train_images_2D, train_labels] = read_images('/Users/ljan/Documents/MATLAB/EE5904/Part1/Homework2/group_1/train');
[val_images, val_images_2D, val_labels] = read_images('/Users/ljan/Documents/MATLAB/EE5904/Part1/Homework2/group_1/val');
%% parameters setting
seq_mode = true; % whether using sequential mode or batch mode
single_perceptron = true;
% number of nodes for 1-n-1 MLP network
learning_rate = 0.001;
epochs = 500;
nodes = 10;
trainFcn = 'trainbr';
out_dir = '/Users/ljan/Documents/MATLAB/EE5904/Part1/Homework2/Q3_outputs';
sub_questions = 'Q3e';
dim_reduction = 'sequential mode';
out_stem = [sub_questions '_' num2str(learning_rate) '_' num2str(epochs) '_'  num2str(seq_mode) '_'  num2str(single_perceptron) '_' dim_reduction];
fig_save_path = fullfile(out_dir, [out_stem '.png']);
%% call different models to train and draw acc plot
% PCA 100
% tr_down = downsample_image(train_images_2D, 0.125);
% val_down = downsample_image(val_images_2D, 0.125);
% tr_pca = PCA_dim_reduct(train_images, 128);
% val_pca = PCA_dim_reduct(val_images, 128);
% [train_acc, val_acc] = train_Perceptron(tr_pca, train_labels, val_pca, val_labels, epochs);
[train_acc, val_acc] = MLP_model(train_images, train_labels, val_images, val_labels, nodes, seq_mode, learning_rate, epochs);
% plot acc figures
plot(train_acc, 'b--', 'LineWidth', 1); hold on;
plot(val_acc, 'r', 'LineWidth', 1);hold off;
legend('Train Accuracy','Validation Accuracy')
title(['Train and Validation Accuracy for MLP sequential mode(n=10, lr=0.001)']);
saveas(gcf, fig_save_path);
%% functions used in this script
function [images, images_2D, lables] = read_images(img_dir)
    % function to read images and lables of images 
    % get all files uner img_dir
    files_info = dir(img_dir);
    num_images = size(files_info, 1) - 2;
    % define images and lebles to store data
    images = zeros(65536, num_images);
    lables = zeros(1, num_images);
    images_2D = zeros(256, 256, num_images);
    % read every image under img_dir
    for i = 1:num_images
        img_path = fullfile(files_info(i+2).folder, files_info(i+2).name);
        I = imread(img_path);
        images_2D(:, :, i) = I;
        V = I(:);
        images(:, i) = V;
        tmp = strsplit(files_info(i+2).name, {'_', '.'});
        lables(i) = str2double(tmp{2});
        clear I
        clear V
        clear tmp
    end
end

function [train_acc, val_acc] = train_Perceptron(tr_images, tr_labels, val_images, val_labels, epochs)
    % Train Roseenblatt's perceptron (single layer perceptron)
    net = perceptron;
    net.trainParam.epochs = 1;
    net.divideFcn='dividetrain';
    net.trainParam.showWindow=false;
    net.trainParam.showCommandLine=false;
    net = init(net);
    train_acc = zeros(epochs, 1);
    val_acc = zeros(epochs, 1);
    for i = 1:epochs
        net = train(net, tr_images, tr_labels);
        tr_pred = net(tr_images);
        train_acc(i) = sum(tr_pred == tr_labels) / size(tr_labels, 2);
        val_pred = net(val_images);
        val_acc(i) = sum(val_pred == val_labels) / size(val_labels, 2);
        disp(['Epoch ' num2str(i) ', Training acc: ' num2str(train_acc(i)*100) '%. Validation acc: ' num2str(val_acc(i)*100) '%.']);
    end
end

function [images_downsampled] = downsample_image(images, scale)
    % dwonsample images to a certain size
    images_downsampled_2D = imresize(images,scale);
    % reshape it to column vector
    num_images = size(images, 3);
    images_downsampled = reshape(images_downsampled_2D, [], num_images);
end

function [images_pca] = PCA_dim_reduct(images, threshold)
    % reduce dimensionality by PCA
    [~,score,~,~] = pca(images');
    images_pca = score(:, 1:threshold)';

end

function [train_acc, val_acc] = MLP_model(tr_images, tr_labels, val_images, val_labels, n, seq_mode, lr, epochs)
    % function to train MLP and make prediction on test set
    % initialize a network, since we are doing classification, we need to call
    % Matlab toolbox PATTERNNET
    net = patternnet(n);
    net.trainFcn = 'traingdx'; 
    net.performFcn = 'crossentropy';
    % set learning rate
    net.trainParam.lr = lr;
    net.trainParam.epochs = 1;
    net.performParam.regularization=0.1;
    net.trainParam.showWindow=false;
    net.trainParam.showCommandLine=false;
    train_acc = zeros(epochs, 1);
    val_acc = zeros(epochs, 1);
    if seq_mode
        % train using adapt function
        for epoch = 1:epochs
            net = adapt(net, tr_images, tr_labels);
            tr_pred = round(net(tr_images));
            train_acc(epoch) = sum(tr_pred == tr_labels) / size(tr_labels, 2);
            val_pred = round(net(val_images));
            val_acc(epoch) = sum(val_pred == val_labels) / size(val_labels, 2);
            disp(['Epoch ' num2str(epoch) ', Training acc: ' num2str(train_acc(epoch)*100) '%. Validation acc: ' num2str(val_acc(epoch)*100) '%.']);
        end
    else
        % train using train function
        for epoch = 1:epochs
            net = train(net, tr_images, tr_labels);
            tr_pred = round(net(tr_images));
            train_acc(epoch) = sum(tr_pred == tr_labels) / size(tr_labels, 2);
            val_pred = round(net(val_images));
            val_acc(epoch) = sum(val_pred == val_labels) / size(val_labels, 2);
            disp(['Epoch ' num2str(epoch) ', Training acc: ' num2str(train_acc(epoch)*100) '%. Validation acc: ' num2str(val_acc(epoch)*100) '%.']);
        end
    end
    
    
end
