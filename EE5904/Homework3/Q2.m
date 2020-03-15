%% EE5904 Part 1 Homemwork3 Q2
% Writen by An Lijun for EE5904
%% Data preparation and parameters
% load minst database
load('MNIST_database.mat');
% Since my student ID is A0194872N, 
% so I need to slect 2 and 7 as class one
% 0, 1, 3, 4, 5, 6, 8, 9 as class zero
TrN = length(train_classlabel);
TeN = length(test_classlabel);
trainIdx = find(train_classlabel == 2 | train_classlabel == 7);
testIdx = find(test_classlabel == 2 | test_classlabel == 7);
trainLabel = zeros(1, TrN);
testLabel = zeros(1, TeN);
% assign zero and one class labels
trainLabel(trainIdx) = 1;
testLabel(testIdx) = 1;
%% Run experiment Q2_1
lambda_set = [0, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000, 10000];
for i = 1:length(lambda_set)
    lambda = lambda_set(i);
    disp(['Processing on lambda: ' num2str(lambda)]);
    [TrPred, TePred] = RBFN_reg(train_data, trainLabel, test_data, lambda);
    fig_title = ['The performance of RBFN-Reg (Lambda: ' num2str(lambda) ')'];
    fig_save_path = fullfile(outdir, ['q2_1_lambda_' num2str(lambda) '.png']);
    evaluation(trainLabel, TrPred, testLabel, TePred, fig_title, fig_save_path, lambda);
end
%% Run experiment Q2_2
width_set = [0.1 1 10 100 1000];
for i = 1:length(width_set)
    width = width_set(i);
    disp(['Processing on width: ' num2str(width)]);
    [TrPred, TePred] = RBFN_FCSB(train_data, trainLabel, test_data, 100, width);
    fig_title = ['The performance of RBFN-FCSB (Width: ' num2str(width) ')'];
    fig_save_path = fullfile(outdir, ['q2_2_width_' num2str(width) '.png']);
    evaluation(trainLabel, TrPred, testLabel, TePred, fig_title, fig_save_path, width);
end
%% Run experiment Q2_3
width_set = [0.1 1 10 100 1000];
outdir = 'C:\Users\e0383065\Documents\MATLAB\EE5904\Part1\Homework3\Q2_outputs\Q2_3';
% get centers from K means clustering
[centers] = K_means_clustering(train_data, 2, TrN);
% get means of class 0 and class 1
[means] = get_means(train_data, trainLabel);
% visualize cluster center and means 
vis_path = fullfile(outdir, 'visuzlaiztion.png');
visualize_mean_center(means, centers, vis_path);

for i = 1:length(width_set)
    width = width_set(i);
    disp(['Processing on width: ' num2str(width)]);
    [TrPred, TePred] = RBFN_KMeans(train_data, trainLabel, test_data, TrN, width, centers);
    fig_title = ['The performance of RBFN-KMeans (Width: ' num2str(width) ')'];
    fig_save_path = fullfile(outdir, ['q2_3_width_' num2str(width) '.png']);
    evaluation(trainLabel, TrPred, testLabel, TePred, fig_title, fig_save_path, width);
end
%% functions used in Q2
function [TrPred, TePred] = RBFN_reg(train_x, train_y, test_x, lambda)
    % Fit a RBFN model with Fixed Centers Selected at Random (RBFN_FCSB):
    %    1.Using training data points as center
    %    2.Adding regularization term lambda 
    % calculate distance and interpolation matrix
    tr_dist = distance_matrix(train_x, train_x);
    te_dist = distance_matrix(test_x, train_x);
    [tr_phi] = RBF_Gaussian(tr_dist, 10000);
    [te_phi] = RBF_Gaussian(te_dist, 10000);
    % add bias to phi
    tr_phi = [ones(size(train_x, 2), 1), tr_phi];
    te_phi = [ones(size(test_x, 2), 1), te_phi];
    % get unique weight matrix w
    w = pinv(tr_phi'*tr_phi + lambda*eye(size(train_x,2) + 1))*tr_phi'*train_y';
    % making predicton using te_phi
    TePred = te_phi * w;
    TrPred = tr_phi * w;
end

function [TrPred, TePred] = RBFN_FCSB(train_x, train_y, test_x, M, width)
    % Fit a RBFN model with Fixed Centers Selected at Random (RBFN_FCSB):
    %    1.Randomly pick M points from training set as centers
    %    2.Adjust width  
    % Radomly select M points as centers
    rand_index = randperm(size(train_x, 2), M);
    centers = train_x(:, rand_index);
    % calculate coefficients
    coef = width^2;
    % calculate distance and interpolation matrix
    tr_dist = distance_matrix(train_x, centers);
    te_dist = distance_matrix(test_x, centers);
    [tr_phi] = RBF_Gaussian(tr_dist, coef);
    [te_phi] = RBF_Gaussian(te_dist, coef);
    % add bias to phi
    tr_phi = [ones(size(train_x, 2), 1), tr_phi];
    te_phi = [ones(size(test_x, 2), 1), te_phi];
    % get unique weight matrix w
    w = pinv(tr_phi)*train_y';
    % making predicton using te_phi
    TePred = te_phi * w;
    TrPred = tr_phi * w;
end

function [TrPred, TePred] = RBFN_KMeans(train_x, train_y, test_x, TrN, width, centers)
    % Fit a RBFN model with K means:
    %    1.Using 2 center from K means as centers of RBFN
    %    2.Set Width = 10 
    % calculate coefficients
    coef = width^2;
    % calculate distance and interpolation matrix
    tr_dist = distance_matrix(train_x, centers);
    te_dist = distance_matrix(test_x, centers);
    [tr_phi] = RBF_Gaussian(tr_dist, coef);
    [te_phi] = RBF_Gaussian(te_dist, coef);
    % add bias to phi
    tr_phi = [ones(size(train_x, 2), 1), tr_phi];
    te_phi = [ones(size(test_x, 2), 1), te_phi];
    % get unique weight matrix w
    w = pinv(tr_phi)*train_y';
    % making predicton using te_phi
    TePred = te_phi * w;
    TrPred = tr_phi * w;

end

function [centers] = K_means_clustering(train_x, num_centers, TrN)
    % Running K Means Clusters on training data given center number
    % randomly pick 2 points as initalization
    init_index = randperm(TrN, num_centers);
    centers = train_x(:, init_index);
    for i=1:1000 %loop untill convergence
        distance1=dist(train_x',centers(:,1));%calculate train data distance to 1,2
        distance2=dist(train_x',centers(:,2));
        center1ind=distance1>distance2;
        center2ind=distance1<distance2;
        cluster1=train_x(:,center1ind);%divide into 2 part
        cluster2=train_x(:,center2ind);
        center(:,1)=mean(cluster1,2);
        center(:,2)=mean(cluster2,2);
        clear cluster1
        clear cluster2
    end
end

function [means] = get_means(train_x, trainLabel)
    % Get means of class 0 and class 1 in training set
    train_zero_idx = trainLabel == 0;
    train_one_idx = trainLabel == 1;
    train_zero_data = train_x(:, train_zero_idx);
    train_one_data = train_x(:, train_one_idx);
    % get means
    means = zeros(size(train_x, 1), 2);
    means(:, 1) = mean(train_zero_data, 2);
    means(:, 2) = mean(train_one_data, 2);
end

function visualize_mean_center(means, centers, fig_save_path)
    figure(1)
    tmp1=reshape(centers(:,1),28,28); 
    subplot(2,2,1);
    imshow(zscore(tmp1));
    title('center 1');
    tmp2=reshape(centers(:,2),28,28);
    subplot(2,2,2);
    imshow(zscore(tmp2));
    title('center 2');

    subplot(2,2,3);
    imshow(zscore(reshape(double(means(:, 1)),28,28)));
    title('mean 0');
    subplot(2,2,4);
    imshow(zscore(reshape(double(means(:, 2)),28,28)));
    title('mean 1');
    saveas(gcf, fig_save_path);
    clf
end

function [dist] = distance_matrix(test_x, train_x)
    % Calculate distance matrix using test_x and train_x 
    % Here we are using Eulcidean distance^2, for reduce computation
    dist = zeros(size(test_x, 2), size(train_x, 2));
    % calculate Euclidean distance 
    % note that for each element is a 784 dim vector
    for i = 1:size(test_x, 2)
        for j = 1:size(train_x, 2)
            r2 = dot(test_x(:, i) - train_x(:, j), test_x(:, i) - train_x(:, j));
            dist(i, j) = r2;
        end
    end

end

function [interpolation] = RBF_Gaussian(dist, std)
    % Gaussian RBF function
    interpolation = exp(dist*-1/(2*std));
end

function evaluation(TrLabel, TrPred, TeLabel, TePred, fig_title, fig_save_path, lambda)
    % Function to evealuate model performance, Copy from Assignment
    TrAcc = zeros(1,1000);
    TeAcc = zeros(1,1000);
    thr = zeros(1,1000);
    TrN = length(TrLabel);
    TeN = length(TeLabel);
    for i = 1:1000
        t = (max(TrPred)-min(TrPred)) * (i-1)/1000 + min(TrPred);
        thr(i) = t;
        TrAcc(i) = (sum(TrLabel(TrPred<t)==0) + sum(TrLabel(TrPred>=t)==1)) / TrN;
        TeAcc(i) = (sum(TeLabel(TePred<t)==0) + sum(TeLabel(TePred>=t)==1)) / TeN;
    end
    % find highest Train acc and test acc
    [best_tr_acc, trIdx] = max(TrAcc);
    [best_te_acc, ~] = max(TeAcc);
    % find best threshold for TrAcc and TeAcc 
    disp(['For Width: ' num2str(lambda) '| Best threshold for train: ' num2str(thr(trIdx)), '| Best Train Acc: ' num2str(best_tr_acc) '| Coreesponding Test Acc: ' num2str(best_te_acc)]);
    plot(thr,TrAcc,'.- ',thr,TeAcc,'^-');
    legend('Train','Test', 'Location','northwest');
    title(fig_title);
    xlabel('Threshold');
    ylabel('Accuracy');
    set(gca,'fontsize',12);
    saveas(gcf, fig_save_path);
    clf

end