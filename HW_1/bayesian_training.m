%% Homework 1 - pen and papper solution (bayesian model training).
%     Authors:
%        - Afonso Ferreira - 86689
%        - Rita Costa - 95968

%% Reading the dataset table
% Important: change 'current folder' to HW_1
dir_path = fullfile('./data/dataset_example.xlsx');
dataset = readtable(dir_path, 'Range', 'A1:F11', 'ReadVariableNames', false);

%% Computing the predictions and updating confusion matrix
true_classes = zeros(10, 1);
scores = zeros(10, 1);

% actual (P, N) on rows, predicted (P, N) on cols
cm = zeros(2, 2);

for i = 1:size(dataset, 1)
    disp("sample " + num2str(i) + "...");
    sample_temp = table2cell(dataset(i, 2:end));
    [true_score, class_out] = b_classifier(sample_temp, dataset);
    disp("predicted = " + num2str(class_out));
    disp("ground-truth = " + num2str(dataset{i, end}));
    disp(newline);
    % updating confusion matrix
    if dataset{i, end} == 0
        if class_out == 0
            cm(1, 1) = cm(1, 1) + 1; % TP
        else
            cm(1, 2) = cm(1, 2) + 1; % FN
        end
    else
        if class_out == 1
            cm(2, 2) = cm(2, 2) + 1; % TN
        else
            cm(2, 1) = cm(2, 1) + 1; % FP
        end
    end
    true_classes(i, 1) = dataset{i, end};
    scores(i, 1) = true_score;
    
end
% a = {0.6, 'A', 0.2, 0.4};
% b_classifier(a, dataset);

%% Computing F1-Score

tp = cm(2, 2);
fp = cm(1, 2);
fn = cm(2, 1);

F1_score = tp / (tp + (1/2)*(fp + fn));
f1_v2 = 2 / (((tp + fn)/tp) + ((tp+fp)/tp));

%% Bayes function

function [true_score, class_out] = b_classifier(sample, dataset)
% Args:
%     Sample: a vector with 4 values for variables [y1, y2, y3, y4]
%     Dataset: a table containing the dataset

    % Defining sample values
    y1_val = sample{1, 1};
    y2_val = sample{1, 2};
    y3_val = sample{1, 3};
    y4_val = sample{1, 4};

    % Definining variable instances
    y1_data = dataset{1:end, 2};
    y3_data = dataset{1:end, 4};
    y4_data = dataset{1:end, 5};
    
    % Definining class instances
    c_data = dataset{1:end, 6};

    % Variabel set {y1} - prior: Normal distribution params
    mean_y1 = mean(y1_data);    
    c_y1 = (1 / (std(y1_data)*sqrt(2*pi)));
    
%     disp("mean = " + num2str(mean_y1));
%     disp("std = " + num2str(std(y1_data)));
% 
%     disp("cte = " + num2str(c_y1));

    % Variabel set {y1} - prior: Normal distribution prob. function
    P_y1 = @(x) c_y1 * exp(-((x - mean_y1)^2)/(2*std(y1_data)^2));

    % Variabel set {y3, y4} - prior: Normal distribution params
    mean_y3y4 = [mean(y3_data), mean(y4_data)]';
    cov_y3y4 = cov(y3_data, y4_data);
%     disp("mean 2D = " + num2str(mean_y3y4));
%     disp("cov_mat = " + num2str(cov_y3y4));
    % Variabel set {y3, y4} - prior: Normal distribution prob. function (D = 2)
    c_y3y4 = (1 / (( 2*pi) * sqrt(det(cov_y3y4))));
    P_y3y4 = @(x) c_y3y4 * exp( (-1/2) * (x-mean_y3y4)' * inv(cov_y3y4) * (x-mean_y3y4));

    % Class priors
    classes = unique(dataset{:, 6});
    prior_c = zeros(length(classes), 1);

    for i = 1:1:length(classes)
        prior_c(i, 1) = sum(c_data==classes(i))/length(c_data);
    end
    
    % Samples with class 0
    inds_c0 = find(c_data == 0);
    
    % Bayesian probabilities
    % Variabel set {y1} - Bayesian P.: Normal distribution params
    b_mean_y1 = mean(dataset{inds_c0, 2});    
    b_c_y1 = (1 / (std(dataset{inds_c0, 2})*sqrt(2*pi)));
%     disp("mean b1= " + num2str(b_mean_y1));
%     disp("std b1= " + num2str(std(dataset{inds_c0, 2})));
    
    % Variabel set {y1} - Bayesian P.: Normal distribution prob. function
    P_b_y1 = @(x) b_c_y1 * exp(-((x - b_mean_y1)^2)/(2*std(dataset{inds_c0, 2})^2));

    %%%
    % Variabel set {y3, y4} - prior: Normal distribution params
    b_mean_y3y4 = [mean(dataset{inds_c0, 4}), mean(dataset{inds_c0, 5})]';
    b_cov_y3y4 = cov(dataset{inds_c0, 4}, dataset{inds_c0, 5});
    
%     disp("mean b2= " + num2str(b_mean_y3y4));
%     disp("std b2= " + num2str(b_cov_y3y4));
    
    % Variabel set {y3, y4} - prior: Normal distribution prob. function (D = 2)
    b_c_y3y4 = (1 / (( 2*pi) * sqrt(det(b_cov_y3y4))));
    P_b_y3y4 = @(x) b_c_y3y4 * exp( (-1/2) * (x-b_mean_y3y4)' * inv(b_cov_y3y4) * (x-b_mean_y3y4));

    
    bayes_1_c0 = P_b_y1(y1_val);
    bayes_2_c0 = sum(char(dataset{inds_c0, 3}) == y2_val);
    bayes_3_4_c0 = P_b_y3y4([y3_val, y4_val]');
    
    % priors for variable sets
    prior_y1 = P_y1(y1_val);
    prior_y2 = sum(char(dataset{:, 3}) == y2_val);
    prior_y3y4 = P_y3y4([y3_val, y4_val]');
    
    % Numerator for posterior probability (class 0)
    num_val_c0 = (bayes_1_c0 * bayes_2_c0 * bayes_3_4_c0 * prior_c(1, 1)) / (prior_y1*prior_y2*prior_y3y4);
    
    % Displaying posterior probabilities for both classes
    disp("posterior probabilites are:");
    disp(num2str(num_val_c0) + " for class 0.");
    disp(num2str(1-num_val_c0) + " for class 1.");
    
    % Output class
    if num_val_c0 >= 0.5
        class_out = 0;
%         true_score = num_val_c0;
    else
        class_out = 1;
%         true_score = 1 - num_val_c0;
    end
    
    if dataset{1:end, 6} == 0
        true_score = num_val_c0;
    else
        true_score = 1 - num_val_c0;
    end
    
end


