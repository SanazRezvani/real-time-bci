clc;
clear;
close all;

%% Real-Time EEG Motor Imagery Decoding Simulation
% This script extends the FBCSP project toward a simulated real-time BCI setting.
% Offline phase:
%   - Train FBCSP + classifier using labelled trials
%
% Real-time simulation phase:
%   - Treat test trials as an incoming EEG stream
%   - Apply sliding windows
%   - Extract FBCSP features
%   - Generate predictions
%   - Log processing latency

%% Configuration
config.dataset_path = 'data_set_IVa_al.mat';
config.spatial_filter = 'CAR';
config.filter_order = 3;
config.train_ratio = 0.70;
config.num_csp_pairs = 1;

config.trial_length_s = 3.5;
config.window_length_s = 1.0;
config.step_size_s = 0.25;

config.classifier_type = 'KNN'; % options: 'LDA', 'SVM', 'KNN'

config.filter_bank = [
     8 12;
    10 14;
    12 16;
    14 18;
    16 20;
    18 22;
    20 24;
    22 26;
    24 28;
    26 30
];

results_folder = 'results';

if ~exist(results_folder, 'dir')
    mkdir(results_folder);
end

%% Load dataset
load(config.dataset_path);

raw_eeg_signal = 0.1 * double(cnt);
sampling_rate = nfo.fs;

trial_start_samples = mrk.pos;
trial_labels = mrk.y;

trial_length_samples = round(config.trial_length_s * sampling_rate);
window_length_samples = round(config.window_length_s * sampling_rate);    % Converts seconds to number of samples
step_size_samples = round(config.step_size_s * sampling_rate);

electrode_positions = [nfo.xpos'; nfo.ypos'];

class_1_label = 1; % right hand
class_2_label = 2; % foot

fprintf('\nReal-time EEG decoding simulation started...\n');
fprintf('Window length: %.2f s\n', config.window_length_s);
fprintf('Step size: %.2f s\n', config.step_size_s);
fprintf('Number of sub-bands: %d\n\n', size(config.filter_bank, 1));

%% Extract full trials from raw EEG first
class_1_trials_raw = [];
class_2_trials_raw = [];

class_1_count = 0;
class_2_count = 0;

for trial_idx = 1:length(trial_labels)

    current_label = trial_labels(trial_idx);

    if isnan(current_label)
        continue;
    end

    start_sample = trial_start_samples(trial_idx);
    end_sample = start_sample + trial_length_samples - 1;

    if end_sample > size(raw_eeg_signal, 1)
        continue;
    end

    trial_signal = raw_eeg_signal(start_sample:end_sample, :);

    if current_label == class_1_label
        class_1_count = class_1_count + 1;
        class_1_trials_raw(:, :, class_1_count) = trial_signal;

    elseif current_label == class_2_label
        class_2_count = class_2_count + 1;
        class_2_trials_raw(:, :, class_2_count) = trial_signal;
    end
end

%% Train/test split
num_class_1_trials = size(class_1_trials_raw, 3);
num_class_2_trials = size(class_2_trials_raw, 3);

num_train_class_1 = floor(config.train_ratio * num_class_1_trials);
num_train_class_2 = floor(config.train_ratio * num_class_2_trials);

train_class_1_raw = class_1_trials_raw(:, :, 1:num_train_class_1);
train_class_2_raw = class_2_trials_raw(:, :, 1:num_train_class_2);

test_class_1_raw = class_1_trials_raw(:, :, num_train_class_1 + 1:end);
test_class_2_raw = class_2_trials_raw(:, :, num_train_class_2 + 1:end);

%% Offline FBCSP training
fprintf('Offline training phase...\n');

fbcsp_model = struct();
fbcsp_model.config = config;
fbcsp_model.sampling_rate = sampling_rate;
fbcsp_model.electrode_positions = electrode_positions;
fbcsp_model.subband = struct();

train_features_class_1_all_bands = [];
train_features_class_2_all_bands = [];

for band_idx = 1:size(config.filter_bank, 1)               % Building FBCSP

    low_cutoff_hz = config.filter_bank(band_idx, 1);
    high_cutoff_hz = config.filter_bank(band_idx, 2);

    fprintf('Training sub-band %d: %d-%d Hz\n', ...
        band_idx, low_cutoff_hz, high_cutoff_hz);

    [b, a] = butter( ...
        config.filter_order, ...
        [low_cutoff_hz high_cutoff_hz] / (sampling_rate / 2), ...
        'bandpass');

    train_class_1_band = filter_trials(train_class_1_raw, b, a);      % Applies filter to all trials
    train_class_2_band = filter_trials(train_class_2_raw, b, a);

    train_class_1_band = apply_spatial_filter_to_trials( ...
        train_class_1_band, config.spatial_filter, electrode_positions);

    train_class_2_band = apply_spatial_filter_to_trials( ...
        train_class_2_band, config.spatial_filter, electrode_positions);

    csp_filters = compute_csp_filters( ...
        train_class_1_band, ...
        train_class_2_band, ...
        config.num_csp_pairs);

    train_features_class_1 = extract_csp_variance_features(train_class_1_band, csp_filters);
    train_features_class_2 = extract_csp_variance_features(train_class_2_band, csp_filters);

    train_features_class_1_all_bands = [
        train_features_class_1_all_bands;
        train_features_class_1
    ];

    train_features_class_2_all_bands = [
        train_features_class_2_all_bands;
        train_features_class_2
    ];

% Save model parameters
    fbcsp_model.subband(band_idx).low_cutoff_hz = low_cutoff_hz;
    fbcsp_model.subband(band_idx).high_cutoff_hz = high_cutoff_hz;
    fbcsp_model.subband(band_idx).b = b;
    fbcsp_model.subband(band_idx).a = a;
    fbcsp_model.subband(band_idx).csp_filters = csp_filters;
end

train_features = [
    train_features_class_1_all_bands, ...
    train_features_class_2_all_bands
];

train_labels = [
    ones(1, size(train_features_class_1_all_bands, 2)), ...
    2 * ones(1, size(train_features_class_2_all_bands, 2))
];

fprintf('\nTraining feature matrix: %d features x %d trials\n', ...
    size(train_features, 1), size(train_features, 2));

%% Train classifier
switch upper(config.classifier_type)

    case 'LDA'
        classifier_model = fitcdiscr(train_features', train_labels', ...
            'DiscrimType', 'pseudoLinear');

    case 'SVM'
        classifier_model = fitcsvm(train_features', train_labels');

    case 'KNN'
        classifier_model = fitcknn(train_features', train_labels', ...
            'NumNeighbors', 5);

    otherwise
        error('Unsupported classifier type.');
end

fbcsp_model.classifier = classifier_model;         % Stores model for real-time use

fprintf('Classifier trained: %s\n\n', config.classifier_type);

%% Create simulated EEG stream (continuous signal) from test trials
% This stream is created by concatenating unseen test trials.
% Ground-truth labels are stored for later comparison.
[test_stream, test_stream_labels] = concatenate_trials_as_stream( ...
    test_class_1_raw, ...
    test_class_2_raw, ...
    class_1_label, ...
    class_2_label);

num_stream_samples = size(test_stream, 1);

%% Real-time sliding-window simulation
fprintf('Real-time simulation phase...\n');

window_start_indices = 1:step_size_samples:(num_stream_samples - window_length_samples + 1);

num_windows = length(window_start_indices);

prediction_log = table();
predictions = zeros(num_windows, 1);
true_labels = zeros(num_windows, 1);
latencies_ms = zeros(num_windows, 1);
window_times_s = zeros(num_windows, 1);

for window_idx = 1:num_windows

    start_idx = window_start_indices(window_idx);
    end_idx = start_idx + window_length_samples - 1;

    eeg_window = test_stream(start_idx:end_idx, :);        % Simulates incoming EEG chunk

    tic;                 % Measures computation time per window

    feature_vector = extract_fbcsp_features_from_window(eeg_window, fbcsp_model);

    predicted_label = predict(fbcsp_model.classifier, feature_vector');

    latency_ms = toc * 1000;     % 

    current_true_label = mode(test_stream_labels(start_idx:end_idx));

    predictions(window_idx) = predicted_label;
    true_labels(window_idx) = current_true_label;
    latencies_ms(window_idx) = latency_ms;
    window_times_s(window_idx) = start_idx / sampling_rate;

    prediction_log = [
        prediction_log;
        table( ...
            window_idx, ...
            window_times_s(window_idx), ...
            predicted_label, ...
            current_true_label, ...
            latency_ms, ...
            'VariableNames', { ...
                'WindowIndex', ...
                'WindowStartTime_s', ...
                'PredictedLabel', ...
                'TrueLabel', ...
                'Latency_ms'})
    ];
end

%% Evaluation
stream_accuracy = mean(predictions == true_labels) * 100;
mean_latency = mean(latencies_ms);
max_latency = max(latencies_ms);

fprintf('\nReal-time simulation results:\n');
fprintf('Number of windows: %d\n', num_windows);
fprintf('Streaming accuracy: %.2f%%\n', stream_accuracy);
fprintf('Mean latency: %.4f ms\n', mean_latency);
fprintf('Max latency: %.4f ms\n', max_latency);

%% Save prediction log
writetable(prediction_log, fullfile(results_folder, 'prediction_log.csv'));

%% Plot predictions over time
figure;
plot(window_times_s, true_labels, 'k--', 'LineWidth', 1.5);
hold on;
plot(window_times_s, predictions, 'b', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Class Label');
yticks([1 2]);
yticklabels({'Right Hand', 'Foot'});
title('Real-Time EEG Decoding Simulation: Predictions Over Time');
legend({'True Label', 'Predicted Label'}, 'Location', 'best');
grid on;
saveas(gcf, fullfile(results_folder, 'realtime_predictions.png'));

%% Plot latency over time
figure;
plot(window_times_s, latencies_ms, 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Latency (ms)');
title('Processing Latency Over Time');
grid on;
saveas(gcf, fullfile(results_folder, 'latency_over_time.png'));

%% Save results
save(fullfile(results_folder, 'realtime_simulation_results.mat'), ...
    'config', ...
    'fbcsp_model', ...
    'prediction_log', ...
    'predictions', ...
    'true_labels', ...
    'latencies_ms', ...
    'stream_accuracy', ...
    'mean_latency', ...
    'max_latency');

fprintf('\nSimulation complete. Results saved in the results folder.\n');

%% Helper functions

% Applies band-pass filter to all trials
function filtered_trials = filter_trials(trials, b, a)
%FILTER_TRIALS Apply zero-phase filtering to each EEG trial.

    filtered_trials = zeros(size(trials));

    for trial_idx = 1:size(trials, 3)
        filtered_trials(:, :, trial_idx) = filtfilt(b, a, trials(:, :, trial_idx));
    end
end

% Applies CAR/Laplacian per trial
function spatial_trials = apply_spatial_filter_to_trials(trials, spatial_filter_type, electrode_positions)
%APPLY_SPATIAL_FILTER_TO_TRIALS Apply spatial filtering to each trial.

    spatial_trials = zeros(size(trials));

    for trial_idx = 1:size(trials, 3)
        spatial_trials(:, :, trial_idx) = apply_spatial_filter( ...
            trials(:, :, trial_idx), ...
            spatial_filter_type, ...
            electrode_positions);
    end
end

function features = extract_csp_variance_features(trials, csp_filters)
%EXTRACT_CSP_VARIANCE_FEATURES Extract CSP variance features from trials.

    num_trials = size(trials, 3);
    num_filters = size(csp_filters, 2);
    features = zeros(num_filters, num_trials);

    for trial_idx = 1:num_trials
        trial_data = trials(:, :, trial_idx)'; % channels x samples
        projected_trial = csp_filters' * trial_data;
        features(:, trial_idx) = var(projected_trial, 0, 2);
    end
end

function feature_vector = extract_fbcsp_features_from_window(eeg_window, fbcsp_model)
%EXTRACT_FBCSP_FEATURES_FROM_WINDOW Extract FBCSP features from one EEG window.

    config = fbcsp_model.config;
    electrode_positions = fbcsp_model.electrode_positions;

    feature_vector = [];

    for band_idx = 1:length(fbcsp_model.subband)

        b = fbcsp_model.subband(band_idx).b;
        a = fbcsp_model.subband(band_idx).a;
        csp_filters = fbcsp_model.subband(band_idx).csp_filters;

        filtered_window = filtfilt(b, a, eeg_window);

        spatial_window = apply_spatial_filter( ...
            filtered_window, ...
            config.spatial_filter, ...
            electrode_positions);

        window_data = spatial_window'; % channels x samples
        projected_window = csp_filters' * window_data;

        band_features = var(projected_window, 0, 2);

        feature_vector = [
            feature_vector;
            band_features
        ];
    end
end

function [stream, stream_labels] = concatenate_trials_as_stream( ...
    test_class_1_raw, test_class_2_raw, class_1_label, class_2_label)
%CONCATENATE_TRIALS_AS_STREAM Concatenate test trials to simulate continuous EEG.

    stream = [];
    stream_labels = [];

    num_class_1_trials = size(test_class_1_raw, 3);
    num_class_2_trials = size(test_class_2_raw, 3);

    min_trials = min(num_class_1_trials, num_class_2_trials);

    for trial_idx = 1:min_trials

        trial_1 = test_class_1_raw(:, :, trial_idx);
        trial_2 = test_class_2_raw(:, :, trial_idx);

        stream = [
            stream;
            trial_1;
            trial_2
        ];

        stream_labels = [
            stream_labels;
            class_1_label * ones(size(trial_1, 1), 1);
            class_2_label * ones(size(trial_2, 1), 1)
        ];
    end
end