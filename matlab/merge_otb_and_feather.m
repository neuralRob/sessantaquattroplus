clear
close all
clc

%% File paths and settings
feather_file = "/Users/roberto/Library/CloudStorage/OneDrive-ScuolaSuperioreSant\'Anna/PhD/experiments/careggi-post-review/20250401/S2/feather/20250401_142445_default_filename.csv";
otb_file = "/Users/roberto/Library/CloudStorage/OneDrive-ScuolaSuperioreSant\'Anna/PhD/experiments/careggi-post-review/20250401/S2/otb/20250401_142444.csv";

feather_fs = 200;      % Target sampling frequency (Hz) for Feather and OTB after downsampling
otb_fs = 2000;         % Original OTB sampling frequency (Hz)
alpha_env = 0.11;      % Exponential envelope smoothing coefficient

%% 1. Load CSV files
otb_data = readtable(otb_file);
feather_data = readtable(feather_file);

%% 2. Uniformly resample the Feather data
% (Assume that the feather file has a "delta" column and then several features,
%  one of which is 'trigger_otb'. We ignore the first two columns.)
delta_ms = feather_data.delta;           % time between samples in ms
t_feather = cumsum([0; delta_ms(1:end-1)])/1000;  % time vector in seconds

% Create uniform time vector at feather_fs
t_uniform = (0:1/feather_fs:max(t_feather))';
num_samples = length(t_uniform);

% Get the feature names (starting from the third column)
feather_features = feather_data.Properties.VariableNames;
feather_features = feather_features(3:end);  % exclude first two fields (e.g., 'delta', etc.)

% Resample each feather feature
feather_resampled_mat = zeros(num_samples, length(feather_features));
for i = 1:length(feather_features)
    signal = feather_data.(feather_features{i});
    
    % Handle "True"/"False" values if present
    if iscell(signal) || isstring(signal)
        signal = string(signal);
        if all(signal == "True" | signal == "False")
            % Convert to numeric (True => 1, False => 0)
            signal = double(signal == "True");
        else
            % Fallback: if non-numeric, use nearest neighbor interpolation
            warning("Non-numeric field '%s': using nearest neighbor", feather_features{i});
            [~, idx_nearest] = min(abs(t_feather - t_uniform'), [], 1);
            feather_resampled_mat(:, i) = signal(idx_nearest);
            continue;
        end
    end
    
    % If signal is logical, convert it to double
    if islogical(signal)
        signal = double(signal);
    end
    
    % Interpolate to uniform time vector
    feather_resampled_mat(:, i) = interp1(t_feather, signal, t_uniform, 'linear', 'extrap');
end

% Convert resampled matrix into a table with appropriate headers
feather_resampled = array2table(feather_resampled_mat, 'VariableNames', feather_features);

%% 3. Compute EMG envelope (for first 8 channels) and simply downsample the remaining OTB channels
% (Assumes that otb_data has all numeric channels.)
otb_features = otb_data.Properties.VariableNames;
num_otb_channels = width(otb_data);

env_subsampled = []; % This will hold all downsampled OTB channels (200 Hz)

% Process first 8 channels: compute the envelope (rectify then exponential filter)
for i = 1:8
    % Get signal from channel i and rectify it
    signal = table2array(otb_data(:, i));
    signal = abs(signal);
    
    % Compute exponential envelope filter
    envelope = zeros(size(signal));
    envelope(1) = signal(1);
    for j = 2:length(signal)
        envelope(j) = alpha_env * signal(j) + (1 - alpha_env) * envelope(j-1);
    end
    
    % Downsample the envelope from otb_fs to feather_fs
    subsampled = envelope(1:otb_fs/feather_fs:end);
    env_subsampled(:, i) = subsampled;
end

% Process remaining channels (e.g., channel 9 might be the trigger signal):
for i = 9:num_otb_channels
    signal = table2array(otb_data(:, i));
    % Downsample without envelope computation
    subsampled = signal(1:otb_fs/feather_fs:end);
    env_subsampled(:, i) = subsampled;
end

% Create a time vector for env_subsampled (sampled at 200 Hz)
t_env = (0:size(env_subsampled,1)-1)' / feather_fs;

%% 4. Trigger detection and alignment between feather_resampled and env_subsampled
% 4a. Detect trigger in feather_resampled: field 'trigger_otb'
if ~ismember('trigger_otb', feather_resampled.Properties.VariableNames)
    error('Field "trigger_otb" not found in feather_resampled.');
end
trigger_feather = feather_resampled.trigger_otb;  % should be a column vector
% Detect the rising edge: the first sample where the difference is positive.
idx_trigger_feather = find(diff([0; trigger_feather]) > 0, 1, 'first');
if isempty(idx_trigger_feather)
    error('No trigger detected in feather_resampled.');
end

% 4b. Detect trigger in env_subsampled: using its Channel9 (assumed to be the trigger voltage)
if size(env_subsampled, 2) < 9
    error('env_subsampled does not have a Channel9 for trigger detection.');
end
trigger_otb_signal = env_subsampled(:, 9);  
idx_trigger_otb = find(trigger_otb_signal < 3e6, 1, 'first'); % first sample where value drops below 3e6
if isempty(idx_trigger_otb)
    error('No trigger detected in env_subsampled (Channel9).');
end

% 4c. Compute the delta (in samples) between the two triggers
delta_samples = idx_trigger_feather - idx_trigger_otb;
fprintf('Delta in samples between triggers (feather - otb): %d\n', delta_samples);

% 4d. Align the datasets using the delta:
% If delta > 0, feather_resampled started earlier, so clip its beginning by abs(delta) samples.
% If delta < 0, env_subsampled started earlier, so clip its beginning by delta samples.
if delta_samples < 0
    % env_subsampled started earlier: clip its beginning
    new_start_env = abs(delta_samples);
    new_start_feather = 1;
elseif delta_samples > 0
    % feather_resampled started earlier: clip its beginning
    new_start_feather = delta_samples;
    new_start_env = 1;
end

% Determine the maximum common length from the new start points
num_samples_feather = height(feather_resampled) - new_start_feather + 1;
num_samples_env = size(env_subsampled, 1) - new_start_env + 1;
N = min(num_samples_feather, num_samples_env);

feather_resampled_aligned = feather_resampled(new_start_feather : new_start_feather + N - 1, :);
env_subsampled_aligned = env_subsampled(new_start_env : new_start_env + N - 1, :);

% Create an aligned time vector (in seconds)
t_aligned = (0:N-1)' / feather_fs;

% Optional: plot both triggers for verification
figure;
subplot(3,1,1);
plot(t_aligned, feather_resampled_aligned.pronosupi);
title(' Feather data (Aligned)');
xlabel('Time (s)'); ylabel('Angle [deg]');

subplot(3,1,2);
plot(t_aligned, env_subsampled_aligned(:, 1));
title('OTB (Channel1, Aligned)');
xlabel('Time (s)'); ylabel('Voltage');

subplot(3,1,3);
plot(t_aligned, env_subsampled_aligned(:, 2));
title('OTB (Channel2, Aligned)');
xlabel('Time (s)'); ylabel('Voltage');
ylim([0 5*1e3]);
