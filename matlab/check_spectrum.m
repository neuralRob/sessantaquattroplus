% Example MATLAB script: show_emg_spectral_content.m
% This script assumes that an EMG matrix called "emg" exists in the workspace,
% where each row represents a sample and each column represents a channel.
% The script computes the FFT and plots the one-sided amplitude spectrum for each channel.

% --- Parameters ---
fs = 2000;  % Sampling frequency in Hz (modify as needed)

% --- Check if the variable "emg" exists ---
if ~exist('emg', 'var')
    error('The variable "emg" was not found in the workspace.');
end

[numSamples, numChannels] = size(emg);

% Choose FFT length as the next power of 2 greater than numSamples.
NFFT = 2^nextpow2(numSamples);

% Create frequency vector for one-sided FFT.
f = fs/2 * linspace(0, 1, NFFT/2+1);

% Loop over each channel to compute and plot spectral content.
for ch = 1:numChannels
    % Compute the FFT for channel ch.
    Y = fft(table2array(emg(:, ch)), NFFT);
    
    % Compute the two-sided spectrum and then the one-sided spectrum.
    P2 = abs(Y / NFFT);
    P1 = P2(1:NFFT/2+1);
    P1(2:end-1) = 2 * P1(2:end-1);  % account for energy in negative frequencies
    
    % Convert the amplitude spectrum to decibels.
    P1_dB = 20*log10(P1);
    
    % Plot the spectral content.
    figure;
    plot(f, P1_dB, 'LineWidth', 1.5);
    grid on;
    xlabel('Frequency (Hz)');
    ylabel('Magnitude (dB)');
    title(sprintf('Spectral Content - Channel %d', ch));
end
