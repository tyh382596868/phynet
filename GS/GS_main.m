%%
clc
clear
%% Load image


img = readmatrix('D:\tyh\simulateData\simulate_data\strong\2pi\1000\0.6\2\1000_pha_simulate.txt', 'Delimiter', ',');

% img = imresize(img,[1920 1920]);
img_convert = double((img));

% img_convert = padarray(img_convert, [640, 640], 'both');

img_phase = (img_convert*1);
figure; imagesc(img_phase);axis image off;
%% amp
amp = readmatrix('D:\tyh\simulateData\simulate_data\strong\2pi\1000\0.6\2\1000_amp_simulate.txt', 'Delimiter', ',');

% amp = imresize(amp,[1920 1920]);
amp_convert = double((amp));

% amp_convert = padarray(amp_convert, [640, 640], 'both');

amp_phase = (amp_convert*1);
figure; imagesc(amp_phase);axis image off;

%% mask
% mask = imbinarize(img_phase,0.02);
% mask = imclose(mask,strel('disk',10));
% figure;imagesc(mask);axis image off;colormap gray;title('Binary mask');
%% Parameters
dx = 2e-6;
dy = 2e-6;
lambda = 532e-9;
z = 0.05;

%% Propagation (仿真散斑)
U = amp_phase.*exp(i*img_phase);
% exp(i*img_phase);
% 
U_prop = prop(U,dx,dx,lambda,z);

figure; imagesc(abs(U_prop));axis image off; colorbar
% figure; imagesc(wrapTo2Pi(angle(U_prop)));axis image off; colorbar

%% reconstruct the phase on the facet with sample from the diffraction pattern
% distance = -0.005; %propagation distance MIND the DIRECTION!

% target_intensity = abs(test_holo);
target_intensity = amp_phase;
% ones(size(abs(U_prop)));
% 
% ones(size(abs(U_prop)));

rand_phi = zeros(size(abs(U_prop)));

source_intensity = abs(U_prop);

% mask = imbinarize(gather(abs(com_prop_cal)),3); %3 %0.02
% mask = imclose(mask,strel('disk',10)); %25
% figure,imshow(mask);
% source_intensity = mask;

source = source_intensity.*exp(i*rand_phi);

phase_error = [];
% amp_cc = [];
% for x = 1:5
for i=1:10000
    fprintf('Iteration %i\n', i);
    
    % use original target intensity and apply phase from forward propagation
    % of the source wavefront from last iteration
    source = source_intensity .*exp(1i*angle(source));
    
    % propagate target wave front back to source plane
    target = prop(source,dx,dy,lambda,-z);
    angle_target = angle(target);
    phase_diff= wrapTo2Pi(angle_target-img_phase);

        
    % use original input intensity and apply phase from back propagation
    target = target_intensity .* exp(1i*angle_target);
%     source = source.*prop_mask;
    
    % propagate source intensity forwards to the target plane
    source = prop(target,dx,dy,lambda,z);
    
    % amp_cc(i) = corr2(source_intensity,abs(source));  
    
    source_phase_full = angle(source);   % store the calculated phase
    phase_error(i) = mean(abs((angle_target-img_phase)), 'all');


    figure(2364);
    subplot(2,2,1);imagesc(wrapTo2Pi(angle_target));axis image;axis off;title('phase reconstruction');
    subplot(2,2,2);imagesc(phase_diff);axis image;axis off;title('phase difference');
    % subplot(2,2,3);plot(amp_cc);title('Amplitude cc');
    subplot(2,2,4);plot(phase_error);title('Phase Error');
    pause(0.02);
    
end

% figure(3958);plot(phase_error);title('error');

% figure;polarhistogram(wrapTo2Pi(phase_diff_core), 40); rlim([0,1000]);

% plotting
% figure(2633);
% subplot(2,1,1);
% imshow(target_intensity);axis image;colormap gray;
% title('Original image')
%
% subplot(2,1,2);
% imagesc(abs(target));axis image;   colormap gray;          %last pattern
% title('reconstructed image');

% figure(2634);
% i = 1:1:i;
% plot(i,(error'));
% title('Error');

% figure(2635);
% imagesc(angle(source)); axis image;
% title('hologram');
% end
% figure; imagesc(wrapTo2Pi(angle(source)-phiProx));
 
% mask = imbinarize(abs(source_intensity),0.02);
% mask = imclose(mask,strel('disk',25));
% phase_err_line = phase_diff(:);
% phase_err_line(~mask(:)) = [];
% phase_error = circ_std(phase_err_line);