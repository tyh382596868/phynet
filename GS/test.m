%%
clc
clear

%%
scale = 'pi\'
numcore = '100'
pha_path = ['D:\tyh\IterativePhaseRetrieval\simulateData\simulate_data\' scale numcore '\' numcore '_pha_simulate.txt'];
amp_path = ['D:\tyh\IterativePhaseRetrieval\simulateData\simulate_data\' scale numcore '\' numcore '_amp_simulate.txt'];

save_path = ['.\' scale numcore '\']
mkdir(save_path)
%% Load image


img = readmatrix(pha_path, 'Delimiter', ',');

% img = imresize(img,[1920 1920]);
img_convert = double((img));

% img_convert = padarray(img_convert, [640, 640], 'both');

img_phase = (img_convert*1);
figure; 
set(gcf, 'Units','pixels','Position',[0,0,3000,3000]); % 设置figure的单位为英寸

imagesc(img_phase);axis image off;saveas(gcf, [save_path 'gt_pha.png'],'png'); % 保存图像为PNG文件
%% amp
amp = readmatrix(amp_path, 'Delimiter', ',');

% amp = imresize(amp,[1920 1920]);
amp_convert = double((amp));

% amp_convert = padarray(amp_convert, [640, 640], 'both');

amp_phase = (amp_convert*1);
figure; 
set(gcf, 'Units','pixels','Position',[0,0,3000,3000]); % 设置figure的单位为英寸
imagesc(amp_phase);axis image off;saveas(gcf, [save_path 'gt_amp.png']); % 保存图像为PNG文件

%% mask
mask = imbinarize(img_phase,0.02);
mask = imclose(mask,strel('disk',10));
figure;imagesc(mask);axis image off;colormap gray;title('Binary mask');saveas(gcf, [save_path 'mask.png']); % 保存图像为PNG文件
%% Parameters
dx = 2e-6;
dy = 2e-6;
lambda = 532e-9;
z = 0.0788;

%% Propagation (仿真散斑)
U = amp_phase.*exp(i*img_phase);
% exp(i*img_phase);
% 
U_prop = prop(U,dx,dx,lambda,z);

figure; set(gcf, 'Units','pixels','Position',[0,0,3000,3000]); imagesc(abs(U_prop));axis image off; colorbar;saveas(gcf, [save_path 'speckle.png']); % 保存图像为PNG文件
figure; imagesc(wrapTo2Pi(angle(U_prop)));axis image off; colorbar

%% reconstruct the phase on the facet with sample from the diffraction pattern



spec = abs(U_prop)
spec_py = readmatrix('D:\tyh\IterativePhaseRetrieval\simulateData\simulate_data\pi\100\100_speckle_prop00788_simulate.txt', 'Delimiter', ',');

diff = spec-spec_py;
figure; imagesc(diff);axis image off; colorbar



