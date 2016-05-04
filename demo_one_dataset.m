clc; clear all; close all;
addpath(genpath('.'));


%% Specify the file will be used
train_images_file = 'train-volume.tif';
train_labels_file = 'train-labels.tif';

%% Read the data onto 3D array
images = imreadtif(train_images_file);
labels = imreadtif(train_labels_file);

%% Scale to range 0 and 1
S0 = im2single(images)/255;
L0 = im2single(labels)/255;

%% Extract the dictionary with resolution 7x7
% Seed the randomness
rng(2016);

aSize = 7;
kSize = 64;
ySize = 512;
xSize = 512;
iSize = size(S0, 3);

% Initialize the plan
plan.elemSize = [ySize, xSize, 1, iSize];
plan.dataSize = [ySize, xSize, 1, iSize];
plan.atomSize = [aSize, aSize, 1, kSize];
plan.dictSize = [aSize, aSize, 1, kSize];
plan.blobSize = [ySize, xSize, 1, kSize];
plan.iterSize = [ySize, xSize, 1, kSize];

plan.alpha  = params; % See param.m
plan.gamma  = params;
plan.delta  = params;
plan.theta  = params;
plan.omega  = params;
plan.lambda = params;
plan.sigma  = params;
plan.rho    = params;

plan.lambda.Value       = 0.001; %10; 1; 0.1; 0.01; 0.001;
plan.weight             = 1;
plan.sigma.Value        = 0.01;
plan.rho.Value          = 0.01;
plan.sigma.AutoScaling  = 0;
plan.rho.AutoScaling    = 0;

plan.Verbose 			= 1;
plan.MaxIter 			= 50;
plan.AbsStopTol 		= 1e-6;
plan.RelStopTol 		= 1e-6;

% Initialize the dictionary
D0 = rand(plan.dictSize,'single');
S0 = reshape(S0, plan.dataSize);

% Run the CSC algorithm
isTrainingDictionary=1;
[resD] = ecsc_gpu(D0, S0, plan, isTrainingDictionary);



