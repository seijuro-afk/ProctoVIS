%% Phone Light Detection Using Face Brightness
clc;
clear;
close all;

%% Read images
ref = imread('reference.jpg');
test = imread('test.jpg');

% Resize if needed
if ~isequal(size(ref),size(test))
    test = imresize(test,[size(ref,1) size(ref,2)]);
end

% Downsample for faster face checks
resizeScale = 0.6;
ref = imresize(ref, resizeScale);
test = imresize(test, resizeScale);

%% Convert to grayscale
grayRef = rgb2gray(ref);
grayTest = rgb2gray(test);

%% Detect faces
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');

bboxRefAll = step(faceDetector,ref);
bboxTestAll = step(faceDetector,test);

if isempty(bboxRefAll)
    error('No face found in reference image.');
end

if isempty(bboxTestAll)
    error('No face found in test image.');
end

% Use largest face
[~,idx] = max(bboxRefAll(:,3).*bboxRefAll(:,4));
bboxRef = bboxRefAll(idx,:);

[~,idx] = max(bboxTestAll(:,3).*bboxTestAll(:,4));
bboxTest = bboxTestAll(idx,:);

% Heuristic for people visible in the background
mainCenter = [bboxTest(1)+bboxTest(3)/2, bboxTest(2)+bboxTest(4)/2];
faceCenters = [bboxTestAll(:,1)+bboxTestAll(:,3)/2, bboxTestAll(:,2)+bboxTestAll(:,4)/2];
distances = hypot(faceCenters(:,1)-mainCenter(1), faceCenters(:,2)-mainCenter(2));
otherFaces = distances > max(bboxTest(3), bboxTest(4)) * 0.75;
peopleInBackground = sum(otherFaces) > 0 || size(bboxTestAll,1) > 1;

%% Extract face regions
faceRef = imcrop(grayRef,bboxRef);
faceTest = imcrop(grayTest,bboxTest);

%% Average face brightness
faceBrightnessRef = mean(faceRef(:));
faceBrightnessTest = mean(faceTest(:));

%% Compute background brightness
mask = true(size(grayRef));

x = round(bboxTest(1));
y = round(bboxTest(2));
w = round(bboxTest(3));
h = round(bboxTest(4));

mask(y:y+h-1,x:x+w-1) = false;

backgroundRef = grayRef(mask);
backgroundTest = grayTest(mask);

backgroundBrightnessRef = mean(backgroundRef);
backgroundBrightnessTest = mean(backgroundTest);

%% Brightness increase
faceIncrease = faceBrightnessTest - faceBrightnessRef;
backgroundIncrease = backgroundBrightnessTest - backgroundBrightnessRef;

%% Decision
faceThreshold = 12;
backgroundThreshold = 5;

phoneLight = (faceIncrease > faceThreshold) && ...
             (backgroundIncrease < backgroundThreshold);

%% Display results
figure;

subplot(2,2,1)
imshow(ref)
title('Reference')

subplot(2,2,2)
imshow(test)
title('Test')

subplot(2,2,3)
imshow(insertShape(ref,'Rectangle',bboxRef,'Color','green','LineWidth',3))
title('Reference Face')

subplot(2,2,4)
imshow(insertShape(test,'Rectangle',bboxTest,'Color','green','LineWidth',3))
title('Test Face')

fprintf("\n-----------------------------\n");
fprintf("Reference Face Brightness : %.2f\n",faceBrightnessRef);
fprintf("Test Face Brightness      : %.2f\n",faceBrightnessTest);
fprintf("Face Increase             : %.2f\n\n",faceIncrease);

fprintf("Reference Background      : %.2f\n",backgroundBrightnessRef);
fprintf("Test Background           : %.2f\n",backgroundBrightnessTest);
fprintf("Background Increase       : %.2f\n",backgroundIncrease);

if phoneLight
    fprintf("\n>>> POSSIBLE PHONE LIGHT DETECTED <<<\n");
else
    fprintf("\nNo significant phone light detected.\n");
end

if peopleInBackground
    fprintf("People appear to be visible in the background.\n");
else
    fprintf("No additional people were detected in the background.\n");
end