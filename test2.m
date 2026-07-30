% detect_and_plot_landmarks.m
% Detect face, crop with padding, run ONNX landmark model, try small variants,
% pick best variant, and plot landmarks on original image.
% Requires: pyenv configured to Python with onnxruntime and numpy.

modelFile = 'models/face_landmarks.onnx';
imgFile = 'selfiw.jpg';
I = imread(imgFile);
origH = size(I,1); origW = size(I,2);

%% 1) Face detection (Viola-Jones)
detector = vision.CascadeObjectDetector(); % FrontalFaceCART
bboxes = detector(I);
if isempty(bboxes)
    error('No face detected. Try a different detector or manual bbox.');
end
% choose largest bbox
[~, idx] = max(bboxes(:,3).*bboxes(:,4));
bbox = double(bboxes(idx, :)); % [x y w h]

%% 2) Expand bbox by padding fraction
pad = 0.18; % 18% padding (adjustable)
x0 = max(1, round(bbox(1) - pad*bbox(3)));
y0 = max(1, round(bbox(2) - pad*bbox(4)));
w0 = min(origW - x0, round(bbox(3) * (1 + 2*pad)));
h0 = min(origH - y0, round(bbox(4) * (1 + 2*pad)));
bbox2 = [x0 y0 w0 h0];

%% 3) Prepare crop and model session
crop = imcrop(I, bbox2);
crop128 = imresize(crop, [128 128]);
X_base = single(crop128); % raw pixels, DO NOT divide by 255 (model normalizes internally)

ort = py.importlib.import_module('onnxruntime');
np  = py.importlib.import_module('numpy');
sess = ort.InferenceSession(modelFile);
pyInputs = sess.get_inputs();
inputName = char(pyInputs{1}.name);

% helper to run model and return MATLAB single vector
function dense = run_model_matlab(Xsingle, sess, inputName, np)
    X_np = np.array(Xsingle, dtype=np.float32);
    X_np = np.expand_dims(X_np, int32(0)); % (1,H,W,C)
    outs = sess.run(py.None, py.dict(pyargs(inputName, X_np)));
    pyOut = outs{1};
    pyList = pyOut.flatten().tolist();
    dense = single(cellfun(@double, cell(pyList)));
end

% helper to map dense->pixel coords in original image given options
function pts_img = map_dense_to_image(dense, bbox_local, inSize, frameSize, swapXY, flipY, assumeNorm)
    pts = reshape(dense, [2, numel(dense)/2])'; % [N x 2] default [x y]
    if swapXY
        pts = pts(:, [2 1]);
    end
    if assumeNorm
        pts(:,1) = pts(:,1) * inSize(2); % to inSize pixels
        pts(:,2) = pts(:,2) * inSize(1);
    end
    % scale from inSize to bbox size and add offset
    scaleX = bbox_local(3) / inSize(2);
    scaleY = bbox_local(4) / inSize(1);
    pts(:,1) = pts(:,1) * scaleX + bbox_local(1);
    pts(:,2) = pts(:,2) * scaleY + bbox_local(2);
    if flipY
        pts(:,2) = bbox_local(2) + bbox_local(4) - (pts(:,2) - bbox_local(2));
    end
    pts_img = pts;
end

%% 4) Try variants (swapXY, flipY) and pick best by in-bbox score
variants = struct();
k = 1;
swapOptions = [false true];
flipOptions = [false true];

% run base model once
dense_base = run_model_matlab(X_base, sess, inputName, np);

for s = swapOptions
    for f = flipOptions
        % determine if outputs are normalized (values <= 1.01)
        assumeNorm = max(dense_base) <= 1.01;
        pts_img = map_dense_to_image(dense_base, bbox2, [128 128], [origH origW], s, f, assumeNorm);
        % score: fraction of points inside padded bbox2 (with small margin)
        margin = 5;
        inside = pts_img(:,1) >= (bbox2(1)-margin) & pts_img(:,1) <= (bbox2(1)+bbox2(3)+margin) & ...
                 pts_img(:,2) >= (bbox2(2)-margin) & pts_img(:,2) <= (bbox2(2)+bbox2(4)+margin);
        score = sum(inside) / size(pts_img,1);
        variants(k).swapXY = s;
        variants(k).flipY = f;
        variants(k).score = score;
        variants(k).pts = pts_img;
        k = k + 1;
    end
end

% choose best variant
scores = [variants.score];
[~, bestIdx] = max(scores);
best = variants(bestIdx);

%% 5) If best score is low, try small bbox expansion and re-run
if best.score < 0.6
    % try larger padding and re-run once
    pad2 = min(0.35, pad + 0.15);
    x0 = max(1, round(bbox(1) - pad2*bbox(3)));
    y0 = max(1, round(bbox(2) - pad2*bbox(4)));
    w0 = min(origW - x0, round(bbox(3) * (1 + 2*pad2)));
    h0 = min(origH - y0, round(bbox(4) * (1 + 2*pad2)));
    bbox3 = [x0 y0 w0 h0];
    crop = imcrop(I, bbox3);
    crop128 = imresize(crop, [128 128]);
    X_base = single(crop128);
    dense_base = run_model_matlab(X_base, sess, inputName, np);
    % re-evaluate variants on bbox3
    variants = struct(); k = 1;
    for s = swapOptions
        for f = flipOptions
            assumeNorm = max(dense_base) <= 1.01;
            pts_img = map_dense_to_image(dense_base, bbox3, [128 128], [origH origW], s, f, assumeNorm);
            margin = 5;
            inside = pts_img(:,1) >= (bbox3(1)-margin) & pts_img(:,1) <= (bbox3(1)+bbox3(3)+margin) & ...
                     pts_img(:,2) >= (bbox3(2)-margin) & pts_img(:,2) <= (bbox3(2)+bbox3(4)+margin);
            score = sum(inside) / size(pts_img,1);
            variants(k).swapXY = s;
            variants(k).flipY = f;
            variants(k).score = score;
            variants(k).pts = pts_img;
            variants(k).bbox = bbox3;
            k = k + 1;
        end
    end
    [~, bestIdx] = max([variants.score]);
    best = variants(bestIdx);
    bbox_used = bbox3;
else
    bbox_used = bbox2;
end

%% 6) Plot best result
pts_final = best.pts;
figure('Name','Mapped landmarks (best variant)');
imshow(I); hold on;
plot(pts_final(:,1), pts_final(:,2), 'y*', 'MarkerSize',6);
for i=1:size(pts_final,1)
    text(pts_final(i,1), pts_final(i,2), sprintf('%d', i), 'Color','cyan','FontSize',8);
end
% draw bbox used
rectangle('Position', bbox_used, 'EdgeColor','g', 'LineWidth',1.5);
title(sprintf('Best variant: swapXY=%d, flipY=%d, score=%.2f', best.swapXY, best.flipY, best.score));
hold off;

%% 7) Console summary
fprintf('Face bbox used: [x y w h] = [%d %d %d %d]\n', bbox_used(1), bbox_used(2), bbox_used(3), bbox_used(4));
fprintf('Best variant: swapXY=%d, flipY=%d, score=%.2f\n', best.swapXY, best.flipY, best.score);
