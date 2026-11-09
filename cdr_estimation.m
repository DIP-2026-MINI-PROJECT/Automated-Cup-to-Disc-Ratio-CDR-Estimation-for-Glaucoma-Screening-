
clc;
clear;
close all;

%% =========================================================
% Automated Cup-to-Disc Ratio (CDR) Estimation
% Drishti-GS Dataset - 10 Test Images
%
% Testing:
% 5 Glaucoma Images
% 5 Normal Images
%
% Pipeline:
% Sampling -> RGB -> Otsu -> Connectivity -> Retina Mask
% -> DFT Illumination Correction -> SVD Reconstruction
% -> SVD Error -> CLAHE -> Sobel -> Vessel Mask
% -> Vessel Removal -> Disc Segmentation -> Cup Segmentation
% -> Morphology -> CDR Estimation -> Classification
%% =========================================================

%% ---------- SELECT 10 TEST IMAGES ----------

glaucomaDir = fullfile('Test','Images','glaucoma');
normalDir   = fullfile('Test','Images','normal');

gFiles = dir(fullfile(glaucomaDir,'*.png'));
nFiles = dir(fullfile(normalDir,'*.png'));

numEach = 5;

if length(gFiles) < numEach || length(nFiles) < numEach
    error('Not enough images in glaucoma or normal folders.');
end

% Select 5 images from each category
selectedFiles = [gFiles(1:numEach); nFiles(1:numEach)];

imageNames = cell(10,1);
imagePaths = cell(10,1);
imageCategories = cell(10,1);

for i = 1:10

    imageNames{i} = erase(selectedFiles(i).name,'.png');

    if i <= 5
        imagePaths{i} = fullfile(glaucomaDir,selectedFiles(i).name);
        imageCategories{i} = 'Glaucoma';
    else
        imagePaths{i} = fullfile(normalDir,selectedFiles(i).name);
        imageCategories{i} = 'Normal';
    end

end

%% ---------- OTSU THRESHOLD FUNCTION ----------

otsuIn = @(v) otsuthresh(histcounts(v,256,'BinLimits',[0 1]));

%% ---------- RESULT STORAGE ----------

resultTruth = repmat({'-'},10,1);
resultPred  = repmat({'-'},10,1);
resultCDR   = nan(10,1);

%% =========================================================
% PROCESS ALL 10 IMAGES
%% =========================================================

for n = 1:10

    name = imageNames{n};

    %% ---------- ACQUISITION ----------

    img = imread(imagePaths{n});
    category = imageCategories{n};

    if size(img,3) == 1
        img = repmat(img,[1 1 3]);
    end

    img = im2uint8(img);

    [rows,cols,channels] = size(img);

    figure('Name',[name ' - Chained Flow'],...
        'NumberTitle','off',...
        'Position',[20 20 1800 1000]);

    subplot(5,5,1);
    imshow(img);
    title('1. Original');

    %% ---------- UNIT 1: SAMPLING ----------

    sm = imresize(img,0.25);

    subplot(5,5,2);
    imshow(sm);
    title(sprintf('2. Sampling 0.25 (%dx%d)',...
        size(sm,1),size(sm,2)));

    % Quantization
    levels = 16;
    step = 256/levels;

    quantized = uint8(floor(double(sm)/step)*step);

    subplot(5,5,3);
    imshow(quantized);
    title('3. 16-Level Quantization');

    %% ---------- UNIT 1: RGB CHANNELS ----------

    R = sm(:,:,1);
    G = sm(:,:,2);
    B = sm(:,:,3);

    subplot(5,5,4);
    imshow(R);
    title('4. Red Channel');

    subplot(5,5,5);
    imshow(G);
    title('5. Green Channel');

    subplot(5,5,6);
    imshow(B);
    title('6. Blue Channel');

    %% ---------- OTSU + CONNECTIVITY + BOUNDARY ----------

    T = graythresh(R);
    BW = imbinarize(R,T);

    subplot(5,5,7);
    imshow(BW);
    title('7. Otsu (R)');

    [Labeled,num] = bwlabel(BW,8);

    if num > 0

        st = regionprops(Labeled,'Area');
        [~,idx] = max([st.Area]);

        otsuLargest = imfill(Labeled == idx,'holes');

    else

        otsuLargest = false(size(BW));

    end

    % Retina mask
    fov = max(sm,[],3) > 0.05*255;

    fov = imopen(fov,strel('disk',3));

    [Labeled2,num2] = bwlabel(fov,8);

    if num2 > 0

        st2 = regionprops(Labeled2,'Area');
        [~,idx2] = max([st2.Area]);

        fov = Labeled2 == idx2;

    end

    largestRegion = bwconvhull(imfill(fov,'holes'));

    boundary = bwperim(largestRegion);

    ovB = sm;

    rr = ovB(:,:,1);
    gg = ovB(:,:,2);
    bb = ovB(:,:,3);

    rr(boundary) = 0;
    gg(boundary) = 255;
    bb(boundary) = 0;

    ovB(:,:,1) = rr;
    ovB(:,:,2) = gg;
    ovB(:,:,3) = bb;

    subplot(5,5,8);
    imshow(ovB);
    title('8. Retina + Boundary');

    retina = imerode(largestRegion,strel('disk',10));

    %% ---------- DFT: ILLUMINATION CORRECTION ----------

    gray = im2double(rgb2gray(sm));

    grayFill = gray;
    grayFill(~largestRegion) = mean(gray(largestRegion));

    F = fft2(grayFill);
    Fshift = fftshift(F);

    spectrum = log(1 + abs(Fshift));

    subplot(5,5,9);
    imshow(spectrum,[]);
    title('9. DFT Magnitude Spectrum');

    [M,N] = size(gray);

    [U,V] = meshgrid((-floor(N/2)):(ceil(N/2)-1),...
                     (-floor(M/2)):(ceil(M/2)-1));

    D = sqrt(U.^2 + V.^2);

    sigma = 1.0;

    H = exp(-(D.^2)/(2*sigma^2));

    illumination = real(ifft2(ifftshift(Fshift.*H)));

    corrected = grayFill - illumination + mean(illumination(:));

    corrected = mat2gray(corrected);

    corrected(~largestRegion) = mean(corrected(largestRegion));

    subplot(5,5,10);
    imshow(corrected);
    title('10. DFT Illumination Correction');

    %% ---------- SVD RECONSTRUCTION + ERROR ----------

    A = corrected;

    [Us,S,Vs] = svd(A,'econ');

    k = min(20,size(S,1));

    reconstructed = Us(:,1:k)*S(1:k,1:k)*Vs(:,1:k)';

    reconstructed = min(max(reconstructed,0),1);

    errorImage = abs(A - reconstructed);

    subplot(5,5,11);
    imshow(reconstructed);
    title(['11. SVD Reconstruction (k=',num2str(k),')']);

    subplot(5,5,12);
    imshow(errorImage,[]);
    title('12. SVD Error Image');

    mseValue = mean((A(:)-reconstructed(:)).^2);

    psnrValue = 10*log10(1/max(mseValue,eps));

    %% ---------- UNIT 3: SVD ERROR PROCESSING ----------

    darkResid = max(reconstructed - A,0).*retina;

    errN = mat2gray(darkResid);

    % Low-pass filtering
    errLP = imgaussfilt(errN,1.5);

    subplot(5,5,13);
    imshow(errLP,[]);
    title('13. Low-Pass on Error');

    % CLAHE
    errCL = adapthisteq(mat2gray(errLP),...
        'ClipLimit',0.005,'NumTiles',[4 4]);

    subplot(5,5,14);
    imshow(errCL);
    title('14. CLAHE on Error');

    % Sobel
    errSobel = imgradient(errCL,'sobel');

    subplot(5,5,15);
    imshow(errSobel,[]);
    title('15. Sobel on Error');

    % Vessel threshold
    vSorted = sort(errCL(retina));

    tV = vSorted(max(1,round(0.92*numel(vSorted))));

    vesselMask = (errCL > tV) & retina;

    vesselMask = bwareaopen(vesselMask,40);

    vesselMaskD = imdilate(vesselMask,strel('disk',1));

    subplot(5,5,16);
    imshow(vesselMask);
    title('16. Vessel Mask');

    % Vessel removal
    Anv = regionfill(corrected,vesselMaskD);

    subplot(5,5,17);
    imshow(Anv);
    title('17. Vessels Removed');

    %% ---------- UNIT 4: DISC SEGMENTATION ----------

    Ae = imgaussfilt(adapthisteq(mat2gray(Anv),...
        'ClipLimit',0.02),3);

    dRet = 2*sqrt(nnz(retina)/pi);

    padR = padarray(retina,[1 1],0);

    distEdge = bwdist(~padR);

    distEdge = distEdge(2:end-1,2:end-1);

    allowed = distEdge > 0.15*dRet;

    if ~any(allowed(:))
        allowed = retina;
    end

    Ab = imgaussfilt(Anv,10);

    Ab(~allowed) = -Inf;

    [~,cIdx] = max(Ab(:));

    [cy,cx] = ind2sub(size(Ab),cIdx);

    % Local window
    [Xg,Yg] = meshgrid(1:size(Ae,2),1:size(Ae,1));

    distC = hypot(Xg-cx,Yg-cy);

    Rw = round(0.20*dRet);

    win = (distC <= Rw) & retina;

    % Otsu disc threshold
    tDisc = otsuIn(Ae(win));

    discRaw = (Ae > tDisc) & win;

    seM = strel('disk',7);

    discEro = imerode(discRaw,seM);

    discDil = imdilate(discRaw,seM);

    discOpen = imopen(discRaw,seM);

    discClose = imclose(discOpen,strel('disk',10));

    % Keep component at detected centre
    Ldisc = bwlabel(discClose,8);

    disc = false(size(discClose));

    discSolidity = 0;

    if max(Ldisc(:)) > 0

        lab = Ldisc((distC <= 0.35*Rw) & (Ldisc > 0));

        if ~isempty(lab)

            discPre = imfill(Ldisc == mode(double(lab)),'holes');

            sp = regionprops(discPre,'Solidity');

            discSolidity = sp(1).Solidity;

            disc = bwconvhull(discPre);

        end

    end

    subplot(5,5,18);
    imshow(discRaw);
    title('18. Raw Disc Mask');

    subplot(5,5,19);
    imshow(discEro);
    title('19. Erosion');

    subplot(5,5,20);
    imshow(discDil);
    title('20. Dilation');

    subplot(5,5,21);
    imshow(discOpen);
    title('21. Opening');

    subplot(5,5,22);
    imshow(discClose);
    title('22. Closing');

    %% ---------- UNIT 4: CUP SEGMENTATION ----------

    Gd = im2double(G);

    Gd(~largestRegion) = mean(Gd(largestRegion));

    Gd = regionfill(Gd,vesselMaskD);

    Gcup = imgaussfilt(adapthisteq(Gd,...
        'ClipLimit',0.02),2);

    if nnz(disc) > 0

        [tCup,sepEM] = otsuthresh(histcounts(...
            Gcup(disc),256,'BinLimits',[0 1]));

        cupRaw = (Gcup > tCup) & disc;

    else

        sepEM = NaN;

        cupRaw = false(size(disc));

    end

    cup = imopen(cupRaw,strel('disk',3));

    cup = imclose(cup,strel('disk',6));

    cup = imfill(bwareafilt(cup,1),'holes') & disc;

    if nnz(cup) > 0

        cup = bwconvhull(cup) & disc;

    end

    subplot(5,5,23);
    imshow(cup);
    title('23. Cleaned Cup');

    %% ---------- PSEUDOCOLOR RESULT ----------

    ov = im2double(sm);

    alpha = 0.45;

    discOnly = disc & ~cup;

    discColor = [0 1 0];

    cupColor = [1 0 0];

    for c = 1:3

        ch = ov(:,:,c);

        ch(discOnly) = (1-alpha)*ch(discOnly) ...
            + alpha*discColor(c);

        ch(cup) = (1-alpha)*ch(cup) ...
            + alpha*cupColor(c);

        ov(:,:,c) = ch;

    end

    %% ---------- CDR CALCULATION ----------

    dS = regionprops(disc,'BoundingBox');

    cS = regionprops(cup,'BoundingBox');

    if ~isempty(dS) && ~isempty(cS)

        CDR = cS(1).BoundingBox(4)/dS(1).BoundingBox(4);

    else

        CDR = NaN;

    end

    cdrArea = nnz(cup)/max(nnz(disc),1);

    subplot(5,5,24);

    imshow(ov);
    hold on;

    plot(cx,cy,'c+','MarkerSize',8,'LineWidth',1.2);

    hold off;

    title(sprintf('24. Disc + Cup | CDR = %.2f',CDR));

    %% ---------- CLASSIFICATION ----------

    if isnan(CDR)

        predLabel = 'Undetermined';

    elseif CDR >= 0.6

        predLabel = 'Glaucoma';

    elseif CDR >= 0.5

        predLabel = 'Borderline';

    else

        predLabel = 'Normal';

    end

    % Segmentation sanity check
    discFrac = nnz(disc)/max(nnz(retina),1);

    segOK = discFrac > 0.01 && discFrac < 0.12 ...
        && discSolidity > 0.75 && nnz(cup) > 0;

    if segOK

        segNote = 'OK';

    else

        segNote = 'CHECK (unreliable)';

    end

    %% ---------- STORE RESULTS ----------

    resultTruth{n} = category;

    resultPred{n} = predLabel;

    resultCDR(n) = CDR;

    %% ---------- IMAGE INFORMATION PANEL ----------

    subplot(5,5,25);

    axis off;

    info = {

        ['Image: ' name]

        ['Category: ' category]

        ['Original Size: ' num2str(rows) ' x ' num2str(cols)]

        ['Otsu Threshold: ' num2str(T,'%.3f')]

        ['Connected Regions: ' num2str(num)]

        ['Largest Area: ' num2str(round(nnz(otsuLargest)/0.25^2)) ' pixels']

        ['CDR (vertical): ' num2str(CDR,'%.2f')]

        ['CDR Area: ' num2str(cdrArea,'%.2f')]

        ['Segmentation: ' segNote]

        ['CDR Label: ' predLabel]

    };

    text(0,1,info,'FontSize',9,...
        'VerticalAlignment','top',...
        'Interpreter','none',...
        'Units','normalized');

    %% ---------- COMMAND WINDOW OUTPUT ----------

    fprintf('\n====================================\n');

    fprintf('IMAGE: %s (%s)\n',name,category);

    fprintf('Original Size: %d x %d x %d\n',...
        rows,cols,channels);

    fprintf('Working Size: %d x %d\n',...
        size(sm,1),size(sm,2));

    fprintf('Otsu Threshold (R): %.4f | Regions: %d\n',...
        T,num);

    fprintf('SVD k=%d | MSE %.6f | PSNR %.2f dB\n',...
        k,mseValue,psnrValue);

    fprintf('Vessel Pixels Removed: %d\n',nnz(vesselMaskD));

    fprintf('Disc Area: %d px | Cup Area: %d px\n',...
        nnz(disc),nnz(cup));

    fprintf('CDR (vertical): %.3f | Area: %.3f\n',...
        CDR,cdrArea);

    fprintf('Cup/rim separability: %.2f\n',sepEM);

    fprintf('Label (folder): %s | Label (CDR): %s\n',...
        category,predLabel);

    fprintf('Segmentation: %s\n',segNote);

    fprintf('====================================\n');

end

%% =========================================================
% FINAL SUMMARY TABLE
%% =========================================================

summaryTable = table(imageNames,resultTruth,...
    resultCDR,resultPred,...
    'VariableNames',{'Image','Category','CDR','CDR_Label'});

disp(' ');
disp('========== FINAL RESULTS FOR 10 IMAGES ==========');

disp(summaryTable);

%% ---------- CLASSIFICATION SUMMARY ----------

correct = strcmp(resultTruth,resultPred);

% Exclude borderline and undetermined predictions
valid = ~strcmp(resultPred,'Borderline') & ...
        ~strcmp(resultPred,'Undetermined');

if any(valid)

    accuracy = sum(correct(valid))/sum(valid)*100;

    fprintf('\nClassification Accuracy: %.2f%%\n',accuracy);

    fprintf('Evaluated Images: %d out of 10\n',sum(valid));

else

    fprintf('\nNo valid predictions for accuracy calculation.\n');

end

fprintf('\n10-IMAGE TESTING COMPLETED.\n');
