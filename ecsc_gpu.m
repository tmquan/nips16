function res = ecsc_gpu(D0, S0, plan, isTrainingDictionary)
    %% If we want to train the dictionary
    if isempty(isTrainingDictionary)
        isTrainingDictionary = 1;
    end
    
    %% Parameters extractions
    elemSize = plan.elemSize;
    dataSize = plan.dataSize;
    atomSize = plan.atomSize;
    dictSize = plan.dictSize;
    blobSize = plan.blobSize;

    numAtoms = dictSize(4);
    % plan.elemSize = [128, 128,  1,   1];
    % plan.dataSize = [128, 128,  1, 512]; % For example
    % plan.atomSize = [ 11,  11,  1,   1];
    % plan.dictSize = [ 11,  11,  1, 100];
    % plan.blobSize = [128, 128,  1, 100];
    % plan.iterSize = [128, 128,  1,  16];


    gNx = gpuArray(prod(blobSize));
    gNd = gpuArray(prod(blobSize));


    glambda = gpuArray(plan.lambda.Value);
    grho    = gpuArray(plan.rho.Value);
    gsigma  = gpuArray(plan.sigma.Value);

    %% Operators here
    %% Mean removal and normalisation projections
    Pzmn    = @(x) bsxfun(@minus,   x, mean(mean(mean(x,1),2),3));
    % Pzmn    = @(x) bsxfun(@minus,   x, mean(mean(mean(mean(x,1),2),3),4));
    % Pnrm    = @(x) bsxfun(@rdivide, x, sqrt(sum(sum(sum(sum(x.^2,1),2),3),4)));
    Pnrm    = @(x) bsxfun(@rdivide, x, sqrt(sum(sum(sum(x.^2,1),2),3)));  
    %Pnrm    = @(x) bsxfun(@rdivide, x, (sum(sum(sum(abs(x).^1,1),2),3)));

    %% Projection of filter to full image size and its transpose
    % (zero-pad and crop respectively)
    Pzp     = @(x) zeropad(x, blobSize);
    PzpT    = @(x) bndcrop(x, dictSize);

    %% Projection of dictionary filters onto constraint set
    Pcn     = @(x) Pnrm(Pzp((PzpT(x))));

    %% Memory reservation
    gS0     = gpuArray(S0); 
    %gS0     = reshape(gS0, dataSize);
    gD0     = gpuArray(D0);
    gD0     = Pnrm(gD0);

    grx = gpuArray(Inf);
    gsx = gpuArray(Inf);
    grd = gpuArray(Inf);
    gsd = gpuArray(Inf);
    geprix = gpuArray(0);
    geduax = gpuArray(0);
    geprid = gpuArray(0);
    geduad = gpuArray(0);

    gX      = gpuArray.zeros(blobSize);
    gY      = gpuArray.zeros(blobSize);
    gYprv   = gY;
    gXf     = gpuArray.zeros(blobSize);
    gYf     = gpuArray.zeros(blobSize);

    % gS      = gS0; 
    %gSf     = gpuArray.zeros(dataSize);

    gD      = gpuArray.zeros(blobSize);
    gG      = gpuArray.zeros(blobSize);
    gGprv   = gpuArray.zeros(blobSize);

    gD      = gpuArray.zeros(blobSize);
    gG      = Pzp(gD); % Zero pad the dictionary
    gGprv   = gG;

    gDf     = gpuArray.zeros(blobSize);
    gGf     = gpuArray.zeros(blobSize);

    gU      = gpuArray.zeros(blobSize);
    gH      = gpuArray.zeros(blobSize);

    gGf     = gpuArray.zeros(blobSize);
    gGf     = fft3(gG);
    
    % Temporary buffers
    gGSf    = gpuArray.zeros(blobSize);
    gYSf    = gpuArray.zeros(blobSize);


    %% Set up algorithm parameters and initialise variables
    res  = struct('itstat', [], 'plan', plan);
    %% Main loops
    k = 1;
    tstart = tic;
    while k <= plan.MaxIter && (grx > geprix | gsx > geduax | ...
                                grd > geprid | gsd > geduad),
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Permutation here
       
        for n=randperm(size(gS0, 4)) %1:size(gS0,4)
            gS = gS0(:,:,:,n);
            
            if isTrainingDictionary
                % gS =  gpuArray(imrotate(gS, 360*rand(1,1), 'crop', 'bilinear')) ;
                gS = permute(gS, [randperm(2), 3, 4]);
                % size(gS0)
                r = randi([0 8],1,1);
                switch r
                    case 0
                        gS = (gS);
                    case 1
                        gS = rot90(gS, 1);
                    case 2
                        gS = rot90(gS, 2);
                    case 3
                        gS = rot90(gS, 3);
                    case 4
                        gS = rot90(gS, 4);
                    case 5
                        gS = fliplr(gS);
                    case 6
                        gS = flipud(gS);
                    case 7
                        gS = gS';
                    otherwise
                        gS = gS;
                end
                figure(3); imagesc(gS(:,:,1)); axis equal off; colormap gray; drawnow; 
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% Compute the signal in DFT domain
            gSf  = fft3(gS); 
            %% Extract the atom iteration
            % for iter = 1:numIters
                % chunk = 1:blobSize(4);
                % march = chunk+(iter-1)*iterSize(4); % marching through the dictionary
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                gD      = gD0; %(:,:,:,march);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                gG      = Pzp(gD); % Zero pad the dictionary, PARTIALLY
                gGf     = fft3(gG);
                % size(gGf)
                % size(gSf)
                gGSf    = bsxfun(@times, conj(gGf), gSf); 

                %% Solve X subproblem
                gXf  = solvedbi_sm(gGf, grho, gGSf + grho*fft3(gY-gU)); 
                gX   = ifft3(gXf); 
                gXr  = gX; %relaxation
				
					
                %% Solve Y subproblem
                gY   = shrink(gXr + gU, (glambda/grho)*plan.weight); % Adjust threshold 
				if isTrainingDictionary
					idx = randperm(numAtoms, floor(0.5*numAtoms));
					gY(:,:,:,idx) = 0;
					% idx = randperm(numAtoms);
					% gY(:,:,:,1:numAtoms) = gY(:,:,:,idx);
				end
                % gT   = mean(gY,4);
                % for k=1:numAtoms
                %     gY(:,:,:,k) = gT;
                % end
                % gY(gY<0) = 0;
                % idx = randperm(numAtoms);
                %gY(:,:,:,:) = gY(:,:,:,idx);
                % gY = reshape(gY, [blobSize(1)*blobSize(3)*sqrt(numAtoms), blobSize(2)*blobSize(3)*sqrt(numAtoms)]);
                % gY = histeq(real(gY));
                % gY = reshape(gY, blobSize);

                gYf  = fft3(gY);
                % gYf = reshape(gYf, [blobSize(1)*blobSize(3)*sqrt(numAtoms), blobSize(2)*blobSize(3)*sqrt(numAtoms)]);
                % gYf = histeq(abs(gYf));
                % gYf = reshape(gYf, blobSize);

                % size(gYf)
                % size(gSf)
                % size(bsxfun(@times, conj(gYf), gSf))
                % gYSf = sum(bsxfun(@times, co  nj(gYf), gSf), 4);
                gYSf = (bsxfun(@times, conj(gYf), gSf));
                %% Solve U subproblem
                gU = gU + gXr - gY;
                
                %% Update params 
                gnX = norm(gX(:)); gnY = norm(gY(:)); gnU = norm(gU(:));
                grx = norm(vec(gX - gY))/max(gnX,gnY);
                gsx = norm(vec(gYprv - gY))/gnU;
                geprix = sqrt(gNx)*plan.AbsStopTol/max(gnX,gnY)+plan.RelStopTol;
                geduax = sqrt(gNx)*plan.AbsStopTol/(grho*gnU)+plan.RelStopTol;

                if plan.rho.Auto,
                    if k ~= 1 && mod(k, plan.rho.AutoPeriod) == 0,
                        if plan.rho.AutoScaling,
                            grhomlt = sqrt(grx/gsx);
                            if grhomlt < 1, grhomlt = 1/grhomlt; end
                            if grhomlt > plan.rho.Scaling, grhomlt = gpuArray(plan.rho.Scaling); end
                        else
                            grhomlt = gpuArray(plan.rho.Scaling);
                        end
                        grsf = 1;
                        if grx > plan.rho.RsdlRatio*gsx, grsf = grhomlt; end
                        if gsx > plan.rho.RsdlRatio*grx, grsf = 1/grhomlt; end
                        grho = grsf*grho;
                        gU = gU/grsf;
                    end
                end

                %% Record information
                gYprv = gY;

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if isTrainingDictionary
                    %% Solve D subproblem
                    % size(gYSf)
                    % size(gG)
                    gDf  = solvedbi_sm(gYf, gsigma, gYSf + gsigma*fft3(gG - gH));
                    %gXf  = solvedbi_sm(gGf, grho, gGSf + grho*fft3(gY-gU)); 
                    gD   = ifft3(gDf);
                    gDr  = gD;

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %% Solve G subproblem
                    gG   = Pcn(gDr + gH);
					% idx = randperm(numAtoms, 2);
					% gG(:,:,:,idx) = 0;
                    % gG =  PzpT(gG);
                   
                    % gG = abs(gG);
                    % gG = Pcn(gG);
                    % gG(gG<0) = 0;
                    % gG(:,:,:,:) = gG(:,:,:,idx);
                    % G = gather(gG);
                    % for d=1:size(gG,4)
                    %     G(1:size(D0,1),1:size(D0,2),:,d) = imrotate(G(1:size(D0,1),1:size(D0,2),:,d),360*rand(1,1), 'crop', 'bilinear') ;
                    %     % gG(1:size(D0,1),1:size(D0,2),d) = imrotate(gG(1:size(D0,1),1:size(D0,2),d), 360*rand(1,1), 'crop', 'bilinear') ;
                    %     % gG(1:size(D0,1),1:size(D0,2),d) = gG(1:size(D0,1),1:size(D0,2),d)';
                    %     % gG(1:size(D0,1),1:size(D0,2),d) = imrotate(gG(1:size(D0,1),1:size(D0,2),d), 90*randi(3), 'crop') ;
                    % end
                    % gG = gpuArray(G);
                    % gG = Pcn(gG);


                    %% Solve H subproblem
                    gH = gH + gDr - gG;

                    %% Update params    
                    gnD = norm(gD(:)); gnG = norm(gG(:)); gnH = norm(gH(:));
                    grd = norm(vec(gD - gG))/max(gnD,gnG);
                    gsd = norm(vec(gGprv - gG))/gnH;
                    geprid = sqrt(gNd)*plan.AbsStopTol/max(gnD,gnG)+plan.RelStopTol;
                    geduad = sqrt(gNd)*plan.AbsStopTol/(gsigma*gnH)+plan.RelStopTol;
                    
                    if plan.sigma.Auto,
                        if k ~= 1 && mod(k, plan.sigma.AutoPeriod) == 0,
                            if plan.sigma.AutoScaling,
                                gsigmlt = sqrt(grd/gsd);
                                if gsigmlt < 1, gsigmlt = 1/gsigmlt; end
                                if gsigmlt > plan.sigma.Scaling, gsigmlt = gpuArray(plan.sigma.Scaling); end
                            else
                                gsigmlt = gpuArray(plan.sigma.Scaling);
                            end
                            gssf = gpuArray(1);
                            if grd > plan.sigma.RsdlRatio*gsd, gssf = gsigmlt; end
                            if gsd > plan.sigma.RsdlRatio*grd, gssf = 1/gsigmlt; end
                            gsigma = gssf*gsigma;
                            gH = gH/gssf;
                        end
                    end
                    %% Record information
                    gGprv = gG;
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %% Collect information
                % Compute l1 norm of Y
                gJl1 = sum(abs(vec( gY)));
                % Compute measure of D constraint violation

                if isTrainingDictionary
                   gJcn = norm(vec(Pcn(gD) - gD));
                end
                % Compute data fidelity term in Fourier domain (note normalisation)
                gJdf = sum(vec(abs(sum(bsxfun(@times,gGf,gYf),4)-gSf).^2))/(2*prod(blobSize));
                gJfn = gJdf + glambda*gJl1;
				% k
				fprintf('Iter %03d, Function value %4.3f\n', k, gJfn);
                % Record and display iteration details
                tk = toc(tstart);
                res.itstat = [res.itstat;...
                    [k gather(gJfn) gather(gJdf) gather(gJl1) gather(grx) gather(gsx)...
                    gather(grd) gather(gsd) gather(geprix) gather(geduax) gather(geprid)...
                    gather(geduad) gather(grho) gather(gsigma) tk]];
                figure(6);
                plot(res.itstat(:,2));
                xlabel('Iterations');
                ylabel('Functional value');drawnow;


                %% Debug
                %G = gather(PzpT(gG));
                %figure(5);
                %tmp = squeeze(G(:,:,1,:));
                %imdisp(dict2img(tmp)); drawnow;

                %% Update D partially
                gD0 = PzpT(gG);
                % [~,idx] = sort(mean(mean(mean(gD0,1),2),3), 'descend');
                % gD0 = gD0(:,:,:,idx);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % end % End chunk
            %% Debug
            D0 = gather(gD0);
            % [~,idx] = sort(mean(mean(mean(D0,1),2),3), 'ascend');
            % D0 = D0(:,:,:,idx);
            % size(D0)
            figure(5);
            % imagesc(tiledict(gD0)); axis equal off; colormap gray; drawnow;
			% size(D0)
            % imagesc(dict2im(squeeze(D0(:,:,1,:)))); axis equal off; colormap gray; drawnow;
            imagesc(tiledict(squeeze(D0(:,:,1,:)))); axis equal off; colormap gray; drawnow;
        end % End for
        % imdisp(tiledict(squeeze(gD0))); axis equal off; colormap gray; drawnow;
        %% Update iterations
        k = k+1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end %% End main loop

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Collect the output
    gGY = ifft3(fft3(gG).*fft3(gY));
    gGS = ifft3(bsxfun(@times, fft3(gG), fft3(gS)));
    res.G  = gather(gG);  
    res.Y  = gather(gY);  
    res.GY = gather(gGY);  
    res.GS = gather(gGS);  
end