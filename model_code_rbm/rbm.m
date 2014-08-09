% Version 1.000 
%
% Code provided by Ruslan Salakhutdinov 
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This program trains Restricted Boltzmann Machine in which
% visible, binary, stochastic pixels are connected to
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.   
% The program assumes that the following variables are set externally:
% maxepoch  -- maximum number of epochs
% numhid    -- number of hidden units 
% batchdata -- the data that is divided into batches (numcases numdims numbatches)
% restart   -- set to 1 if learning starts from beginning 
% CD        -- number of CD steps.

% This is a slightly modified version of the program that was originally written 
% by Geoffrey Hinton and Ruslan Salakhutdinov.
function [vishid,visbiases,hidbiases,errors,learningTrack] = rbm(batchdata,opts,learningTrackSamples)
    CD = opts.CD;
    numhid = opts.numhid;
    maxepoch = opts.epochs_each*(2*numel(opts.etas)-1);
    randn('state',opts.randstate);
    rand('state',opts.randstate);

    anneal_lr = 0; 
    use_etas = zeros(maxepoch,1);
    for i=1:numel(opts.etas)-1
        use_etas(1+(i-1)*2*opts.epochs_each:(i-1)*2*opts.epochs_each+opts.epochs_each) = opts.etas(i);
        use_etas(1+(i-1)*2*opts.epochs_each+opts.epochs_each:i*2*opts.epochs_each) = linspace(opts.etas(i),opts.etas(i+1),opts.epochs_each);
    end
    use_etas(end-opts.epochs_each+1:end) = opts.etas(end);
        

    weightcost  = 0.0;  
    initialmomentum  = opts.momentum;
    finalmomentum    = opts.momentum;

    [numcases numdims numbatches]=size(batchdata);

    epoch=1;
    
    % Initializing symmetric weights and biases. 
    vishid     =  0.01*randn(numdims, numhid);
    hidbiases  = zeros(1,numhid);
    visbiases  = zeros(1,numdims);

    vishidinc  = zeros(numdims,numhid);
    hidbiasinc = zeros(1,numhid);
    visbiasinc = zeros(1,numdims);
    
    errors = zeros(maxepoch,1);
    if nargin > 2
        learningTrack = zeros(maxepoch/10,1);
    end
    for epoch = epoch:maxepoch
%      fprintf(1,'epoch %d\r',epoch); 

     epsilonw_0      = use_etas(epoch);   % Learning rate for weights 
     epsilonvb_0     = use_etas(epoch);   % Learning rate for biases of visible units 
     epsilonhb_0     = use_etas(epoch);   % Learning rate for biases of hidden units 
     if anneal_lr == 1 
       CD1 = ceil(epoch/3);  
       epsilonw = epsilonw_0/(1*CD1);
       epsilonvb = epsilonvb_0/(1*CD1);
       epsilonhb = epsilonhb_0/(1*CD1);
     else
       epsilonw = epsilonw_0;
       epsilonvb = epsilonvb_0;
       epsilonhb = epsilonhb_0;
     end 


     errsum=0;
     for batch = 1:numbatches,
       %fprintf(1,'epoch %d batch %d\r',epoch,batch); 

       visbias = repmat(visbiases,numcases,1);
       hidbias = repmat(hidbiases,numcases,1); 
       %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       data = batchdata(:,:,batch);
       %data = data > rand(numcases,numdims);  

       poshidprobs = 1./(1 + exp(-data*vishid - hidbias));    
       posprods    = data' * poshidprobs;
       poshidact   = sum(poshidprobs);
       posvisact = sum(data);

       %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       poshidprobs_temp = poshidprobs;

       %%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       for cditer=1:CD
         poshidstates = poshidprobs_temp > rand(numcases,numhid);
         negdata = 1./(1 + exp(-poshidstates*vishid' - visbias));
         negdata = negdata > rand(numcases,numdims); 
         poshidprobs_temp = 1./(1 + exp(-negdata*vishid - hidbias));
       end 
       neghidprobs = poshidprobs_temp;     

       negprods  = negdata'*neghidprobs;
       neghidact = sum(neghidprobs);
       negvisact = sum(negdata); 

       %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       err= sum(sum( (data-negdata).^2 ));
       errsum = err + errsum;

       if epoch>50,
         momentum=finalmomentum;
       else
         momentum=initialmomentum;
       end;

       %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        vishidinc = momentum*vishidinc + ...
                    epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
        visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
        hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);

        vishid = vishid + vishidinc;
        visbiases = visbiases + visbiasinc;
        hidbiases = hidbiases + hidbiasinc;

        %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
     end    
      errors(epoch) = errsum;
      if nargin > 2 && mod(epoch,10)==0
          logZ = calculate_true_partition(vishid,hidbiases,visbiases);
          loglik= calculate_logprob(vishid,hidbiases,visbiases,logZ,learningTrackSamples);
          learningTrack(epoch/10) = loglik.all;
      end
%       fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum); 
    end
end


