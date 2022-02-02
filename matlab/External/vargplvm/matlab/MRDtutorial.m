% MRDTUTORIAL Demonstration of MRD in various scenarios.
%
% COPYRIGHT: Andreas C. Damianou, 2013
%
% VARGPLVM

%-----------------    SIMPLE INITIAL DEMO -------------------------------%
%----- Main demo used: demToySvargplvm1.m

fprintf(1,['', ...
 '#####  Toy - basic demo: ####\n', ...
 'This is a simple demo which uses a toy dataset that considers two\n', ...
 'modalities: each of the modalities has one private 1-D signal and they\n', ...
 'both share a 1-D shared signal. The final dataset is mapped to 10-D. \n', ...
 'The demo allows training a MRD model so that the learned weights segment\n', ...
 'the output space and recover the true signals. \n', ...
 '\n',...
 'The demo demonstrates a basic options (many more are generally\n', ...
 'available):\n\n', ...
 'Experiment with different latent space initialisations, different kernels\n', ...
 '(linear, non-linear, etc) and with and without constraining the latent\n', ...
 'space with dynamics. Visualise the learned spaces and scales, see the\n', ...
 'difference in smoothness when dynamics are used/not used, compare the\n', ...
 'variational bounds (try as many opt. iterations as possible), etc. \n\n', ...
 'For possible options check ''svargplvm_init.m'' and the various demos.\n']);
 

fprintf(1,['\n\n', ...
 '# a)  Example 1 of running the demo:\n', ...
 '*** Non-dynamical case, initialise X separately for each modality and then concatenate:\n', ...
 '--------------\n', ...
 '>> experimentNo = 1;\n', ...
 '>> latentDimPerModel = 4;\n', ...
 '>> initial_X = ''separately'';\n', ...
 '>> demToySvargplvm1\n',...
 ' Press any key to start...\n\n']);
pause
clear; close all;
experimentNo = 1;
latentDimPerModel = 4;
initial_X = 'separately';
demToySvargplvm1;

%%
%{
%------- One-liner for the above:
% Get some toy data
Yall = util_createMRD_toy_data();
% kernel to use
baseKern = {{'linard2','white', 'bias'},{'linard2','white', 'bias'}}; 
% Learn MRD with some arguments.
[X,model] = MRDEmbed(Yall, 'initial_X','separately','latentDimPerModel',4,'baseKern',baseKern,'initVardistIters',200,'itNo',200);
%}


%%
fprintf(1,['\n\n', ...
'# b)   Example 2 of running the demo:\n', ...
'*** Non-dynamical case, initialise X once for the outputs that are first\n', ...
 'concatenated\n', ...
 '--------------\n', ...
 '>> experimentNo = 2;\n', ...
 '>> latentDim = 6;\n', ...
 '>> initial_X = ''concatenated''; \n', ...
 '>> demToySvargplvm1\n', ...
 ' Press any key to start...\n\n']);
pause
clear; close all;
experimentNo = 2;
latentDim = 6;
initial_X = 'concatenated'; 
demToySvargplvm1

%%
fprintf(1,['\n\n', ...
' # c)  Example 3 of running the demo:\n', ...
' *** Dynamics: \n', ...
' The above runs can be combined with dynamics (requires more iterations to converge): \n', ...
' >> experimentNo = 3;\n', ...
' >> initial_X = ''concatenated'';\n', ...
' >> initVardistIters = 600; \n', ...
' >> itNo = 1000;\n', ...
' >> dynamicKern = {''rbf'', ''white'', ''bias''}; \n', ...
' >> dynamicsConstrainType = {''time''};\n', ...
' >> demToySvargplvm1\n', ...
 ' Press any key to start...\n\n']);
pause
clear; close all;
experimentNo = 3;
initial_X = 'concatenated';
initVardistIters = 600;
itNo = 1000;
dynamicKern = {'rbf', 'white', 'bias'}; % SEE KERN toolbox for kernels\n', ...
dynamicsConstrainType = {'time'};
demToySvargplvm1

%%
fprintf(1,['\n\n', ...
 'For more options / variations, check/alter the variables in the first section \n', ...
 'the demo as well as the initialisation function svargplvm_init.m which\n', ...
 'lists all possible options.',...
 '\n###  END of Toy - basic demo: ####\n\n\n']);




%% ----------------- YALE FACES - pretrained ----------------------------%%
%----- Main demo used: demYaleDemonstration.m
fprintf(1,['', ...
 '#####  Yale faces demo - pretrained model: ####\n', ...
 'The Yale faces demo (see paper) separates the private\n', ...
 '(face characteristics) and shared (illumination condition) infocmation\n',...
 'from a set of high-dimensional images.\n\n',...
 'To start with, we will load a pre-trained model and explore the results.\n',...
 'To train the model yourself, see next section of this tutorial.\n',...
 ' >> demYaleDemonstration\n', ...
 ' Press any key to start...\n\n']);
pause
clear; close all;
demYaleDemonstration


%% ----------------- YALE FACES - train ----------------------------%%
%----- Main demo used: demYaleSvargplvm4.m
fprintf(1,['', ...
 '#####  Yale faces demo - training: ####\n', ...
 'This section shows how to train the model for the Yale faces demo\n',...
 'illustrated in the previous section.\n', ...
 'WARNING: This will take a LONG time..!\n',...
 ' Press any key to start...\n\n']);
pause

clear; close all;
latentDimPerModel = 7;
itNo = [1000 1000 1000 1000 1000]; 
mappingKern = 'rbfardjit';
DgtN = 1; % Use the 'smart' framework which is efficient when D >> N
initial_X = 'concatenated'; % See svargplvm_init.m for more options
indTr = [1:192]; % All data are training data
demYaleSvargplvm4



%% ----------------- Classification demo ----------------------------%%
%----- Main demo used: demOilSvargplvm2.m
fprintf(1,['', ...
 '#####  Classification demo on subsets of the oil data: ####\n', ...
 '\n The data are given as one modality and the training labels as another.\n',...
 '\n After training, we give the test data in one modality and ask the model\n', ...
 '\n to predict the class labels in the other modality. ',...
 ' Press any key to start...\n\n']);
pause
clear; close all;
demOilSvargplvm3