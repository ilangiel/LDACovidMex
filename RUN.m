clearvars; close all;
delete('RUN.txt'); delete('RUN.mat');  delete('*.png');

addpath(genpath('./codes'));
addpath(genpath('~/codes'));

diary('RUN.txt');
diary on;

% DATA FILES

FILE = '.../covid19Tratado2020EDOMEX_PAL_edad_VIVOS.csv';

SELECT_COLS = [2,3:16]; 

disp('1-DATASET PROCESSING');

disp(['Loading dataset: ', FILE]);
t = clock;
[ALL_DATA, ATTR] = load_dataset(FILE, SELECT_COLS);
dt = etime(clock,t); disp(['Elapsed time ', num2str(dt,'%.2f'), ' seconds.']); 


disp('Generating text corpus...');
t = clock;
corpus = generate_corpus(ALL_DATA, ATTR, FILE);
dt = etime(clock,t); disp(['Elapsed time ', num2str(dt,'%.2f'), ' seconds.']); 


disp('Generating documents...');
t = clock;
documents = tokenizedDocument(corpus);
dt = etime(clock,t); disp(['Elapsed time ', num2str(dt,'%.2f'), ' seconds.']); 


disp('Generating tra/val partitions...');
t = clock;
cvp = cvpartition(numel(documents),'HoldOut',0.1);
documentsTrain = documents(cvp.training);
documentsValidation = documents(cvp.test);
dt = etime(clock,t); disp(['Elapsed time ', num2str(dt,'%.2f'), ' seconds.']); 


disp('Generating tra bag-of-words...');
t = clock;
bag = bagOfWords(documentsTrain);
dt = etime(clock,t); disp(['Elapsed time ', num2str(dt,'%.2f'), ' seconds.']); 



disp('2-PERPLEXITY ANALISYS');
numTopicsRange = 5:15;
minPerplexity = inf;
for i = 1:numel(numTopicsRange)
    numTopics = numTopicsRange(i);
    
    disp(['Running LDA(', num2str(numTopics),')...']);
    
    t = clock;
    mdl = fitlda(bag,numTopics, ...
        'Solver','savb', ...
        'Verbose',0);
    dt = etime(clock,t); disp(['Elapsed time ', num2str(dt,'%.2f'), ' seconds.']);
    
    [~,validationPerplexity(i)] = logp(mdl,documentsValidation);
    timeElapsed(i) = mdl.FitInfo.History.TimeSinceStart(end);
    
    if validationPerplexity(i)<minPerplexity
        best_model = mdl;
        minPerplexity = validationPerplexity(i);
    end
    
end

disp('Saving perplexity plot...');
FIG_NAME = 'perplexity.png';
fig = figure;
set(fig,'visible','off');
set(gcf, 'Position',  [100, 100, 2048, 1024])
plot(numTopicsRange,validationPerplexity,'o-')
grid on
xlabel("Number of Topics")
ylabel("Validation Perplexity")
export_fig(FIG_NAME,'-png','-transparent');


disp('3-BEST NUMBER OF TOPICS');
disp('Using lowest perplexity model...');
[~,i] = min(validationPerplexity);
numTopics = numTopicsRange(i);
disp(['Running LDA(', num2str(numTopics),')...']);
if exist('best_model','var')    
    mdl = best_model;
    disp('Model loaded!');
else
    t = clock;
    mdl = fitlda(bag,numTopics, ...
        'Solver','savb', ...
        'Verbose',0);
    dt = etime(clock,t); disp(['Elapsed time ', num2str(dt,'%.2f'), ' seconds.']); 
end

disp('4-FILTERING TOPICS WITH HIGH ACTIVATIONS');
numTopics = mdl.NumTopics;
disp(numTopics);


NUM_TOP_WORDS = 5;

for i = 1:numTopics
    top = topkwords(mdl,NUM_TOP_WORDS,i);
    topWords(i) = join(top.Word,", ");
end

topWords = topWords(~cellfun(@isempty, topWords));
disp(topWords);


save('RUN.mat');

diary off;



