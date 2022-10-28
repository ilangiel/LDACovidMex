function corpus = generate_corpus(DATA, ATTR, FILE_NAME)

num_docs = size(DATA,1);
num_words = size(DATA,2);
corpus = cell(num_docs,1);

[~,CORPUS_NAME,~] = fileparts(FILE_NAME);

     for d=1:num_docs
         
         for w=1:num_words
             STR= char(DATA(d,w));
             SPACE = ' ';
             corpus{d} = [corpus{d},STR,SPACE];       
        end
     end
    

end
