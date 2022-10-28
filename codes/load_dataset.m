function [DATA, ATTR] = load_dataset(FILE_NAME, SELECTED_COLS)

if nargin<2, SELECTED_COLS=nan; end

TABLE = readtable(FILE_NAME);
ATTR = TABLE.Properties.VariableNames;

if isnan(SELECTED_COLS)
    DATA = TABLE{:,:};
else      
    DATA = TABLE{:,SELECTED_COLS};
    ATTR = ATTR(SELECTED_COLS);
end

end