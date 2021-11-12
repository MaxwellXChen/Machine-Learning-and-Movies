%% Import Data
T = readtable('IMDbmovies.csv');

%% Process Data
data = removevars(T,{'imdb_title_id','production_company','writer','actors','description','director','date_published'});
data = reduceColChar(data, data.usa_gross_income);
data = reduceColDoub(data, data.metascore);
data = reduceColChar(data, data.budget);
data = string2dollar(data, data.budget,'budget');
data = string2dollar(data, data.usa_gross_income,'usa_gross_income');
data = string2dollar(data, data.worlwide_gross_income,'worlwide_gross_income');

writetable(data,'movieData.csv')


%% Helper Functions

function [T] = reduceColChar(T, col)
%%reduces the number of rows based on whether or not they are empty.
count = 0;
ind = [];

for i = 1:length(col)
    notEmpty = ~isequal(col{i},'');
    if ~notEmpty
        ind = [ind i];
    end
    count = count + notEmpty;
end
T(ind,:) = [];
end

function [T] = reduceColDoub(T, col)
%%reduces the number of rows based on whether or not they are empty.
count = 0;
ind = [];

for i = 1:length(col)
    notEmpty = ~isnan(col(i));
    if ~notEmpty
        ind = [ind i];
    end
    count = count + notEmpty;
end
T(ind,:) = [];
end

function [T] = reduceColDatetime(T, col)
%%reduces the number of rows based on whether or not they are empty.
count = 0;
ind = [];

for i = 1:length(col)
    notEmpty = ~isnat(col(i));
    if ~notEmpty
        ind = [ind i];
    end
    count = count + notEmpty;
end
T(ind,:) = [];
end

function [T] = string2dollar(T, col, name)
%%converts string to dollar amount.
count = 0;
ind = [];

for i = 1:length(col)
    notDollar = ~isequal(col{i}(1),'$');
    if notDollar
        ind = [ind i];
    else
        temp = col{i}(3:end);
        col{i} = str2num(temp);
    end
    count = count + ~notDollar;
end
T{:,name} = col;
T(ind,:) = [];
end
