#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

char* fields[7]={"Date","Open","High","Low","Close","Shares Traded","Turnover"};
char* dates[30];
float data[30][6];
float A = 0.00001;

/*
* Removing unnessary characters such st " and spaces while parsing csv
*/
void removeChar(char *str, char garbage) {

    char *src, *dst;
    for (src = dst = str; *src != '\0'; src++) {
        *dst = *src;
        if (*dst != garbage) dst++;
    }
    *dst = '\0';
}

/*
* Function to parse csv
*/
void parse(char* file)
{
    FILE *f = fopen(file,"r");
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    int index =0,innerindex;
    if (f == NULL)
        printf("Error reading file");
    read = getline(&line, &len, f);
    while ((read = getline(&line, &len, f)) != -1) {
        removeChar(line,' ');
        removeChar(line,'"');
        char *pt;
        pt = strtok (line,",");
        pt = strtok (NULL, ",");
        dates[index] = pt;
        innerindex=0;
        while (pt != NULL) {
            float value = atof(pt);  
            data[index][innerindex] = value;
            pt = strtok (NULL, ",");
            innerindex++;
        }
        index++;
    }
}

/*
* Least Squares Algorithm
*/
void least_square()
{
    int i=0,j=0,n=29;
    float SumX=0, SumY=0, SumXY=0, SumXX=0,a,b;
    for(i=0;i<29;i++)
    {
        SumX += i;
        SumXX+= i*i;
        SumXY+= i*data[i][3];
        SumY += data[i][3];
    }
    a = (SumY*SumXX - SumX*SumXY)/(n*SumXX-SumX*SumX);
    b = (n*SumXY-SumX*SumY)/(n*SumXX-SumX*SumX);
    printf("\n%f %f",a,b);
    printf("\n Predicted value on 26th Feb: %f", a+b*29);
    printf("\n Predicted value on 1st March: %f", a+b*30);
    printf("\n Predicted value on 5st March: %f", a+b*34);
}

int main()
{
    parse("data.csv");
    least_square();
    return 0;

}
