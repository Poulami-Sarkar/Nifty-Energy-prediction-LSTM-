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
* Gradient Desecent Algorithm
*/
void gradient_descent()
{
    float b_0 =14750,b_1=0.1,sqerr=0,lastsqerr=0,error,errorX,minerr=999999999,a_0,a_1;
    int epoch =0,i,y,y_pred;
    while(epoch<1000)
    {
        lastsqerr = sqerr;
        sqerr =0;
        error =0;
        errorX=0;
        //Sum of squared error SUM((y-y')^2) 
        for(i=0;i<29;i++)
        {
            y_pred = b_0+b_1*i;
            error = (y_pred - data[i][3]); 
            sqerr = pow(error,2);
            errorX += (error*i);
            error+=error;
        }
        sqerr /= 29;
        // b_0 = b_0 - A*(1/n)*SUM((y-y'))
        b_0 =b_0-A*error/29;
        // b_1 = b_1 - A*(1/n)*SUM((y-y')*x)
        b_1 -= A*errorX/29;
        epoch++;
        if(minerr > sqerr)
            minerr = sqerr;
            a_0 = b_0;
            a_1 = b_1;
        //printf("\nerror%d: %f",epoch,sqerr);
    }
    printf("\nMinimum error: %f",minerr);
    printf("\n%f %f %d",a_0,a_1,epoch);
    printf("\n Predicted value on 26th Feb: %f", a_0+a_1*29);
    printf("\n Predicted value on 1st March: %f", a_0+a_1*30);
    printf("\n Predicted value on 5st March: %f", a_0+a_1*34);
    printf("\n Actual value on 26th Feb: %f", data[29][3]);
    for (i=0;i<30;i++)
    {
        ;
        //printf("\n %f",a_0+a_1*i);
    }
}

int main()
{
    //printf("hello");
    int i,j;
    parse("data.csv");
    gradient_descent();
    return 0;
}