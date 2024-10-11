#include <stdio.h>

// print Celsius-Fahrenheit table from 0 to 100

int main()
{
    int lower, upper, step;
    float fahr, celsius;

    lower = 0; // lower limit of temperature table
    upper = 100; // upper limit
    step = 10; // step size

    celsius = lower;
    while (celsius <= upper) {
        fahr = (celsius * 9.0/5.0) + 32.0;
        printf("Temperature in Celsius and Fahrenheit\n");
        printf("%2.0f %3.0f\n", celsius, fahr);
        celsius = celsius + step;
    }
}