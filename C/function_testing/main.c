#include <stdio.h>
#include "math_functions.h" // Include the header file

int main() {
    int x = 5, y = 3;

    int sum = add(x, y);          // Call the add function
    int difference = subtract(x, y); // Call the subtract function

    printf("Sum: %d\n", sum);
    printf("Difference: %d\n", difference);

    return 0;
}
