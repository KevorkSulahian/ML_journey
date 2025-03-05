#include <stdio.h>

// Function to calculate pay with overtime consideration
void calcpay(float *p, float r, float h) {
    if (h > 40) {
        *p = (r * 40) + (r * (h - 40) * 1.5);  // Regular + Overtime pay
    } else {
        *p = r * h;  // Regular pay
    }
}

int main() {
    int empno;
    float rate, hours, pay;

    while (1) {
        // Read employee number, rate, and hours
        if (scanf("%d %f %f", &empno, &rate, &hours) < 3) 
            break; // Exit loop if input is incomplete

        // Calculate pay using call-by-location
        calcpay(&pay, rate, hours);

        // Print employee details
        printf("Employee=%d Rate=%.2f Hours=%.2f Pay=%.2f\n", empno, rate, hours, pay);
    }

    return 0; // Successful program termination
}
