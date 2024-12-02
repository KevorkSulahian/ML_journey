#include "stdio.h"

main () {
  float numbers[5];
  float total = 0.0;
  float average;
  int i;
  
  for (i = 0; i < 5; i++) {
    scanf("%f", &numbers[i]);
    total += numbers[i];
  }
  average = total/5;
  
  printf("The total is: %.1f\n", total);
  printf("The average is: %.1f\n", average);
}