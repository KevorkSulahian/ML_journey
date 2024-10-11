#include <stdio.h>

int main() {
    int c;

    c = getchar();
    while (c != EOF) {
        putchar(c);
        c = getchar();
        // print EOF
        printf("%d\n", EOF);
    }
}