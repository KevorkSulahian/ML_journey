#include <stdio.h>
#include <ctype.h>

#define TABSIZE 8
#define MAXLINE 1024

int main() {
    int c, i = 0;
    int len = 0; /* current line length */
    int maxline = MAXLINE; /* maximum allowed line length */
    char prevc = ' ';

    while ((c = getchar()) != EOF) {
        if (c == '\n') { // newline character
            ++len;
            if (len > maxline) {
                printf("\n");
                len = 0;
            }
            putchar(c);
        } else if (isspace(c)) { // space or tab character
            if (isspace(c) != isspace(prevc)) { // only print the first space or tab per line
                putchar(' ');
                ++len;
            }
        } else { // non-graphic character
            printf("\\%03o", c); // print in octal format
            ++len;
        }

        prevc = c; // update previous character for checking spaces/tabs
    }

    return 0;
}
