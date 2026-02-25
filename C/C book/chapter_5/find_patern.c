#include <stdio.h>
#include <string.h>

#define MAXLINE 1000  // Max line length

int get_line(char line[], int maxlen);

int main(int argc, char *argv[]) {
    char line[MAXLINE];

    if (argc != 2) {
        printf("Usage: find pattern\n");
        return 1;
    }

    while (get_line(line, MAXLINE) > 0)
        if (strstr(line, argv[1]) != NULL)  // strstr() finds substring
            printf("%s", line);

    return 0;
}

/* Read a line into 'line' and return length */
int get_line(char line[], int maxlen) {
    if (fgets(line, maxlen, stdin) == NULL)
        return 0;
    return strlen(line);
}
