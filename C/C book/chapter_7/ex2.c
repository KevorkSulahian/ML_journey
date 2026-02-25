#include <stdio.h>

#define MAXLINE 1024

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s file1 file2\n", argv[0]);
        return 1;
    }

    FILE *fp1 = fopen(argv[1], "r");
    if (fp1 == NULL) {
        perror("Error opening file1");
        return 1;
    }

    FILE *fp2 = fopen(argv[2], "r");
    if (fp2 == NULL) {
        perror("Error opening file2");
        fclose(fp1);
        return 1;
    }

    int lineNum = 1; // Line number counter
    char line1[MAXLINE];
    char line2[MAXLINE];

    while (fgets(line1, MAXLINE, fp1) != NULL && fgets(line2, MAXLINE, 
fp2) != NULL) {
        if (line1[0] == '\n' && line2[0] != '\n') {
            printf("File2 is longer than file1 at line %d\n", lineNum);
            fclose(fp1);
            fclose(fp2);
            return 1;
        } else if (line2[0] == '\n' && line1[0] != '\n') {
            printf("File1 is longer than file2 at line %d\n", lineNum);
            fclose(fp1);
            fclose(fp2);
            return 1;
        }

        int i = 0;
        while (line1[i] == line2[i]) {
            ++i;
        }

        if (i < MAXLINE) { // Check if we've reached the end of a line
            printf("Files differ at line %d, character %d: '%c' vs. '%c'\n", lineNum, i + 1, line1[i], line2[i]);
            fclose(fp1);
            fclose(fp2);
            return 0;
        }
    }

    // If we've reached here, the files are equal up to the end of one or both
    if (fgets(line1, MAXLINE, fp1) != NULL || fgets(line2, MAXLINE, fp2) 
!= NULL) {
        printf("Files differ at line %d\n", lineNum);
    }

    fclose(fp1);
    fclose(fp2);

    return 0;
}
// This program compares two files line by line and character by character.