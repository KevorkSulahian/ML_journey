#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct arguments {
    char **files;
    unsigned int files_count;
} arguments;


void parse_arguments(int argc, char **argv, arguments *args) {
    
    args->files = malloc(argc * sizeof(char *));
    int index = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: ./main [file1] or [--help]\n");
            exit(0);
        } else {
            args->files[index] = argv[i];
            index++;
        }
    }

    args->files_count = index;
}

#define MAX_LEN 128

int read_file(char *path, char **buffer) {
    int tmp_capacity = MAX_LEN;

    char *tmp = malloc(tmp_capacity * sizeof(char));
    int tmp_size = 0;
    if (tmp == NULL) {
        perror("Memory allocation error");
        exit(1);
    }
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        perror("File opening error");
        exit(1);
    }
    int size = 0;

    do {

        if(tmp_size + MAX_LEN > tmp_capacity) {
            tmp_capacity *= 2;
            tmp = realloc(tmp, tmp_capacity * sizeof(char));
            if (tmp == NULL) {
                perror("Memory allocation error");
                exit(1);
            }
        }


        size = fread(tmp + tmp_size, sizeof(char), MAX_LEN, f);
        tmp_size += size;
    } while (size > 0);

    fclose(f);
    tmp[tmp_size] = '\0';
    *buffer = tmp;

    return tmp_size;
};

int main(int argc, char **argv) {
    arguments args = {0};
    parse_arguments(argc, argv, &args);

    char *buffer = NULL;
    int buffer_size = 0;
    for (int i = 0; i < args.files_count; i++) {
        char *content = NULL;
        int size = read_file(args.files[i], &content);

        buffer = realloc(buffer, buffer_size + size + 1);
        memcpy(buffer + buffer_size, content, size);
        buffer_size += size;

        free(content);
    }

    // int x = read_file(args.files[0], &buffer);
    printf("%s\n", buffer);
    free(buffer);
    free(args.files);   
    // printf("%d\n", x);
    
    return 0;
}

// how to compile with gcc
// gcc main.c -o main
// ./main