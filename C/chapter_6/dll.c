#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXLINE 1000

struct lnode {
    char *text;
    struct lnode *prev;
    struct lnode *next;
};

int main() {
    struct lnode *head = NULL;
    struct lnode *tail = NULL;
    struct lnode *current;
    char line[MAXLINE];

    // Read lines into the doubly linked list
    while (fgets(line, MAXLINE, stdin) != NULL) {
        char *save = malloc(strlen(line) + 1);
        if (save == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
        strcpy(save, line);

        struct lnode *new = malloc(sizeof(struct lnode));
        if (new == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
        new->text = save;
        new->next = NULL;
        new->prev = tail;

        if (head == NULL) {
            head = new;
        }
        if (tail != NULL) {
            tail->next = new;
        }
        tail = new;
    }

    // Print lines in reverse
    for (current = tail; current != NULL; current = current->prev) {
        printf("%s", current->text);
    }

    // Free memory
    current = head;
    while (current != NULL) {
        struct lnode *next = current->next;
        free(current->text);
        free(current);
        current = next;
    }

    return 0;
}
