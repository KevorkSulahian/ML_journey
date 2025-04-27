#include <stdio.h>
#include <stdlib.h>

struct lnode {
    int value;
    struct lnode *next;
};

struct list {
    struct lnode *head;
    struct lnode *tail;
};

void list_add(lst, value)
    struct list *lst;
    int value;
{
    struct lnode *newnode = malloc(sizeof(*newnode));
    if (!newnode) {
        fprintf(stderr, "Out of memory\n");
        exit(EXIT_FAILURE);
    }
    newnode->value = value;
    newnode->next = NULL;

    if (lst->head == NULL) {
        lst->head = lst->tail = newnode;
    } else {
        lst->tail->next = newnode;
        lst->tail = newnode;
    }
}

struct lnode * list_find(lst, value)
    struct list *lst;
    int value;
{
    struct lnode *cur;
    for (cur = lst->head; cur != NULL; cur = cur->next) {
        if (cur->value == value)
            return cur;
    }
    return NULL;
}

void list_dump(lst)
    struct list *lst;
{
    struct lnode *cur;
    printf("\nDump:\n");
    for (cur = lst->head; cur != NULL; cur = cur->next) {
        printf("  %d\n", cur->value);
    }
}

int main()
{
    void list_add();
    void list_dump();
    struct lnode * list_find();

    struct list mylist;
    struct lnode * mynode;

    mylist.head = NULL;
    mylist.tail = NULL;

    list_add(&mylist, 10);
    list_add(&mylist, 20);
    list_add(&mylist, 30);
    list_dump(&mylist);


    list_add(&mylist, 40);
    list_dump(&mylist);

}
