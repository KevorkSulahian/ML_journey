#include <stdio.h>

int getch() { return getchar(); }
void ungetch(int c) { ungetc(c, stdin); }

int get_int(int *pn) {
  int c, sign;

  while ((c = getch()) == ' ' || c == '\n' || c == '\t')
    ;   /* skip white space */
  sign = 1;
  if (c == '+' || c == '-') { /* record sign */
    sign = (c == '+') ? 1 : -1;
    c = getch();
  }
  for (*pn = 0; c >= '0' && c <= '9'; c = getch())
    *pn = 10 * *pn + c - '0';
  *pn *= sign;
  if (c != EOF)
    ungetch(c);
  return c;
}

int main() {
    int num;
    printf("Enter an integer: ");
    get_int(&num);
    printf("You entered: %d\n", num);
    return 0;
}
