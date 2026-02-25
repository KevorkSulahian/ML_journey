#include <stdio.h>
#include <string.h>

struct date {
    int day;
    int month;
    int year;
    int yearday;
    char mon_name[4];
};

static int day_tab[2][13] = {
    {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
    {0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
};

int is_leap(int year) {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

int day_of_year(struct date *pd) {
    int i;
    int leap = is_leap(pd->year);

    int day = pd->day;
    for (i = 1; i < pd->month; i++)
        day += day_tab[leap][i];
    return day;
}

void month_day(struct date *pd) {
    int i;
    int leap = is_leap(pd->year);

    int day = pd->yearday;
    for (i = 1; day > day_tab[leap][i]; i++)
        day -= day_tab[leap][i];

    pd->month = i;
    pd->day = day;
}

int main() {
    struct date d;

    // Example 1: Convert date to day of year
    d.day = 24;
    d.month = 4;
    d.year = 2025;
    d.yearday = day_of_year(&d);
    printf("Date: %d-%02d-%02d -> Day of year: %d\n", d.year, d.month, d.day, d.yearday);

    // Example 2: Convert day of year back to date
    d.yearday = 100;
    d.year = 2025;
    month_day(&d);
    printf("Day of year: 100 in %d -> Date: %d-%02d-%02d\n", d.year, d.year, d.month, d.day);

    return 0;
}
