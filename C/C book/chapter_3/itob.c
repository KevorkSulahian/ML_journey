// Function to convert integer n to binary string representation
void reverse(char s[]);

void itob(int n, char s[]) {
    int i = 0;
    do {
        s[i++] = (n % 2) + '0'; // Convert remainder (0 or 1) to a character
        n /= 2;                // Divide n by 2
    } while (n > 0);
    s[i] = '\0';               // Null-terminate the string
    reverse(s);                // Reverse the string to get correct order
}

// Function to convert integer n to hexadecimal string representation
void itoh(int n, char s[]) {
    int i = 0;
    char hex[] = "0123456789abcdef"; // Hexadecimal characters
    do {
        s[i++] = hex[n % 16]; // Convert remainder to corresponding hex character
        n /= 16;              // Divide n by 16
    } while (n > 0);
    s[i] = '\0';              // Null-terminate the string
    reverse(s);               // Reverse the string to get correct order
}